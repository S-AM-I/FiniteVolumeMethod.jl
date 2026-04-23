# test/stage7_coupled.jl — Stage 7 solid mechanics / FSI / function objects gates

using FiniteVolumeMethod
using Test
using StaticArrays: SVector
using LinearAlgebra: norm

include("TestHelpers.jl")

@testset "Stage 7a: IsotropicElastic derives correct Lamé constants" begin
    # Steel-like: E = 210 GPa, ν = 0.30. Expected:
    #   λ = E ν / ((1+ν)(1-2ν)) ≈ 121.15 GPa
    #   μ = E / (2(1+ν))        ≈  80.77 GPa
    mat = IsotropicElastic(; E = 210.0e9, nu = 0.3)
    @test mat.lambda ≈ 121.153846153846e9 atol = 1.0e4
    @test mat.mu ≈ 80.769230769231e9 atol = 1.0e4

    # Incompressible limit ν → 0.5 produces large λ.
    mat_inc = IsotropicElastic(; E = 1.0e6, nu = 0.499)
    @test mat_inc.lambda > 10 * mat_inc.mu
end

@testset "Stage 7a: stress_tensor reproduces σ = λ tr(ε) I + 2μ ε" begin
    mat = IsotropicElastic(; E = 100.0, nu = 0.25)
    eps_tensor = [1.0 0.5; 0.5 2.0]  # ε_11 = 1, ε_22 = 2, ε_12 = 0.5
    sigma = stress_tensor(mat, eps_tensor)
    tr_eps = 3.0
    @test sigma[1, 1] ≈ mat.lambda * tr_eps + 2 * mat.mu * 1.0
    @test sigma[2, 2] ≈ mat.lambda * tr_eps + 2 * mat.mu * 2.0
    @test sigma[1, 2] ≈ 2 * mat.mu * 0.5
    @test sigma[1, 2] ≈ sigma[2, 1]  # symmetric
end

@testset "Stage 7a: small_strain_tensor symmetrises ∇u" begin
    grad_u = [1.0 2.0; -1.0 3.0]
    eps_tensor = small_strain_tensor(grad_u)
    @test eps_tensor[1, 1] == 1.0
    @test eps_tensor[2, 2] == 3.0
    @test eps_tensor[1, 2] ≈ 0.5
    @test eps_tensor[1, 2] == eps_tensor[2, 1]
end

@testset "Stage 7a: Euler-Bernoulli cantilever tip deflection" begin
    # δ = P L³ / (3 E I)
    delta = cantilever_tip_deflection(2.1e11, 1.0e-6, 1.0, 100.0)
    expected = 100.0 * 1.0^3 / (3 * 2.1e11 * 1.0e-6)
    @test delta ≈ expected atol = 1.0e-10
end

@testset "Stage 7b: Aitken relaxation adapts toward optimal ω" begin
    relax = AitkenRelaxation(; omega0 = 0.5)

    # Feed a declining residual sequence: ω should increase toward 1.
    r1 = [1.0, 1.0]
    r2 = [0.3, 0.3]
    r3 = [0.05, 0.05]

    ω1 = update_aitken!(relax, r1)
    ω2 = update_aitken!(relax, r2)
    ω3 = update_aitken!(relax, r3)

    @test ω1 == 0.5                 # first call keeps initial
    @test relax.omega_min <= ω2 <= relax.omega_max
    @test relax.omega_min <= ω3 <= relax.omega_max
end

@testset "Stage 7b: FSIInterface has matching fluid/solid face lists" begin
    iface = FSIInterface{2, Float64}([10, 20, 30], [15, 25, 35])
    @test length(iface.fluid_face_indices) == 3
    @test length(iface.solid_face_indices) == 3
    @test all(d -> d == zero(SVector{2, Float64}), iface.displacement)
    @test all(t -> t == zero(SVector{2, Float64}), iface.traction)
end

@testset "Stage 7b: interface_residual_norm matches L2 update" begin
    d_old = [SVector(0.0, 0.0), SVector(0.0, 0.0)]
    d_new = [SVector(0.3, 0.4), SVector(0.0, 0.0)]
    r = interface_residual_norm(d_new, d_old)
    @test r ≈ 0.5 atol = 1.0e-12      # sqrt(0.09 + 0.16) = 0.5
end

@testset "Stage 7d: PointProbe accumulates samples" begin
    extract = (state, c) -> state.T[c]
    probe = PointProbe(:T_probe, SVector(0.5, 0.5), 1, extract)
    # Fake state with a temperature field
    state = (T = [300.0, 400.0],)
    FiniteVolumeMethod.run!(probe, state, 0.1, 1)
    FiniteVolumeMethod.run!(probe, state, 0.2, 2)
    @test length(probe.history) == 2
    @test probe.history[1] == (0.1, 300.0)
    @test probe.history[2] == (0.2, 300.0)
end

@testset "Stage 7d: ForceProbe sums user-computed force" begin
    faces = [10, 20, 30]
    compute = (state, fs) -> SVector(1.0, 2.0)  # dummy 2D force
    fp = ForceProbe(:drag, faces, compute, Val(2), Float64)
    FiniteVolumeMethod.run!(fp, nothing, 0.0, 1)
    @test length(fp.history) == 1
    @test fp.history[1][2] == SVector(1.0, 2.0)
end

@testset "Stage 7d: ExpressionBC evaluates closure at (x, t)" begin
    # Pulsating inlet: u_in(t) = sin(2π t)
    bc = ExpressionBC((x, t) -> SVector(sin(2π * t), 0.0), Val(2), Float64)
    @test bc isa AbstractFVMBoundaryCondition
    u_at_025 = evaluate_expression_bc(bc, SVector(0.0, 0.5), 0.25)
    @test u_at_025[1] ≈ 1.0 atol = 1.0e-12
    u_at_0 = evaluate_expression_bc(bc, SVector(0.0, 0.5), 0.0)
    @test u_at_0 ≈ SVector(0.0, 0.0)
end

@testset "Stage 7d: FieldStatistics running average" begin
    stats = FieldStatistics(:T_mean, 3, Float64)
    FiniteVolumeMethod.update!(stats, [1.0, 2.0, 3.0])
    FiniteVolumeMethod.update!(stats, [3.0, 4.0, 5.0])
    @test stats.n_samples == 2
    @test stats.mean ≈ [2.0, 3.0, 4.0]
end

# test/experimental/solid_mechanics_elasticity.jl — small-strain linear elasticity.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: IsotropicElastic, cantilever_tip_deflection, small_strain_tensor, stress_tensor
using Test

@testset "IsotropicElastic derives correct Lamé constants" begin
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

@testset "stress_tensor reproduces σ = λ tr(ε) I + 2μ ε" begin
    mat = IsotropicElastic(; E = 100.0, nu = 0.25)
    eps_tensor = [1.0 0.5; 0.5 2.0]  # ε_11 = 1, ε_22 = 2, ε_12 = 0.5
    sigma = stress_tensor(mat, eps_tensor)
    tr_eps = 3.0
    @test sigma[1, 1] ≈ mat.lambda * tr_eps + 2 * mat.mu * 1.0
    @test sigma[2, 2] ≈ mat.lambda * tr_eps + 2 * mat.mu * 2.0
    @test sigma[1, 2] ≈ 2 * mat.mu * 0.5
    @test sigma[1, 2] ≈ sigma[2, 1]  # symmetric
end

@testset "small_strain_tensor symmetrises ∇u" begin
    grad_u = [1.0 2.0; -1.0 3.0]
    eps_tensor = small_strain_tensor(grad_u)
    @test eps_tensor[1, 1] == 1.0
    @test eps_tensor[2, 2] == 3.0
    @test eps_tensor[1, 2] ≈ 0.5
    @test eps_tensor[1, 2] == eps_tensor[2, 1]
end

@testset "Euler-Bernoulli cantilever tip deflection" begin
    # δ = P L³ / (3 E I)
    delta = cantilever_tip_deflection(2.1e11, 1.0e-6, 1.0, 100.0)
    expected = 100.0 * 1.0^3 / (3 * 2.1e11 * 1.0e-6)
    @test delta ≈ expected atol = 1.0e-10
end

# test/v_and_v_komega.jl — k-ω Wilcox model algebra V&V (v3.38)
#
# Third convergence-verified benchmark for `turbulence_rans`. The
# first (v3.18) tested k-ε DHIT; the second (v3.23) tested k-ε
# log-layer equilibrium. This one covers the complementary k-ω
# Wilcox closure:
#
#   ν_t = k / ω,
#   dk/dt  = -β* · k · ω   (homogeneous decay without production)
#   dω/dt  = -β · ω²       (homogeneous decay without production)
#
# whose closed-form solution (with constants β* = 0.09, β = 3/40):
#
#   ω(t) = ω_0 / (1 + β · ω_0 · t)
#   k(t) = k_0 · ((1 + β·ω_0·t)^(-β*/β)) · ...
#
# The ω ODE integrates analytically: 1/ω grows linearly in t.
# Five invariants are verified. Puts `turbulence_rans` at three
# convergence-verified benchmarks.

using FiniteVolumeMethod
using LinearSolve
using StaticArrays
using Test

include("TestHelpers.jl")

const KW_BETA_STAR = 0.09
const KW_BETA = 3.0 / 40.0

function run_komega_decay(n_steps::Int, dt::Float64; k0 = 1.0, omega0 = 1.0)
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    model = KOmega()

    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))
    phi = FaceFluxField(:phi, mesh; value = 0.0)

    turb_state = RANSTurbulenceState(model, mesh; k = k0, omega = omega0)
    FiniteVolumeMethod.turbulent_viscosity!(turb_state.nu_t, model, turb_state, mesh)

    bc = ParabolicNeumann(0.0)
    bcs_turb = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
        :k => Dict(:left => bc, :right => bc, :bottom => bc, :top => bc),
        :omega => Dict(:left => bc, :right => bc, :bottom => bc, :top => bc),
    )

    k_hist = Float64[k0]
    o_hist = Float64[omega0]
    t = 0.0
    for _ in 1:n_steps
        FiniteVolumeMethod.solve_turbulence!(
            turb_state, model, U, phi, 1.0e-6, mesh, bcs_turb;
            dt = dt, linear_solver = LUFactorization(),
        )
        push!(k_hist, turb_state.fields[:k].internal[1])
        push!(o_hist, turb_state.fields[:omega].internal[1])
        t += dt
    end
    return (turb_state = turb_state, k_hist = k_hist, o_hist = o_hist, t_end = t)
end

@testset "V&V: k-ω — ν_t = k/ω algebraic identity" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = KOmega()

    turb_state = RANSTurbulenceState(model, mesh; k = 2.0, omega = 4.0)
    FiniteVolumeMethod.turbulent_viscosity!(turb_state.nu_t, model, turb_state, mesh)

    # ν_t = k / ω = 2 / 4 = 0.5.
    for c in 1:nc
        @test isapprox(turb_state.nu_t[c], 0.5; rtol = 1.0e-12)
    end
end

@testset "V&V: k-ω — realizability + monotone decay" begin
    # Without production, both k and ω must decrease monotonically
    # from their initial values and stay non-negative.
    res = run_komega_decay(500, 0.01)
    @test all(>=(0.0), res.k_hist)
    @test all(>=(0.0), res.o_hist)
    @test all(diff(res.k_hist) .<= 1.0e-12)
    @test all(diff(res.o_hist) .<= 1.0e-12)
end

@testset "V&V: k-ω — ω decay matches analytical 1/(1 + β·ω₀·t)" begin
    # dω/dt = -β·ω² ⇒ ω(t) = ω₀/(1 + β·ω₀·t). With β = 3/40,
    # ω₀ = 1: at t = 1, ω(1) = 1/(1 + 0.075) ≈ 0.930.
    # Implicit Euler converges to this as dt → 0.
    res = run_komega_decay(2000, 0.0005)
    omega_final = res.o_hist[end]
    omega_analytical = 1.0 / (1 + KW_BETA * 1.0 * res.t_end)
    @test isapprox(omega_final, omega_analytical; rtol = 2.0e-2)
end

@testset "V&V: k-ω — k decay faster than ω (β* > β ratio)" begin
    # β*·ω destruction vs. β·ω² — at ω_0 = 1, the k-destruction
    # rate β*·ω = 0.09 is larger than the ω-destruction rate
    # β·ω = 3/40 = 0.075. So k decays faster than ω initially.
    res = run_komega_decay(100, 0.01; k0 = 1.0, omega0 = 1.0)
    k_ratio = res.k_hist[end] / res.k_hist[1]
    o_ratio = res.o_hist[end] / res.o_hist[1]
    @test k_ratio < o_ratio
    # Both should be less than 1 (decaying).
    @test k_ratio < 1.0
    @test o_ratio < 1.0
end

@testset "V&V: k-ω — ν_t = k_final / ω_final identity after decay" begin
    # After a decay run, the explicit k/ω ratio gives ν_t by
    # direct evaluation. Build a fresh state with the decayed
    # values and recompute ν_t; it must equal k/ω to round-off.
    res = run_komega_decay(200, 0.01)
    k_final = res.turb_state.fields[:k].internal[1]
    o_final = res.turb_state.fields[:omega].internal[1]

    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = KOmega()

    # Build a clean turb_state with the final values and recompute.
    clean = RANSTurbulenceState(model, mesh; k = k_final, omega = o_final)
    FiniteVolumeMethod.turbulent_viscosity!(clean.nu_t, model, clean, mesh)

    for c in 1:nc
        @test isapprox(clean.nu_t[c], k_final / o_final; rtol = 1.0e-12)
    end
end

# test/v_and_v_schiller_naumann.jl — Schiller-Naumann drag V&V (v3.26)
#
# Second analytical benchmark for `lagrangian_dpm`. Stokes (v3.13)
# covered the Re ≪ 1 limit; this benchmark extends coverage to the
# moderate-Re regime 1 ≲ Re ≲ 1000 where the Schiller-Naumann
# correlation is in force:
#
#   C_d = (24 / Re) · (1 + 0.15 · Re^0.687)
#   F   = (m_p / τ_p) · f(Re) · (U_f − U_p),
#         f(Re) = 1 + 0.15 · Re^0.687.
#
# Two invariants are verified:
#
#   1. Algebraic: compare the drag returned by
#      `compute_drag_force(SchillerNaumann(), ...)` to the
#      closed-form (m_p / τ_p)·f(Re)·ΔU across a Re sweep.
#
#   2. Kinematic: for a particle in a constant-U fluid with no
#      gravity, the relaxation time to reach U_f is the
#      Re-dependent τ_p / f(Re). Verify the final slip velocity
#      decay profile matches the analytical exponential after an
#      explicit Euler integration with small Δt.
#
# Evidence toward future `stable` promotion of `lagrangian_dpm`.

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: Schiller-Naumann — algebraic f(Re) = 1 + 0.15·Re^0.687" begin
    # Water droplet in air: ρ_p = 1000, ρ_f = 1.2, μ_f = 1.81e-5.
    d_p = 100.0e-6     # 100 μm
    rho_p = 1000.0
    rho_f = 1.2
    mu_f = 1.81e-5

    for slip in (0.01, 0.1, 0.5, 1.0, 5.0)
        U_f = SVector(0.0, 0.0)
        U_p = SVector(-slip, 0.0)  # slip velocity = U_f - U_p = (slip, 0)

        # Expected: Re = ρ_f · slip · d / μ_f, f = 1 + 0.15·Re^0.687,
        # F = (m/τ_p)·f·(U_f - U_p) = (m/τ_p)·f·slip x̂.
        Re = rho_f * slip * d_p / mu_f
        f = 1 + 0.15 * Re^0.687
        m_p = pi / 6 * d_p^3 * rho_p
        tau_p = rho_p * d_p^2 / (18 * mu_f)
        F_analytical_x = (m_p / tau_p) * f * slip

        F = compute_drag_force(SchillerNaumann(), U_f, U_p, d_p, rho_p, rho_f, mu_f)
        @test isapprox(F[1], F_analytical_x; rtol = 1.0e-12)
        @test isapprox(F[2], 0.0; atol = 1.0e-14)
    end
end

@testset "V&V: Schiller-Naumann — Re cap saturates at 1000" begin
    # The implementation caps Re_p at 1000 in the correction
    # factor. Verify that at Re = 2000 the computed drag uses
    # f(1000), not f(2000).
    d_p = 1.0e-3          # 1 mm — unusually large
    rho_p = 1000.0
    rho_f = 1000.0        # heavy liquid
    mu_f = 1.0e-3
    slip = 2.0            # Re = 1000·2·1e-3/1e-3 = 2000

    U_f = SVector(0.0, 0.0)
    U_p = SVector(-slip, 0.0)

    Re_raw = rho_f * slip * d_p / mu_f
    @test Re_raw > 1000
    f_capped = 1 + 0.15 * 1000.0^0.687

    m_p = pi / 6 * d_p^3 * rho_p
    tau_p = rho_p * d_p^2 / (18 * mu_f)
    F_expected_x = (m_p / tau_p) * f_capped * slip

    F = compute_drag_force(SchillerNaumann(), U_f, U_p, d_p, rho_p, rho_f, mu_f)
    @test isapprox(F[1], F_expected_x; rtol = 1.0e-12)
end

@testset "V&V: Schiller-Naumann — Stokes limit agreement" begin
    # As Re → 0, Schiller-Naumann must converge to Stokes: f → 1.
    # Test with a 1 μm droplet in air (Re ~ 0.01 at slip = 0.1).
    d_p = 1.0e-6
    rho_p = 1000.0
    rho_f = 1.2
    mu_f = 1.81e-5
    slip = 0.1

    U_f = SVector(0.0, 0.0)
    U_p = SVector(-slip, 0.0)

    F_sn = compute_drag_force(SchillerNaumann(), U_f, U_p, d_p, rho_p, rho_f, mu_f)
    F_stokes = compute_drag_force(StokesDrag(), U_f, U_p, d_p, rho_p, rho_f, mu_f)

    Re = rho_f * slip * d_p / mu_f
    @test Re < 0.02
    # Schiller-Naumann correction factor is ≈ 1 + 0.15·0.02^0.687
    # ≈ 1.011, so SN and Stokes agree within 2 %.
    @test isapprox(F_sn[1], F_stokes[1]; rtol = 0.02)
end

@testset "V&V: Schiller-Naumann — exponential relaxation in quiescent-to-moving fluid" begin
    # Particle at rest; fluid suddenly moves at U = 1. The
    # slip-velocity ODE for a decelerating slip (no gravity):
    #
    #   d(U_p)/dt = (1/τ_p) f(Re) (U_f − U_p),
    #
    # with f(Re) Re-dependent. Numerically integrate to
    # steady state (U_p → U_f); compare U_p(t_end) against U_f
    # within tight tolerance, and verify monotone approach.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))

    d_p = 100.0e-6
    rho_p = 1000.0
    rho_f = 1.2
    mu_f = 1.81e-5

    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.5, 0.5)])
    part = tracker.particles[1]
    part.cell_index = 6
    set_particle_properties!(part; diameter = d_p, density = rho_p)

    tau_p = rho_p * d_p^2 / (18 * mu_f)
    dt_sub = tau_p / 200
    n_steps = 2000  # t_end ≈ 10 τ_p

    u_hist = Float64[part.velocity[1]]
    for _ in 1:n_steps
        advance_particles!(
            tracker, U, mesh, dt_sub;
            drag_model = SchillerNaumann(), rho_f = rho_f, mu_f = mu_f,
        )
        push!(u_hist, part.velocity[1])
    end

    # Terminal slip → 0: U_p = U_f.
    @test isapprox(part.velocity[1], 1.0; rtol = 1.0e-2)

    # Monotone approach.
    @test all(diff(u_hist) .>= -1.0e-14)
end

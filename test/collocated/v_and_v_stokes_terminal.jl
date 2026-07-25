# test/v_and_v_stokes_terminal.jl — Stokes terminal-velocity V&V (v3.13)
#
# Verifies `advance_particles!` + `StokesDrag` against the exact
# analytical solution of a single-particle settling ODE
#
#   m_p · dv/dt = F_drag + m_p · g,   F_drag = (m_p / τ_p) (U_f - v)
#
# For quiescent fluid (U_f = 0):
#
#   dv/dt = g - v / τ_p,      τ_p = ρ_p · d² / (18 μ_f)
#
# Closed-form solution with v(0) = 0:
#
#   v(t) = v_t · (1 - exp(-t / τ_p)),   v_t = g · τ_p
#
# The forward-Euler solver in `advance_particles!` approximates this
# as v_{n+1} = (1 - Δt/τ_p) v_n + Δt g, whose exact discrete fixed
# point is v_t (matches analytical at steady state) and which
# converges to the analytical response as Δt/τ_p → 0 at first
# order. Evidence for promoting `lagrangian_dpm` from
# `experimental`/`smoke_tested` to `provisional`/`convergence_verified`.

using FiniteVolumeMethod
using LinearAlgebra: norm
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

# Air at STP, water droplet sized to keep Re_p ≪ 1 (Stokes regime).
const STOKES_PARAMS = (
    d_p = 10.0e-6,       # 10 μm droplet
    rho_p = 1.0e3,       # water
    rho_f = 1.2,         # air
    mu_f = 1.81e-5,      # air
    g = 9.81,
)

function stokes_analytical()
    p = STOKES_PARAMS
    tau_p = p.rho_p * p.d_p^2 / (18 * p.mu_f)
    v_t = p.g * tau_p
    return (tau_p = tau_p, v_t = v_t)
end

function run_settling(n_steps::Int, dt_over_tau::Float64)
    p = STOKES_PARAMS
    an = stokes_analytical()
    dt = dt_over_tau * an.tau_p

    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))

    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.5, 0.5)])
    part = tracker.particles[1]
    part.cell_index = 6
    set_particle_properties!(part; diameter = p.d_p, density = p.rho_p)

    gravity = SVector(0.0, -p.g)

    # Record vertical velocity at each fluid step.
    v_y = zeros(Float64, n_steps + 1)
    v_y[1] = part.velocity[2]
    for n in 1:n_steps
        advance_particles!(
            tracker, U, mesh, dt;
            drag_model = StokesDrag(), rho_f = p.rho_f, mu_f = p.mu_f,
            gravity = gravity,
        )
        v_y[n + 1] = part.velocity[2]
    end
    return (v_y = v_y, dt = dt, part = part, tau_p = an.tau_p, v_t = an.v_t)
end

@testset "V&V: Stokes terminal velocity — steady-state asymptote" begin
    # Δt/τ_p = 0.01 is well inside the Euler stability + accuracy
    # regime; run out to 5 τ_p (v should reach 99.3% of terminal).
    res = run_settling(500, 0.01)

    v_final = -res.v_y[end]         # downward is negative y
    rel_err = abs(v_final - res.v_t * (1 - exp(-5.0))) / res.v_t
    @test rel_err < 1.0e-2

    # No lateral drift — particle settles straight down.
    @test res.part.position[1] ≈ 0.5 atol = 1.0e-14
    @test res.part.velocity[1] ≈ 0.0 atol = 1.0e-14

    # Downward progress is monotone and bounded by v_t.
    @test all(diff(res.v_y) .<= 0.0)
    @test all(-res.v_y[2:end] .<= res.v_t + 1.0e-12)
end

@testset "V&V: Stokes terminal velocity — mid-transient accuracy" begin
    # With Δt/τ_p = 0.01, step n=100 corresponds to t = τ_p, where
    # v(τ_p) / v_t = 1 - e^{-1} ≈ 0.6321. The forward-Euler
    # approximant is 1 - (1 - Δt/τ_p)^n = 0.6340. Both agree to <0.5%.
    res = run_settling(100, 0.01)

    v_mid = -res.v_y[end]
    target = res.v_t * (1 - exp(-1.0))
    rel_err = abs(v_mid - target) / target
    @test rel_err < 2.0e-2
end

@testset "V&V: Stokes terminal velocity — Euler first-order in Δt" begin
    # The discrete fixed point is exact (both equal g τ_p), so the
    # asymptote error is Δt-independent. Instead, measure the Euler
    # truncation error at a finite horizon t = τ_p. Error decays
    # like Δt (first-order Euler on a linear ODE).
    an = stokes_analytical()
    target = an.v_t * (1 - exp(-1.0))

    errors = Float64[]
    for dt_over_tau in (0.04, 0.02, 0.01)
        n_steps = round(Int, 1.0 / dt_over_tau)
        res = run_settling(n_steps, dt_over_tau)
        push!(errors, abs(-res.v_y[end] - target))
    end

    # Rate between the first two and last two halvings.
    orders = [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]

    # Forward Euler on a stiff-free linear ODE: p ≈ 1. Floating
    # point slack + finite-horizon saturation permits 0.8 < p < 1.3.
    for p in orders
        @test 0.8 < p < 1.3
    end

    # Monotone error decrease.
    @test errors[1] > errors[2] > errors[3]
end

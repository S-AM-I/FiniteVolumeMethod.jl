# # Stokes Terminal Velocity
# This case verifies `advance_particles!` with `StokesDrag` against the
# exact solution of the single-particle settling ODE in quiescent fluid:
# ```math
# \frac{\mathrm{d}v}{\mathrm{d}t} = g - \frac{v}{\tau_p}, \qquad
# \tau_p = \frac{\rho_p d_p^2}{18 \mu_f}, \qquad
# v(t) = v_t \left(1 - e^{-t/\tau_p}\right), \quad v_t = g \tau_p.
# ```
# A 10 μm water droplet in air keeps $Re_p \ll 1$ (Stokes regime). The
# forward-Euler update has $v_t$ as its exact discrete fixed point and
# converges to the transient response at first order in $\Delta t$.
#
# ## Acceptance Gates
# - Steady asymptote at $t = 5\tau_p$ within 1% of
#   $v_t(1 - e^{-5})$; monotone approach bounded by $v_t$
# - Mid-transient at $t = \tau_p$ within 2% of $v_t(1 - e^{-1})$
# - First-order $\Delta t$ convergence: observed orders in $(0.8, 1.3)$
# - No lateral drift (position and velocity to $10^{-14}$)

using FiniteVolumeMethod
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

d_p = 10.0e-6
rho_p = 1.0e3
rho_f = 1.2
mu_f = 1.81e-5
g = 9.81

tau_p = rho_p * d_p^2 / (18 * mu_f)
v_t = g * tau_p

# ## Settling Run
function run_settling(n_steps, dt_over_tau)
    dt = dt_over_tau * tau_p
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))

    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.5, 0.5)])
    part = tracker.particles[1]
    part.cell_index = 6
    set_particle_properties!(part; diameter = d_p, density = rho_p)

    v_y = zeros(Float64, n_steps + 1)
    for n in 1:n_steps
        advance_particles!(
            tracker, U, mesh, dt;
            drag_model = StokesDrag(), rho_f = rho_f, mu_f = mu_f,
            gravity = SVector(0.0, -g),
        )
        v_y[n + 1] = part.velocity[2]
    end
    return (v_y = v_y, dt = dt, part = part)
end

# ## Steady Asymptote and Transient
res_long = run_settling(500, 0.01)
asymptote_error = abs(-res_long.v_y[end] - v_t * (1 - exp(-5.0))) / v_t
monotone = all(diff(res_long.v_y) .<= 0.0)
bounded = all(-res_long.v_y[2:end] .<= v_t + 1.0e-12)
lateral_pos = abs(res_long.part.position[1] - 0.5)
lateral_vel = abs(res_long.part.velocity[1])

res_mid = run_settling(100, 0.01)
mid_target = v_t * (1 - exp(-1.0))
mid_error = abs(-res_mid.v_y[end] - mid_target) / mid_target

# ## Δt Convergence at Fixed Horizon t = τ_p
dt_ratios = (0.04, 0.02, 0.01)
dt_errors = map(dt_ratios) do dt_over_tau
    res = run_settling(round(Int, 1.0 / dt_over_tau), dt_over_tau)
    abs(-res.v_y[end] - mid_target)
end
dt_orders = [log2(dt_errors[i] / dt_errors[i + 1]) for i in 1:(length(dt_errors) - 1)]

# ## Visualisation — Settling Curve and Convergence
t_hist = range(0.0, 5.0; length = length(res_long.v_y))
t_dense = range(0.0, 5.0; length = 300)

fig1 = Figure(fontsize = 24, size = (1000, 450))
ax1 = Axis(
    fig1[1, 1], xlabel = "t / τ_p", ylabel = "v / v_t",
    title = "Stokes settling"
)
lines!(
    ax1, t_dense, 1 .- exp.(-t_dense), color = :black, linewidth = 2,
    label = "Exact"
)
scatter!(
    ax1, t_hist[1:20:end], -res_long.v_y[1:20:end] ./ v_t, color = :red,
    markersize = 10, label = "Forward Euler (Δt = 0.01 τ_p)"
)
axislegend(ax1, position = :rb)
ax2 = Axis(
    fig1[1, 2], xlabel = "Δt / τ_p", ylabel = "|v(τ_p) − exact|",
    xscale = log10, yscale = log10, title = "Δt convergence"
)
scatterlines!(
    ax2, collect(dt_ratios), collect(dt_errors), marker = :circle,
    color = :blue, linewidth = 2, markersize = 12
)
lines!(
    ax2, collect(dt_ratios), dt_errors[1] .* (collect(dt_ratios) ./ dt_ratios[1]),
    color = :gray, linestyle = :dash, linewidth = 1.5
)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("stokes_terminal_velocity.png"), fig1)
end

# ## Acceptance
@test asymptote_error < 1.0e-2 #src
@test monotone #src
@test bounded #src
@test lateral_pos < 1.0e-14 #src
@test lateral_vel < 1.0e-14 #src
@test mid_error < 2.0e-2 #src
for p in dt_orders #src
    @test 0.8 < p < 1.3 #src
end #src
@test dt_errors[1] > dt_errors[2] > dt_errors[3] #src
@assert asymptote_error < 1.0e-2 #hide
@assert monotone #hide
@assert bounded #hide
@assert lateral_pos < 1.0e-14 #hide
@assert lateral_vel < 1.0e-14 #hide
@assert mid_error < 2.0e-2 #hide
@assert all(p -> 0.8 < p < 1.3, dt_orders) #hide
@assert dt_errors[1] > dt_errors[2] > dt_errors[3] #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "asymptote_relative_error" => asymptote_error,
            "mid_transient_relative_error" => mid_error,
            "dt_orders" => dt_orders,
            "terminal_velocity" => v_t,
            "tau_p" => tau_p,
        ),
        artifacts = ["stokes_terminal_velocity.png"],
        notes = [
            "Verification-stage exact-solution evidence for lagrangian_dpm: advance_particles! with StokesDrag against the closed-form settling ODE — exact discrete fixed point, transient agreement, and first-order dt convergence.",
        ],
        summary = Dict(
            "droplet_diameter" => d_p,
            "dt_ratios" => collect(dt_ratios),
        ),
    )
end

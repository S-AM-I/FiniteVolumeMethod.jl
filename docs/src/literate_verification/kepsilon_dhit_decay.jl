# # Decaying Homogeneous Turbulence (Standard k-ε)
# This case verifies `solve_turbulence!` for `StandardKEpsilon` against the
# closed-form ODE solution of decaying homogeneous isotropic turbulence
# (DHIT). With uniform fields, no mean-shear production, zero face flux, and
# Neumann boundaries, the k-ε transport system degenerates to
# ```math
# \frac{\mathrm{d}k}{\mathrm{d}t} = -\varepsilon, \qquad
# \frac{\mathrm{d}\varepsilon}{\mathrm{d}t} = -C_{\varepsilon 2}\,\frac{\varepsilon^2}{k},
# ```
# whose exact solution (with $\tau = \varepsilon_0 t / k_0$) is
# ```math
# k(t) = k_0 \left(1 + (C_{\varepsilon 2} - 1)\tau\right)^{-1/(C_{\varepsilon 2} - 1)}, \qquad
# \varepsilon(t) = \varepsilon_0 \left(1 + (C_{\varepsilon 2} - 1)\tau\right)^{-C_{\varepsilon 2}/(C_{\varepsilon 2} - 1)}.
# ```
# With the standard $C_{\varepsilon 2} = 1.92$ the asymptotic decay is
# $k \sim \tau^{-1.087}$.
#
# ## Acceptance Gates
# - Realizability: $k$, $\varepsilon$, $\nu_t$ non-negative throughout, with
#   strictly monotone decay (no oscillation or overshoot)
# - Endpoint agreement with the exact solution to $\leq 1\%$ at
#   $\Delta t / t = 10^{-3}$
# - First-order convergence in $\Delta t$ (implicit Euler): observed order
#   in $(0.8, 1.3)$

using FiniteVolumeMethod
using FiniteVolumeMethod: RANSTurbulenceState, solve_turbulence!
using FiniteVolumeMethod.Parabolic: NeumannBC
using LinearSolve
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

k0 = 1.0
eps0 = 1.0
C2 = 1.92

k_exact(t) = k0 * (1 + (C2 - 1) * eps0 * t / k0)^(-1 / (C2 - 1))
eps_exact(t) = eps0 * (1 + (C2 - 1) * eps0 * t / k0)^(-C2 / (C2 - 1))

# ## Numerical Setup
# A 4 × 4 mesh suffices — the fields are uniform across cells by
# construction, so mesh topology is immaterial. Zero velocity gives no
# mean-shear production; zero face flux gives no convection; the linear
# solver is pinned for cross-platform determinism.
function run_dhit(n_steps, dt)
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    model = StandardKEpsilon()
    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))
    phi = FaceFluxField(:phi, mesh; value = 0.0)
    turb_state = RANSTurbulenceState(model, mesh; k = k0, epsilon = eps0)
    FiniteVolumeMethod.turbulent_viscosity!(turb_state.nu_t, model, turb_state, mesh)

    bc_neumann = NeumannBC(0.0)
    bcs_turb = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
        :k => Dict(
            :left => bc_neumann, :right => bc_neumann,
            :bottom => bc_neumann, :top => bc_neumann,
        ),
        :epsilon => Dict(
            :left => bc_neumann, :right => bc_neumann,
            :bottom => bc_neumann, :top => bc_neumann,
        ),
    )

    k_hist = Float64[k0]
    e_hist = Float64[eps0]
    t = 0.0
    for _ in 1:n_steps
        solve_turbulence!(
            turb_state, model, U, phi, 1.0e-6, mesh, bcs_turb;
            dt = dt, linear_solver = LUFactorization(),
        )
        push!(k_hist, turb_state.fields[:k].internal[1])
        push!(e_hist, turb_state.fields[:epsilon].internal[1])
        t += dt
    end
    return (turb_state = turb_state, k_hist = k_hist, e_hist = e_hist, t_end = t)
end

# ## Realizability and Monotone Decay
res_coarse = run_dhit(200, 0.005)

realizable = all(>=(0.0), res_coarse.k_hist) &&
    all(>=(0.0), res_coarse.e_hist) &&
    all(>=(0.0), res_coarse.turb_state.nu_t)
monotone = all(diff(res_coarse.k_hist) .<= 1.0e-14) &&
    all(diff(res_coarse.e_hist) .<= 1.0e-14)

# ## Endpoint Agreement
# 1000 steps of $\Delta t = 10^{-3}$ over $t \in [0, 1]$.
res_fine = run_dhit(1000, 0.001)
k_err = abs(res_fine.k_hist[end] - k_exact(res_fine.t_end)) / k_exact(res_fine.t_end)
e_err = abs(res_fine.e_hist[end] - eps_exact(res_fine.t_end)) / eps_exact(res_fine.t_end)

# ## Temporal Convergence
step_sets = ((100, 0.01), (200, 0.005), (400, 0.0025))
dt_errors = map(step_sets) do (n_steps, dt)
    res = run_dhit(n_steps, dt)
    abs(res.k_hist[end] - k_exact(res.t_end))
end
dt_orders = [log2(dt_errors[i] / dt_errors[i + 1]) for i in 1:(length(dt_errors) - 1)]

# ## Visualisation — Decay Histories
t_fine = range(0.0, res_fine.t_end; length = length(res_fine.k_hist))
t_dense = range(0.0, res_fine.t_end; length = 300)

fig1 = Figure(fontsize = 24, size = (600, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "t", ylabel = "k, ε",
    title = "DHIT decay (standard k-ε)"
)
lines!(ax1, t_dense, k_exact.(t_dense), color = :black, linewidth = 2, label = "k exact")
lines!(
    ax1, t_dense, eps_exact.(t_dense), color = :black, linewidth = 2,
    linestyle = :dash, label = "ε exact"
)
scatter!(
    ax1, t_fine[1:50:end], res_fine.k_hist[1:50:end], color = :blue,
    markersize = 10, label = "k numerical"
)
scatter!(
    ax1, t_fine[1:50:end], res_fine.e_hist[1:50:end], color = :red,
    marker = :utriangle, markersize = 10, label = "ε numerical"
)
axislegend(ax1, position = :rt)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("kepsilon_dhit_decay.png"), fig1)
end

# ## Visualisation — Temporal Convergence
dts = [dt for (_, dt) in step_sets]
fig2 = Figure(fontsize = 24, size = (600, 500))
ax2 = Axis(
    fig2[1, 1], xlabel = "Δt", ylabel = "|k(T) − k_exact(T)|",
    xscale = log10, yscale = log10,
    title = "Implicit-Euler convergence"
)
scatterlines!(
    ax2, dts, collect(dt_errors), marker = :circle, color = :blue,
    linewidth = 2, markersize = 12, label = "k endpoint error"
)
lines!(
    ax2, dts, dt_errors[1] .* (dts ./ dts[1]),
    color = :gray, linestyle = :dash, linewidth = 1.5, label = "O(Δt)"
)
axislegend(ax2, position = :rb)
resize_to_layout!(fig2)
fig2
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("kepsilon_dhit_convergence.png"), fig2)
end

# ## Acceptance
@test realizable #src
@test monotone #src
@test k_err < 1.0e-2 #src
@test e_err < 1.0e-2 #src
@test all(dt_errors[i] > dt_errors[i + 1] for i in 1:(length(dt_errors) - 1)) #src
for p in dt_orders #src
    @test 0.8 < p < 1.3 #src
end #src
@assert realizable #hide
@assert monotone #hide
@assert k_err < 1.0e-2 #hide
@assert e_err < 1.0e-2 #hide
@assert all(dt_errors[i] > dt_errors[i + 1] for i in 1:(length(dt_errors) - 1)) #hide
@assert all(p -> 0.8 < p < 1.3, dt_orders) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "k_endpoint_relative_error" => k_err,
            "epsilon_endpoint_relative_error" => e_err,
            "dt_errors" => collect(dt_errors),
            "dt_orders" => dt_orders,
        ),
        artifacts = ["kepsilon_dhit_decay.png", "kepsilon_dhit_convergence.png"],
        notes = [
            "Verification-stage exact-solution evidence for turbulence_rans: k-epsilon source terms integrate the DHIT decay ODE through solve_turbulence! with implicit Euler.",
            "Covers source terms only — shear production is verified separately by the log-layer equilibrium benchmark.",
        ],
        summary = Dict(
            "C_eps2" => C2,
            "t_end" => res_fine.t_end,
            "dt_values" => collect(dts),
        ),
    )
end

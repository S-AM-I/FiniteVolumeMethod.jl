# # Unsteady Heat Equation Decay
# This case benchmarks the transient `solve_solid_conduction` path against
# the closed-form separable solution of the 1D heat equation
# ```math
# \frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2},
# \qquad T(0, t) = T(L, t) = 0, \qquad T(x, 0) = \sin(\pi x / L),
# ```
# whose solution is
# ```math
# T(x, t) = \sin(\pi x / L)\, e^{-\pi^2 \alpha t / L^2}.
# ```
# Neumann top/bottom boundaries reduce the 2D solve to strictly 1D, which
# is itself checked (y-invariance to $10^{-10}$).
#
# ## Acceptance Gates
# - Endpoint interior-band $L^2$ error $< 5 \times 10^{-3}$ at
#   $40 \times 8$, 100 steps, $t_{\text{end}} = 0.5$
# - Spatial convergence $O(h^2)$: observed order in $(1.55, 2.3)$ across
#   $N \in \{20, 40, 80\}$ at $\Delta t$ small enough that the temporal
#   floor does not contaminate (4000 steps)
# - Temporal convergence $O(\Delta t)$ (implicit Euler): coarse-regime rate
#   $> 0.6$, monotone error decrease

using FiniteVolumeMethod
using FiniteVolumeMethod: solve_solid_conduction
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using LinearSolve
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

L = 1.0
Ly = 0.2
alpha = 0.1

T_exact(x, t) = sin(pi * x / L) * exp(-pi^2 * alpha * t / L^2)

# ## Transient Solve
# $\rho = C_p = 1$ and $k = \alpha$ give $\alpha_{\text{eff}} = \alpha$;
# the linear solver is pinned for cross-platform determinism.
function run_unsteady(Nx, Ny, n_steps, t_end)
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, L, Ly)
    solid = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = alpha)
    bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => NeumannBC(0.0),
        :top => NeumannBC(0.0),
    )

    Tf = CollocatedScalarField(:T, mesh)
    for c in 1:length(mesh.cell_volumes)
        Tf.internal[c] = sin(pi * mesh.cell_centers[1, c] / L)
    end

    dt = t_end / n_steps
    for _ in 1:n_steps
        Tf = solve_solid_conduction(
            mesh, solid, bcs_T;
            dt = dt, T_old = copy(Tf.internal),
            linear_solver = LUFactorization(),
        )
    end
    return mesh, Tf
end

function interior_l2_error(mesh, Tf, t_end)
    err_sq = 0.0
    vol = 0.0
    for c in 1:length(mesh.cell_volumes)
        x = mesh.cell_centers[1, c]
        if 0.1 < x < 0.9
            err_sq += mesh.cell_volumes[c] * (Tf.internal[c] - T_exact(x, t_end))^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

# ## Endpoint Agreement and 1D Invariance
t_end = 0.5
mesh_ep, Tf_ep = run_unsteady(40, 8, 100, t_end)
endpoint_error = interior_l2_error(mesh_ep, Tf_ep, t_end)

buckets = Dict{Float64, Vector{Float64}}()
for c in 1:length(mesh_ep.cell_volumes)
    x = round(mesh_ep.cell_centers[1, c]; digits = 10)
    push!(get!(buckets, x, Float64[]), Tf_ep.internal[c])
end
y_spread = maximum(
    length(vals) >= 2 ? maximum(vals) - minimum(vals) : 0.0
        for (_, vals) in buckets
)

# ## Spatial Convergence
# 4000 steps keep the $O(\Delta t)$ temporal error ($\approx 6 \times
# 10^{-5}$) below the finest spatial error.
spatial_errors = Float64[]
for N in (20, 40, 80)
    mesh, Tf = run_unsteady(N, 4, 4000, t_end)
    push!(spatial_errors, interior_l2_error(mesh, Tf, t_end))
end
spatial_orders = [
    log2(spatial_errors[i] / spatial_errors[i + 1])
        for i in 1:(length(spatial_errors) - 1)
]

# ## Temporal Convergence
# Fixed fine mesh ($80 \times 4$) so the spatial floor is small; the rate is
# measured in the coarse-$\Delta t$ regime where implicit Euler dominates.
temporal_errors = Float64[]
for n_steps in (50, 100, 200)
    mesh, Tf = run_unsteady(80, 4, n_steps, t_end)
    push!(temporal_errors, interior_l2_error(mesh, Tf, t_end))
end
temporal_rate = log2(temporal_errors[1] / temporal_errors[2])

# ## Visualisation — Endpoint Profile
xs = sort(collect(keys(buckets)))
T_profile = [first(buckets[x]) for x in xs]
x_dense = range(0.0, L; length = 200)

fig1 = Figure(fontsize = 24, size = (600, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "x", ylabel = "T(x, t = 0.5)",
    title = "Unsteady heat decay"
)
lines!(ax1, x_dense, T_exact.(x_dense, t_end), color = :black, linewidth = 2, label = "Exact")
scatter!(ax1, xs, T_profile, color = :red, markersize = 10, label = "Numerical (40 × 8)")
axislegend(ax1, position = :rt)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("unsteady_heat_profile.png"), fig1)
end

# ## Visualisation — Convergence
h_vals = [L / N for N in (20, 40, 80)]
dt_vals = [t_end / n for n in (50, 100, 200)]

fig2 = Figure(fontsize = 24, size = (1000, 450))
axS = Axis(
    fig2[1, 1], xlabel = "h", ylabel = "L² error",
    xscale = log10, yscale = log10, title = "Spatial (4000 steps)"
)
scatterlines!(
    axS, h_vals, spatial_errors, marker = :circle, color = :blue,
    linewidth = 2, markersize = 12
)
lines!(
    axS, h_vals, spatial_errors[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5
)
axT = Axis(
    fig2[1, 2], xlabel = "Δt", ylabel = "L² error",
    xscale = log10, yscale = log10, title = "Temporal (80 × 4)"
)
scatterlines!(
    axT, dt_vals, temporal_errors, marker = :utriangle, color = :red,
    linewidth = 2, markersize = 12
)
lines!(
    axT, dt_vals, temporal_errors[1] .* (dt_vals ./ dt_vals[1]),
    color = :gray, linestyle = :dash, linewidth = 1.5
)
resize_to_layout!(fig2)
fig2
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("unsteady_heat_convergence.png"), fig2)
end

# ## Acceptance
@test endpoint_error < 5.0e-3 #src
@test y_spread < 1.0e-10 #src
for p in spatial_orders #src
    @test 1.55 < p < 2.3 #src
end #src
@test all(spatial_errors[i] > spatial_errors[i + 1] for i in 1:(length(spatial_errors) - 1)) #src
@test temporal_rate > 0.6 #src
@test temporal_errors[1] > temporal_errors[2] #src
@test temporal_errors[2] > temporal_errors[3] - 1.0e-12 #src
@assert endpoint_error < 5.0e-3 #hide
@assert y_spread < 1.0e-10 #hide
@assert all(p -> 1.55 < p < 2.3, spatial_orders) #hide
@assert all(spatial_errors[i] > spatial_errors[i + 1] for i in 1:(length(spatial_errors) - 1)) #hide
@assert temporal_rate > 0.6 #hide
@assert temporal_errors[1] > temporal_errors[2] #hide
@assert temporal_errors[2] > temporal_errors[3] - 1.0e-12 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "endpoint_l2_error" => endpoint_error,
            "y_spread" => y_spread,
            "spatial_orders" => spatial_orders,
            "temporal_rate" => temporal_rate,
        ),
        artifacts = ["unsteady_heat_profile.png", "unsteady_heat_convergence.png"],
        notes = [
            "Benchmark-stage exact-solution evidence for conjugate_heat_transfer: the transient solve_solid_conduction path against the separable heat-equation decay, with independent spatial and temporal convergence studies.",
        ],
        summary = Dict(
            "alpha" => alpha,
            "t_end" => t_end,
            "spatial_meshes" => [20, 40, 80],
            "temporal_steps" => [50, 100, 200],
        ),
    )
end

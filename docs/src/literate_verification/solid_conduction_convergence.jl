# # Steady Solid Conduction Grid Convergence
# This case verifies `solve_solid_conduction` against the closed-form
# series solution of Laplace's equation on the unit square with mixed
# Dirichlet boundary conditions:
# ```math
# -\nabla^2 T = 0 \ \text{on}\ [0,1]^2, \qquad
# T(x, 0) = T(0, y) = T(1, y) = 0, \qquad T(x, 1) = 1,
# ```
# with the exact solution
# ```math
# T(x, y) = \frac{4}{\pi} \sum_{n = 1, 3, 5, \ldots}
# \frac{1}{n} \sin(n\pi x)\,\frac{\sinh(n\pi y)}{\sinh(n\pi)}.
# ```
# At the centre $(0.5, 0.5)$ the series sums to exactly $1/4$ by symmetry.
#
# ## Acceptance Gates
# - Interior-band volume-weighted $L^2$ error converges at $O(h^2)$:
#   observed order in $(1.8, 2.2)$ across $N \in \{20, 40, 80\}$, monotone
#   decrease, finest error $< 10^{-4}$
# - Centre-cell temperature within 3% of the exact $1/4$ on the finest mesh
#
# The interior band excludes the first 10% near each boundary, where the
# $T = 0$ / $T = 1$ corner singularities concentrate discretization error.

using FiniteVolumeMethod
using FiniteVolumeMethod: solve_solid_conduction
using FiniteVolumeMethod.Parabolic: DirichletBC
using LinearSolve
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

function T_exact(x, y; n_terms = 50)
    s = 0.0
    for n in 1:2:(2 * n_terms - 1)
        s += (4 / π) * (1 / n) * sin(n * π * x) * sinh(n * π * y) / sinh(n * π)
    end
    return s
end

# ## Refinement Study
# The linear solver is pinned for cross-platform determinism.
function solve_conduction(N)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    solid = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(1.0),
    )
    Tf = solve_solid_conduction(mesh, solid, bcs; linear_solver = LUFactorization())
    return mesh, Tf
end

function interior_l2_error(mesh, Tf, N)
    err_sq = 0.0
    vol = 0.0
    for c in 1:(N * N)
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.1 < x < 0.9 && 0.1 < y < 0.9
            err_sq += mesh.cell_volumes[c] * (Tf.internal[c] - T_exact(x, y))^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

mesh_sizes = [20, 40, 80]
solutions = [solve_conduction(N) for N in mesh_sizes]
errors_L2 = [
    interior_l2_error(mesh, Tf, N)
        for ((mesh, Tf), N) in zip(solutions, mesh_sizes)
]
orders = [log2(errors_L2[i] / errors_L2[i + 1]) for i in 1:(length(errors_L2) - 1)]

# ## Centre-Cell Check
mesh_fine, Tf_fine = solutions[end]
_, c_center = findmin(
    abs2.(mesh_fine.cell_centers[1, :] .- 0.5) .+
        abs2.(mesh_fine.cell_centers[2, :] .- 0.5)
)
T_center = Tf_fine.internal[c_center]
center_error = abs(T_center - 0.25) / 0.25

# ## Visualisation — Fields
N_fine = mesh_sizes[end]
T_num = [Tf_fine.internal[(j - 1) * N_fine + i] for i in 1:N_fine, j in 1:N_fine]
T_ex = [
    T_exact(
            mesh_fine.cell_centers[1, (j - 1) * N_fine + i],
            mesh_fine.cell_centers[2, (j - 1) * N_fine + i],
        ) for i in 1:N_fine, j in 1:N_fine
]
x_centers = [mesh_fine.cell_centers[1, i] for i in 1:N_fine]
y_centers = [mesh_fine.cell_centers[2, (j - 1) * N_fine + 1] for j in 1:N_fine]

fig1 = Figure(fontsize = 24, size = (1200, 400))
ax1 = Axis(fig1[1, 1], xlabel = "x", ylabel = "y", title = "Numerical", aspect = DataAspect())
heatmap!(ax1, x_centers, y_centers, T_num, colormap = :thermal)
ax2 = Axis(fig1[1, 2], xlabel = "x", ylabel = "y", title = "Series solution", aspect = DataAspect())
heatmap!(ax2, x_centers, y_centers, T_ex, colormap = :thermal)
ax3 = Axis(fig1[1, 3], xlabel = "x", ylabel = "y", title = "|Error|", aspect = DataAspect())
hm = heatmap!(ax3, x_centers, y_centers, abs.(T_num .- T_ex), colormap = :hot)
Colorbar(fig1[1, 4], hm)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("solid_conduction_fields.png"), fig1)
end

# ## Visualisation — Convergence
h_vals = 1.0 ./ mesh_sizes
fig2 = Figure(fontsize = 24, size = (600, 500))
ax = Axis(
    fig2[1, 1], xlabel = "h = 1/N", ylabel = "L² error",
    xscale = log10, yscale = log10,
    title = "Solid conduction convergence"
)
scatterlines!(
    ax, h_vals, errors_L2, label = "L² (interior)", marker = :circle,
    color = :blue, linewidth = 2, markersize = 12
)
lines!(
    ax, h_vals, errors_L2[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5, label = "O(h²)"
)
axislegend(ax, position = :rb)
for i in eachindex(orders)
    x_mid = sqrt(h_vals[i] * h_vals[i + 1])
    text!(
        ax, x_mid, errors_L2[i] * 0.6;
        text = "$(round(orders[i], digits = 2))", fontsize = 14, color = :blue
    )
end
resize_to_layout!(fig2)
fig2
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("solid_conduction_convergence.png"), fig2)
end

# ## Acceptance
@test all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)) #src
for p in orders #src
    @test 1.8 < p < 2.2 #src
end #src
@test errors_L2[end] < 1.0e-4 #src
@test center_error < 0.03 #src
@assert all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)) #hide
@assert all(p -> 1.8 < p < 2.2, orders) #hide
@assert errors_L2[end] < 1.0e-4 #hide
@assert center_error < 0.03 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "l2_errors" => errors_L2,
            "observed_orders" => orders,
            "center_relative_error" => center_error,
        ),
        artifacts = ["solid_conduction_fields.png", "solid_conduction_convergence.png"],
        notes = [
            "Verification-stage exact-solution evidence for conjugate_heat_transfer: solve_solid_conduction against the Laplace series solution with mixed Dirichlet BCs.",
        ],
        summary = Dict(
            "mesh_sizes" => mesh_sizes,
            "h_values" => h_vals,
            "T_center" => T_center,
        ),
    )
end

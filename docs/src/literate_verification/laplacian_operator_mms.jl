# # Laplacian Operator MMS Convergence
# This case verifies the collocated Laplacian operator
# `assemble_laplacian!` by the method of manufactured solutions,
# independently of any pressure-velocity coupling loop:
# ```math
# \varphi(x, y) = \sin(\pi x)\sin(\pi y), \qquad
# -\nabla^2 \varphi = 2\pi^2 \sin(\pi x)\sin(\pi y),
# ```
# on $[0, 1]^2$ with homogeneous Dirichlet boundaries. Solving the
# assembled system with the manufactured forcing recovers $\varphi$ at
# second order.
#
# ## Acceptance Gates
# - $L^2$ order at the finest transition in $(1.8, 2.2)$; $L^\infty$
#   order $> 1.7$
# - Monotone error decrease in both norms across
#   $N \in \{10, 20, 40, 80\}$; finest $L^2 < 10^{-3}$

using FiniteVolumeMethod
using FiniteVolumeMethod: CollocatedEquation, assemble_laplacian!, to_linear_problem
using FiniteVolumeMethod.Parabolic: DirichletBC
using LinearSolve
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

phi_exact(x, y) = sin(π * x) * sin(π * y)
f_forcing(x, y) = 2π^2 * sin(π * x) * sin(π * y)

# ## MMS Solve
# The linear solver is pinned for cross-platform determinism.
function solve_laplacian_mms(N)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(0.0),
    )
    eq = CollocatedEquation(mesh)
    assemble_laplacian!(eq, 1.0, mesh, bcs)
    n_cells = length(mesh.cell_volumes)
    for c in 1:n_cells
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        eq.b[c] += mesh.cell_volumes[c] * f_forcing(x, y)
    end
    sol = solve(to_linear_problem(eq), LUFactorization())

    err_inf = 0.0
    err_sq = 0.0
    vol_tot = 0.0
    for c in 1:n_cells
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        e = abs(sol.u[c] - phi_exact(x, y))
        err_inf = max(err_inf, e)
        err_sq += mesh.cell_volumes[c] * e^2
        vol_tot += mesh.cell_volumes[c]
    end
    return err_inf, sqrt(err_sq / vol_tot)
end

mesh_sizes = [10, 20, 40, 80]
results = [solve_laplacian_mms(N) for N in mesh_sizes]
errors_inf = [r[1] for r in results]
errors_L2 = [r[2] for r in results]
orders_L2 = [log2(errors_L2[i] / errors_L2[i + 1]) for i in 1:(length(mesh_sizes) - 1)]
orders_inf = [log2(errors_inf[i] / errors_inf[i + 1]) for i in 1:(length(mesh_sizes) - 1)]

# ## Visualisation — Convergence
h_vals = 1.0 ./ mesh_sizes
fig1 = Figure(fontsize = 24, size = (600, 500))
ax = Axis(
    fig1[1, 1], xlabel = "h = 1/N", ylabel = "Error",
    xscale = log10, yscale = log10,
    title = "Laplacian operator MMS"
)
scatterlines!(
    ax, h_vals, errors_L2, label = "L²", marker = :circle,
    color = :blue, linewidth = 2, markersize = 12
)
scatterlines!(
    ax, h_vals, errors_inf, label = "L∞", marker = :utriangle,
    color = :red, linewidth = 2, markersize = 12
)
lines!(
    ax, h_vals, errors_L2[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5, label = "O(h²)"
)
axislegend(ax, position = :rb)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("laplacian_mms_convergence.png"), fig1)
end

# ## Acceptance
@test 1.8 < orders_L2[end] < 2.2 #src
@test orders_inf[end] > 1.7 #src
@test all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)) #src
@test all(errors_inf[i] > errors_inf[i + 1] for i in 1:(length(errors_inf) - 1)) #src
@test errors_L2[end] < 1.0e-3 #src
@assert 1.8 < orders_L2[end] < 2.2 #hide
@assert orders_inf[end] > 1.7 #hide
@assert all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)) #hide
@assert all(errors_inf[i] > errors_inf[i + 1] for i in 1:(length(errors_inf) - 1)) #hide
@assert errors_L2[end] < 1.0e-3 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "l2_errors" => errors_L2,
            "linf_errors" => errors_inf,
            "l2_orders" => orders_L2,
            "linf_orders" => orders_inf,
        ),
        artifacts = ["laplacian_mms_convergence.png"],
        notes = [
            "Verification-stage manufactured-solution evidence for collocated_operators: assemble_laplacian! recovers the manufactured field at O(h^2), independently of any solver loop.",
        ],
        summary = Dict(
            "mesh_sizes" => mesh_sizes,
            "h_values" => h_vals,
        ),
    )
end

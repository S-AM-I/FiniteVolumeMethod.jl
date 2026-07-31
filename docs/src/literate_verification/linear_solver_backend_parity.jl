# # Linear-Solver Backend Parity
# This case verifies the linear-solver infrastructure on a reference
# Poisson problem with known analytic solution
# ($-\nabla^2 \varphi = 2\pi^2 \sin\pi x \sin\pi y$,
# $\varphi = \sin\pi x \sin\pi y$, homogeneous Dirichlet):
# 1. **LU (direct)** solves to the discretization error of the operator
#    ($O(h^2) \approx 10^{-3}$ at $N = 32$)
# 2. **CG and GMRES (Krylov)** match the direct solution pointwise within
#    their advertised tolerance
# 3. Under refinement the direct solution converges at $O(h^2)$ — the
#    backend does not limit the discretization
#
# ## Acceptance Gates
# - LU interior $L^2$ error $< 5 \times 10^{-3}$ at $N = 32$
# - $\max_c |u_{\text{CG}} - u_{\text{LU}}|$ and
#   $\max_c |u_{\text{GMRES}} - u_{\text{LU}}| < 10^{-6}$
# - LU refinement orders in $(1.7, 2.3)$ across $N \in \{16, 32, 64\}$,
#   monotone

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

phi_exact(x, y) = sin(pi * x) * sin(pi * y)

function solve_poisson(linear_solver, N)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(0.0),
    )
    eq = CollocatedEquation(mesh)
    assemble_laplacian!(eq, 1.0, mesh, bcs)
    for c in 1:length(mesh.cell_volumes)
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        eq.b[c] += 2 * pi^2 * sin(pi * x) * sin(pi * y) * mesh.cell_volumes[c]
    end
    sol = LinearSolve.solve(to_linear_problem(eq), linear_solver)
    return mesh, sol.u
end

function interior_l2_error(mesh, u)
    err_sq = 0.0
    vol = 0.0
    for c in 1:length(mesh.cell_volumes)
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.1 < x < 0.9 && 0.1 < y < 0.9
            err_sq += mesh.cell_volumes[c] * (u[c] - phi_exact(x, y))^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

# ## Backend Parity at N = 32
mesh32, u_lu = solve_poisson(LUFactorization(), 32)
_, u_cg = solve_poisson(KrylovJL_CG(), 32)
_, u_gmres = solve_poisson(KrylovJL_GMRES(), 32)

lu_error = interior_l2_error(mesh32, u_lu)
cg_deviation = maximum(abs, u_cg .- u_lu)
gmres_deviation = maximum(abs, u_gmres .- u_lu)

# ## Refinement with the Direct Backend
mesh_sizes = [16, 32, 64]
refine_errors = map(mesh_sizes) do N
    mesh, u = solve_poisson(LUFactorization(), N)
    interior_l2_error(mesh, u)
end
refine_orders = [
    log2(refine_errors[i] / refine_errors[i + 1])
        for i in 1:(length(refine_errors) - 1)
]

# ## Visualisation — Parity and Convergence
h_vals = 1.0 ./ mesh_sizes
fig1 = Figure(fontsize = 24, size = (1000, 450))
ax1 = Axis(
    fig1[1, 1], ylabel = "max |u − u_LU|", yscale = log10,
    xticks = (1:2, ["CG", "GMRES"]), title = "Krylov parity vs direct"
)
barplot!(ax1, [1, 2], [cg_deviation, gmres_deviation] .+ 1.0e-18, color = [:blue, :red])
hlines!(ax1, [1.0e-6], color = :black, linestyle = :dash)
ax2 = Axis(
    fig1[1, 2], xlabel = "h", ylabel = "L² error",
    xscale = log10, yscale = log10, title = "LU refinement"
)
scatterlines!(
    ax2, h_vals, refine_errors, marker = :circle, color = :blue,
    linewidth = 2, markersize = 12
)
lines!(
    ax2, h_vals, refine_errors[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5
)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("linear_solver_parity.png"), fig1)
end

# ## Acceptance
@test lu_error < 5.0e-3 #src
@test cg_deviation < 1.0e-6 #src
@test gmres_deviation < 1.0e-6 #src
@test refine_errors[1] > refine_errors[2] > refine_errors[3] #src
for p in refine_orders #src
    @test 1.7 < p < 2.3 #src
end #src
@assert lu_error < 5.0e-3 #hide
@assert cg_deviation < 1.0e-6 #hide
@assert gmres_deviation < 1.0e-6 #hide
@assert refine_errors[1] > refine_errors[2] > refine_errors[3] #hide
@assert all(p -> 1.7 < p < 2.3, refine_orders) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "lu_l2_error" => lu_error,
            "cg_max_deviation" => cg_deviation,
            "gmres_max_deviation" => gmres_deviation,
            "refinement_orders" => refine_orders,
        ),
        artifacts = ["linear_solver_parity.png"],
        notes = [
            "Verification-stage evidence for linear_solver_infra: direct and Krylov backends agree pointwise on the reference Poisson problem, and the direct backend preserves the O(h^2) discretization convergence.",
        ],
        summary = Dict(
            "mesh_sizes" => mesh_sizes,
            "backends" => ["LUFactorization", "KrylovJL_CG", "KrylovJL_GMRES"],
        ),
    )
end

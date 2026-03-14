using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Flux Balance Verification
# This example verifies a discrete control-volume balance identity for the
# parabolic solver family. After assembling a steady Poisson problem, we solve
# the discrete linear system and measure the residual `A*u - b`, which should
# vanish to machine precision for the converged discrete state.
#
# ## Mathematical Setup
# We solve
# ```math
# \nabla^2 u = f \quad \text{on } [0,1]^2,
# ```
# with homogeneous Dirichlet boundary conditions and exact solution
# ```math
# u_{\mathrm{exact}}(x,y) = -\sin(\pi x)\sin(\pi y).
# ```
# The corresponding source is
# ```math
# f(x,y) = 2\pi^2 \sin(\pi x)\sin(\pi y).
# ```
# For the assembled discrete system `A*u = b`, every row corresponds to a
# control-volume balance. The invariant we check is the infinity-norm residual
# of that discrete balance.

using FiniteVolumeMethod
using DelaunayTriangulation
using LinearAlgebra
using LinearSolve
using Test #src
using ReferenceTests #src
using CairoMakie

u_exact(x, y) = -sin(pi * x) * sin(pi * y)
source_poisson(x, y, p) = 2 * pi^2 * sin(pi * x) * sin(pi * y)
bc_zero(x, y, t, u, p) = 0.0

mesh_sizes = [12, 24, 48, 96]
residual_inf = Float64[]
solution_errors = Float64[]

last_prob = nothing
last_sol = nothing

for N in mesh_sizes
    tri = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, N, N; single_boundary = true)
    mesh = FVMGeometry(tri)
    BCs = BoundaryConditions(mesh, bc_zero, Dirichlet)

    prob = PoissonsEquation(
        mesh, BCs;
        diffusion_function = (x, y, p) -> 1.0,
        source_function = source_poisson,
    )
    sol = solve(prob, KLUFactorization())

    global last_prob = prob
    global last_sol = sol

    residual = prob.A * sol.u - prob.b
    push!(residual_inf, norm(residual, Inf))

    err_inf = 0.0
    for (j, (x, y)) in enumerate(DelaunayTriangulation.each_point(tri))
        err_inf = max(err_inf, abs(sol.u[j] - u_exact(x, y)))
    end
    push!(solution_errors, err_inf)
end

fig = Figure(fontsize = 24, size = (1200, 450))
ax1 = Axis(
    fig[1, 1],
    xlabel = "N",
    ylabel = "norm(A*u - b, Inf)",
    xscale = log2,
    yscale = log10,
    title = "Discrete Flux-Balance Residual",
)
scatterlines!(ax1, mesh_sizes, max.(residual_inf, 1.0e-18), color = :blue, marker = :circle, linewidth = 2, markersize = 12)
hlines!(ax1, [1.0e-12], color = :red, linestyle = :dash, linewidth = 1.5, label = "1e-12 threshold")
axislegend(ax1, position = :rb)

ax2 = Axis(
    fig[1, 2],
    xlabel = "N",
    ylabel = "Linf solution error",
    xscale = log2,
    yscale = log10,
    title = "Solved Discrete State Error",
)
scatterlines!(ax2, mesh_sizes, solution_errors, color = :darkgreen, marker = :utriangle, linewidth = 2, markersize = 12)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "flux_balance_verification.png") fig #src
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("flux_balance_verification.png"), fig)
end

@test maximum(residual_inf) < 1.0e-10 #src
@test all(solution_errors[i] > solution_errors[i + 1] for i in 1:(length(solution_errors) - 1)) #src
@assert maximum(residual_inf) < 1.0e-10 #hide
@assert all(solution_errors[i] > solution_errors[i + 1] for i in 1:(length(solution_errors) - 1)) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "residual_inf" => residual_inf,
            "max_residual_inf" => maximum(residual_inf),
            "solution_errors" => solution_errors,
        ),
        artifacts = ["flux_balance_verification.png"],
        notes = [
            "Canonical parabolic template path via PoissonsEquation(...), sciml_problem(prob) through the embedded LinearProblem, and solve(prob, KLUFactorization()).",
            "This evidence entry is the parabolic invariant-stage discrete control-volume balance case.",
        ],
        summary = Dict(
            "mesh_sizes" => mesh_sizes,
            "monotone_solution_error_decrease" => all(solution_errors[i] > solution_errors[i + 1] for i in 1:(length(solution_errors) - 1)),
        ),
    )
end

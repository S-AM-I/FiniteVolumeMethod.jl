using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Poisson Equation Convergence
# This example verifies the steady-state solver by solving the Poisson equation
# on the unit square with homogeneous Dirichlet boundary conditions and
# confirming second-order convergence of `SteadyFVMProblem`.
#
# ## Mathematical Setup
# We solve:
# ```math
# 0 = \nabla^2 u + S(x,y), \qquad u\big|_{\partial\Omega} = 0,
# ```
# with exact solution $u_{\text{exact}}(x,y) = \sin(\pi x)\sin(\pi y)$
# and source term $S(x,y) = 2\pi^2\sin(\pi x)\sin(\pi y)$.
#
# ## Inputs
# - **Mesh sizes**: $N \in \{10, 20, 40, 80\}$
# - **Diffusion coefficient**: $D = 1$
# - **Solver**: `DynamicSS(Rosenbrock23())`

using FiniteVolumeMethod, DelaunayTriangulation
using OrdinaryDiffEq, SteadyStateDiffEq
using Test #src
using ReferenceTests #src
using CairoMakie

u_exact(x, y) = sin(pi * x) * sin(pi * y)

source_poisson(x, y, t, u, p) = 2 * pi^2 * sin(pi * x) * sin(pi * y)

D_poisson(x, y, t, u, p) = 1.0

bc_poisson(x, y, t, u, p) = zero(u)

# ## Grid Refinement Study
mesh_sizes = [10, 20, 40, 80]
errors_Linf = Float64[]
errors_L2 = Float64[]
last_tri = nothing
last_sol = nothing

for N in mesh_sizes
    tri = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, N, N; single_boundary = true)
    mesh = FVMGeometry(tri)
    BCs = BoundaryConditions(mesh, bc_poisson, Dirichlet)

    initial_condition = zeros(DelaunayTriangulation.num_points(tri))

    prob = FVMProblem(
        mesh, BCs;
        diffusion_function = D_poisson,
        source_function = source_poisson,
        initial_condition = initial_condition,
        final_time = Inf
    )
    steady_prob = SteadyFVMProblem(prob)
    sol = solve(steady_prob, DynamicSS(Rosenbrock23()))

    ## Cache finest mesh result for visualisation
    global last_tri = tri
    global last_sol = sol

    ## Compute errors
    err_inf = 0.0
    err_l2 = 0.0
    n_pts = 0
    for (j, (x, y)) in enumerate(DelaunayTriangulation.each_point(tri))
        e = abs(sol.u[j] - u_exact(x, y))
        err_inf = max(err_inf, e)
        err_l2 += e^2
        n_pts += 1
    end
    push!(errors_Linf, err_inf)
    push!(errors_L2, sqrt(err_l2 / n_pts))
end

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_Linf = convergence_rates(errors_Linf)
rates_L2 = convergence_rates(errors_L2)

# ## Visualisation — Solution Comparison
# Three panels at the finest mesh: numerical, exact, and error.
tri_fine = last_tri
sol_fine = last_sol

u_num = sol_fine.u
u_ex = [u_exact(x, y) for (x, y) in DelaunayTriangulation.each_point(tri_fine)]
u_err = abs.(u_num .- u_ex)

fig1 = Figure(fontsize = 24, size = (1500, 450))
ax1 = Axis(fig1[1, 1], xlabel = "x", ylabel = "y", title = "Numerical", aspect = DataAspect())
tricontourf!(ax1, tri_fine, u_num, colormap = :viridis)
ax2 = Axis(fig1[1, 2], xlabel = "x", ylabel = "y", title = "Exact", aspect = DataAspect())
tricontourf!(ax2, tri_fine, u_ex, colormap = :viridis)
ax3 = Axis(fig1[1, 3], xlabel = "x", ylabel = "y", title = "Error", aspect = DataAspect())
tricontourf!(ax3, tri_fine, u_err, colormap = :hot)
resize_to_layout!(fig1)
fig1
@test_reference joinpath(@__DIR__, "../figures", "poisson_convergence_solution.png") fig1 #src

# ## Visualisation — Convergence Plot
h_vals = 1.0 ./ mesh_sizes
fig2 = Figure(fontsize = 24, size = (600, 500))
ax = Axis(
    fig2[1, 1], xlabel = "h = 1/N", ylabel = "Error",
    xscale = log10, yscale = log10,
    title = "Poisson Convergence (Steady-State Solver)"
)
scatterlines!(
    ax, h_vals, errors_Linf, label = "L∞", marker = :circle,
    color = :blue, linewidth = 2, markersize = 12
)
scatterlines!(
    ax, h_vals, errors_L2, label = "L2", marker = :utriangle,
    color = :red, linewidth = 2, markersize = 12
)
lines!(
    ax, h_vals, errors_Linf[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5, label = "O(h²)"
)
axislegend(ax, position = :rb)
for i in eachindex(rates_Linf)
    x_mid = sqrt(h_vals[i] * h_vals[i + 1])
    text!(ax, x_mid, errors_Linf[i] * 0.6; text = "$(round(rates_Linf[i], digits = 2))", fontsize = 14, color = :blue)
end
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "poisson_convergence_rates.png") fig2 #src

# ## Test Assertions
@test all(r -> r > 1.7, rates_Linf) #src
@test all(r -> r > 1.7, rates_L2) #src
@assert all(r -> r > 1.7, rates_Linf) #hide
@assert all(r -> r > 1.7, rates_L2) #hide

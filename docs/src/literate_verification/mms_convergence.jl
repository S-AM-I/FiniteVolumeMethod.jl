using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # MMS Convergence (Parabolic)
# This example verifies the parabolic (vertex-centred) solver using the
# **method of manufactured solutions** (MMS). We choose an exact solution,
# derive the corresponding source term, and confirm second-order convergence
# as the mesh is refined.
#
# ## Mathematical Setup
# We solve the diffusion equation with a source term on the unit square
# $\Omega = [0,1]^2$ with homogeneous Dirichlet boundary conditions:
# ```math
# \pdv{u}{t} = \nabla^2 u + S(x,y,t), \qquad u\big|_{\partial\Omega} = 0.
# ```
# The manufactured exact solution is:
# ```math
# u_{\text{exact}}(x,y,t) = \sin(\pi x)\sin(\pi y)\,e^{-t},
# ```
# which yields the source term:
# ```math
# S(x,y,t) = (2\pi^2 - 1)\sin(\pi x)\sin(\pi y)\,e^{-t}.
# ```
#
# ## Inputs
# - **Mesh sizes**: $N \in \{25, 50, 100, 200\}$ (characteristic cell size $h = 1/N$)
# - **Diffusion coefficient**: $D = 1$
# - **Final time**: $t_f = 1.0$
# - **Time integrator**: `Tsit5()`

using FiniteVolumeMethod, DelaunayTriangulation
using OrdinaryDiffEq
using Test #src
using ReferenceTests #src
using CairoMakie

# Define the exact solution, source, and PDE coefficients:
u_exact(x, y, t) = sin(pi * x) * sin(pi * y) * exp(-t)

function source_mms(x, y, t, u, p)
    return (2 * pi^2 - 1) * sin(pi * x) * sin(pi * y) * exp(-t)
end

D_mms(x, y, t, u, p) = 1.0

bc_mms(x, y, t, u, p) = zero(u)

# ## Grid Refinement Study
# For each mesh size we build a triangulation, solve the PDE, and compute
# the $L^\infty$ and $L^2$ errors at $t = 1$.
mesh_sizes = [25, 50, 100, 200]
t_final = 1.0
errors_Linf = Float64[]
errors_L2 = Float64[]

for N in mesh_sizes
    tri = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, N, N; single_boundary = true)
    mesh = FVMGeometry(tri)
    BCs = BoundaryConditions(mesh, bc_mms, Dirichlet)

    f0 = (x, y) -> u_exact(x, y, 0.0)
    initial_condition = [f0(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]

    prob = FVMProblem(
        mesh, BCs;
        diffusion_function = D_mms,
        source_function = source_mms,
        initial_condition = initial_condition,
        final_time = t_final
    )
    sol = solve(prob, Tsit5(); saveat = [t_final])

    ## Compute errors at the final time
    err_inf = 0.0
    err_l2 = 0.0
    n_pts = 0
    for (j, (x, y)) in enumerate(DelaunayTriangulation.each_point(tri))
        e = abs(sol.u[end][j] - u_exact(x, y, t_final))
        err_inf = max(err_inf, e)
        err_l2 += e^2
        n_pts += 1
    end
    push!(errors_Linf, err_inf)
    push!(errors_L2, sqrt(err_l2 / n_pts))
end

# ## Convergence Rates
# We compute the rate as $p = \log_2(e_N / e_{2N})$:
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_Linf = convergence_rates(errors_Linf)
rates_L2 = convergence_rates(errors_L2)

# ## Visualisation — Solution Comparison
# Three panels at the finest mesh: numerical, exact, and error.
tri_fine = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, mesh_sizes[end], mesh_sizes[end]; single_boundary = true)
mesh_fine = FVMGeometry(tri_fine)
BCs_fine = BoundaryConditions(mesh_fine, bc_mms, Dirichlet)
f0_fine = (x, y) -> u_exact(x, y, 0.0)
ic_fine = [f0_fine(x, y) for (x, y) in DelaunayTriangulation.each_point(tri_fine)]
prob_fine = FVMProblem(
    mesh_fine, BCs_fine;
    diffusion_function = D_mms,
    source_function = source_mms,
    initial_condition = ic_fine,
    final_time = t_final
)
sol_fine = solve(prob_fine, Tsit5(); saveat = [t_final])

u_num = sol_fine.u[end]
u_ex = [u_exact(x, y, t_final) for (x, y) in DelaunayTriangulation.each_point(tri_fine)]
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
@test_reference joinpath(@__DIR__, "../figures", "mms_convergence_solution.png") fig1 #src

# ## Visualisation — Convergence Plot
h_vals = 1.0 ./ mesh_sizes
fig2 = Figure(fontsize = 24, size = (600, 500))
ax = Axis(
    fig2[1, 1], xlabel = "h = 1/N", ylabel = "Error",
    xscale = log10, yscale = log10,
    title = "MMS Convergence (Parabolic Solver)"
)
scatterlines!(
    ax, h_vals, errors_Linf, label = "L∞", marker = :circle,
    color = :blue, linewidth = 2, markersize = 12
)
scatterlines!(
    ax, h_vals, errors_L2, label = "L2", marker = :utriangle,
    color = :red, linewidth = 2, markersize = 12
)
## O(h^2) reference slope
lines!(
    ax, h_vals, errors_Linf[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5, label = "O(h²)"
)
axislegend(ax, position = :rb)
## Annotate rates
for i in eachindex(rates_Linf)
    x_mid = sqrt(h_vals[i] * h_vals[i + 1])
    text!(ax, x_mid, errors_Linf[i] * 0.6; text = "$(round(rates_Linf[i], digits = 2))", fontsize = 14, color = :blue)
end
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "mms_convergence_rates.png") fig2 #src

# ## Test Assertions
@test all(r -> r > 1.7, rates_Linf) #src
@test all(r -> r > 1.7, rates_L2) #src
@assert all(r -> r > 1.7, rates_Linf) #hide
@assert all(r -> r > 1.7, rates_L2) #hide

using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # MHD div(B) Preservation
# This example verifies that the constrained transport (CT) algorithm
# preserves $\nabla\cdot\mathbf{B} = 0$ to machine precision when the
# magnetic field is initialised from a vector potential.
#
# ## Mathematical Setup
# We advect a weak circular magnetic field loop of radius $R_0 = 0.3$
# and amplitude $A_0 = 10^{-3}$ across a periodic $[0,1]^2$ domain
# at velocity $(v_x, v_y) = (1, 0.5)$.
#
# The vector potential:
# ```math
# A_z(x,y) = \begin{cases}A_0(R_0 - r) & r < R_0,\\0 & r \geq R_0,\end{cases}
# ```
# where $r = \sqrt{(x-0.5)^2 + (y-0.5)^2}$.
#
# ## Inputs
# - **Grid sizes**: $N \in \{16, 32, 64\}$ (2D, so $N \times N$ cells)
# - **Solver**: HLLD + MUSCL(Minmod)
# - **Final time**: $t_f = 1.0$

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = IdealMHDEquations{2}(eos)

R0 = 0.3
A0 = 1.0e-3
vx_bg, vy_bg = 1.0, 0.5
rho_bg, P_bg = 1.0, 1.0

# Initial condition (cell-centred values):
function loop_ic(x, y)
    r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
    if r < R0
        Bx = -A0 * (y - 0.5) / r
        By = A0 * (x - 0.5) / r
    else
        Bx = 0.0
        By = 0.0
    end
    return SVector(rho_bg, vx_bg, vy_bg, 0.0, P_bg, Bx, By, 0.0)
end

# Vector potential for divergence-free initialisation:
function Az_loop(x, y)
    r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
    return r < R0 ? A0 * (R0 - r) : 0.0
end

# ## Grid Resolution Study
grid_sizes = [16, 32, 64]
divB_max_values = Float64[]

for N in grid_sizes
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, N)
    prob = HyperbolicProblem2D(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        loop_ic; final_time = 1.0, cfl = 0.4
    )
    coords, U, t_final, ct = solve_hyperbolic(prob; vector_potential = Az_loop)
    divB = max_divB(ct, mesh)
    push!(divB_max_values, divB)
end

# ## Visualisation — Solution and div(B) Field at Finest Resolution
N_fine = grid_sizes[end]
mesh_fine = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N_fine, N_fine)
prob_fine = HyperbolicProblem2D(
    law, mesh_fine, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    loop_ic; final_time = 1.0, cfl = 0.4
)
coords_fine, U_fine, t_fine, ct_fine = solve_hyperbolic(prob_fine; vector_potential = Az_loop)

nx, ny = N_fine, N_fine
xc = [coords_fine[1][i] for i in 1:nx]
yc = [coords_fine[2][j] for j in 1:ny]

Bmag = [
    begin
            w = conserved_to_primitive(law, U_fine[i, j])
            sqrt(w[6]^2 + w[7]^2)
        end for i in 1:nx, j in 1:ny
]

divB_field = compute_divB(ct_fine, mesh_fine.dx, mesh_fine.dy, nx, ny)
divB_abs = abs.(divB_field)

fig1 = Figure(fontsize = 24, size = (1100, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "x", ylabel = "y",
    title = "|B| at t = $(round(t_fine, digits = 2))", aspect = DataAspect()
)
hm1 = heatmap!(ax1, xc, yc, Bmag, colormap = :viridis)
Colorbar(fig1[1, 2], hm1)
ax2 = Axis(
    fig1[1, 3], xlabel = "x", ylabel = "y",
    title = "|div(B)|", aspect = DataAspect()
)
hm2 = heatmap!(ax2, xc, yc, divB_abs, colormap = :inferno)
Colorbar(fig1[1, 4], hm2)
resize_to_layout!(fig1)
fig1
@test_reference joinpath(@__DIR__, "../figures", "mhd_divb_solution.png") fig1 #src

# ## Visualisation — max|div(B)| vs Grid Size
fig2 = Figure(fontsize = 24, size = (600, 450))
ax = Axis(
    fig2[1, 1], xlabel = "N", ylabel = "max |∇·B|",
    yscale = log10, title = "CT Divergence Preservation"
)
scatterlines!(ax, grid_sizes, divB_max_values, color = :blue, marker = :circle, linewidth = 2, markersize = 12)
hlines!(ax, [eps(Float64)], color = :red, linestyle = :dash, linewidth = 1.5, label = "Machine ε")
axislegend(ax, position = :rt)
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "mhd_divb_convergence.png") fig2 #src

# ## Test Assertions
# All max|div(B)| values should be at machine precision, independent of N.
@test all(d -> d < 1.0e-12, divB_max_values) #src
@assert all(d -> d < 1.0e-12, divB_max_values) #hide

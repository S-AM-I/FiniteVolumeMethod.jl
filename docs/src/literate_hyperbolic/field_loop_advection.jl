using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Field Loop Advection
# This tutorial demonstrates the constrained transport (CT) algorithm
# by advecting a weak magnetic field loop across a periodic domain.
# The key lesson is that initialising discontinuous magnetic fields by
# point evaluation fails to satisfy $\nabla\cdot\vb B = 0$; instead,
# one must use a **vector potential** initialisation via Stokes' theorem.
#
# ## Problem Setup
# A circular magnetic field loop of radius $R_0 = 0.3$ and amplitude
# $A_0 = 10^{-3}$ is advected at velocity $(v_x, v_y) = (1, 0.5)$
# across $[0, 1]^2$ with periodic boundary conditions.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = IdealMHDEquations{2}(eos)

R0 = 0.3
A0 = 1.0e-3
vx_bg, vy_bg = 1.0, 0.5
rho_bg, P_bg = 1.0, 1.0

# The initial condition sets the field loop using point evaluation
# (for the cell-centred values):
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

# ## Vector Potential
# The vector potential $A_z(x,y)$ gives a divergence-free magnetic field
# by construction. For a circular loop:
# ```math
# A_z(x,y) = \begin{cases}A_0(R_0 - r) & r < R_0,\\0 & r \geq R_0,\end{cases}
# ```
# where $r = \sqrt{(x-0.5)^2 + (y-0.5)^2}$.
function Az_loop(x, y)
    r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
    return r < R0 ? A0 * (R0 - r) : 0.0
end

# ## Solving
N = 50
mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, N)

prob = HyperbolicProblem2D(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    loop_ic; final_time = 1.0, cfl = 0.4
)

# The `vector_potential` keyword triggers `initialize_ct_from_potential!`,
# which uses Stokes' theorem to compute face-centred $B$ values:
coords, U, t_final, ct = solve_hyperbolic(prob; vector_potential = Az_loop)
coords |> tc #hide

# ## Checking $\nabla\cdot\vb B$
# With CT + vector potential initialisation, the discrete divergence
# remains at machine precision:
divB_max = max_divB(ct, mesh)
divB_max < 1.0e-10 || @warn("divB exceeds tolerance: $divB_max") #hide

# ## Visualisation
using CairoMakie

nx, ny = N, N
xc = [coords[i, 1][1] for i in 1:nx]
yc = [coords[1, j][2] for j in 1:ny]
## Compute |B| from the solution
Bmag = [
    begin
            w = conserved_to_primitive(law, U[i, j])
            sqrt(w[6]^2 + w[7]^2)
        end for i in 1:nx, j in 1:ny
]
rho = [conserved_to_primitive(law, U[i, j])[1] for i in 1:nx, j in 1:ny]

fig = Figure(fontsize = 24, size = (1100, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "y",
    title = "|B| at t = $(round(t_final, digits = 2))", aspect = DataAspect()
)
hm1 = heatmap!(ax1, xc, yc, Bmag, colormap = :viridis)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(
    fig[1, 3], xlabel = "x", ylabel = "y",
    title = "Density", aspect = DataAspect()
)
hm2 = heatmap!(ax2, xc, yc, rho, colormap = :inferno)
Colorbar(fig[1, 4], hm2)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "field_loop_advection.png") fig #src

# The field loop has been advected across the domain. Since the magnetic
# field is very weak ($A_0 = 10^{-3}$), the density remains nearly uniform.
# The maximum $|\nabla\cdot\vb B| = $ $(round(divB_max, sigdigits=2))
# confirms that CT preserves the divergence-free constraint throughout
# the simulation.
rho_variation = maximum(rho) - minimum(rho) #hide
rho_variation < 0.01 || @warn("Density variation exceeds tolerance: $rho_variation") #hide

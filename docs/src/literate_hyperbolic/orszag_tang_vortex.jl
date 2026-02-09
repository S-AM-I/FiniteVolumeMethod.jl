using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Orszag-Tang Vortex
# The Orszag-Tang vortex is an iconic 2D MHD test problem that demonstrates
# the transition to MHD turbulence. Starting from smooth initial conditions,
# the flow develops a complex pattern of interacting shocks and current sheets.
#
# ## Problem Setup
# The domain is $[0, 1]^2$ with periodic boundary conditions and $\gamma = 5/3$.
# The initial condition is a superposition of velocity and magnetic field
# vortices:
# ```math
# \rho = \gamma^2, \quad P = \gamma, \quad v_x = -\sin(2\pi y), \quad v_y = \sin(2\pi x),
# ```
# ```math
# B_x = -\frac{\sin(2\pi y)}{\sqrt{4\pi}}, \quad B_y = \frac{\sin(4\pi x)}{\sqrt{4\pi}}.
# ```

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = IdealMHDEquations{2}(eos)

N = 50
mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, N)

function ot_ic(x, y)
    rho = gamma^2
    P = gamma
    vx = -sin(2 * pi * y)
    vy = sin(2 * pi * x)
    vz = 0.0
    Bx = -sin(2 * pi * y) / sqrt(4 * pi)
    By = sin(4 * pi * x) / sqrt(4 * pi)
    Bz = 0.0
    return SVector(rho, vx, vy, vz, P, Bx, By, Bz)
end

# ## Vector Potential Initialisation
# For MHD with constrained transport, we initialise the magnetic field
# using a vector potential $A_z(x,y)$ to guarantee $\nabla\cdot\vb B = 0$
# to machine precision. The vector potential satisfying
# $B_x = \partial A_z/\partial y$ and $B_y = -\partial A_z/\partial x$ is:
function Az_ot(x, y)
    return cos(2 * pi * y) / (2 * pi * sqrt(4 * pi)) +
           cos(4 * pi * x) / (4 * pi * sqrt(4 * pi))
end

# ## Solving
# We use the HLLD Riemann solver with MUSCL reconstruction and periodic BCs.
prob = HyperbolicProblem2D(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    ot_ic; final_time=0.5, cfl=0.4
)

# The `vector_potential` keyword tells `solve_hyperbolic` to initialise
# the face-centred magnetic field from $A_z$ via Stokes' theorem:
coords, U, t_final, ct = solve_hyperbolic(prob; vector_potential=Az_ot)
coords |> tc #hide

# ## Checking $\nabla\cdot\vb B$
# The constrained transport algorithm should keep $|\nabla\cdot\vb B|$
# at machine precision:
divB_max = max_divB(ct, mesh)

@assert divB_max < 1e-12 #hide

# ## Visualisation
using CairoMakie

nx, ny = N, N
xc = [coords[1][i] for i in 1:nx]
yc = [coords[2][j] for j in 1:ny]
rho = [conserved_to_primitive(law, U[i, j])[1] for i in 1:nx, j in 1:ny]
P_vals = [conserved_to_primitive(law, U[i, j])[5] for i in 1:nx, j in 1:ny]

fig = Figure(fontsize=24, size=(1100, 500))
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y",
           title="Density at t = $(round(t_final, digits=2))", aspect=DataAspect())
hm1 = heatmap!(ax1, xc, yc, rho, colormap=:viridis)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3], xlabel="x", ylabel="y",
           title="Pressure", aspect=DataAspect())
hm2 = heatmap!(ax2, xc, yc, P_vals, colormap=:magma)
Colorbar(fig[1, 4], hm2)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "orszag_tang_vortex.png") fig #src

# The density and pressure fields show the characteristic pattern of
# interacting shocks and current sheets. The maximum $|\nabla\cdot\vb B|$
# is $(round(divB_max, sigdigits=2)), confirming that constrained
# transport maintains the divergence-free constraint to machine precision.

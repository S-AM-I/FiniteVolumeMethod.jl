using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # SRMHD Cylindrical Blast Wave
# This tutorial demonstrates the 2D special relativistic MHD (SRMHD)
# solver on a cylindrical blast wave problem. A high-pressure region
# is placed at the centre of the domain, driving a relativistic
# expansion into a magnetised ambient medium. The magnetic field
# breaks the cylindrical symmetry, producing an elongated shock
# structure along the field direction.
#
# ## Problem Setup
# The domain is $[-0.5, 0.5]^2$ with transmissive boundaries.
# A uniform magnetic field $B_x = 0.5$ threads the domain.
# The initial pressure is $P = 1$ inside a circle of radius $r = 0.1$
# and $P = 0.01$ outside.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 4.0 / 3.0
eos = IdealGasEOS(gamma)
law = SRMHDEquations{2}(eos)

P_in = 1.0
P_out = 0.01
rho0 = 0.01
r_blast = 0.1
Bx0 = 0.5

function blast_ic(x, y)
    r = sqrt(x^2 + y^2)
    P = r < r_blast ? P_in : P_out
    return SVector(rho0, 0.0, 0.0, 0.0, P, Bx0, 0.0, 0.0)
end

# Vector potential for uniform $B_x$:
Az_blast(x, y) = Bx0 * y

# ## Solving
N = 100
mesh = StructuredMesh2D(-0.5, 0.5, -0.5, 0.5, N, N)

prob = HyperbolicProblem2D(
    law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    TransmissiveBC(), TransmissiveBC(),
    blast_ic; final_time = 0.4, cfl = 0.25
)

coords, U, t_final, ct = solve_hyperbolic(prob; vector_potential = Az_blast)
coords |> tc #hide

# ## Divergence Check
divB_max = max_divB(ct, mesh)
divB_max < 1.0e-10 || @warn("divB exceeds tolerance: $divB_max") #hide

# ## Visualisation
using CairoMakie

nx, ny = N, N
xc = [coords[i, 1][1] for i in 1:nx]
yc = [coords[1, j][2] for j in 1:ny]

rho = [conserved_to_primitive(law, U[i, j])[1] for i in 1:nx, j in 1:ny]
P = [conserved_to_primitive(law, U[i, j])[5] for i in 1:nx, j in 1:ny]

fig = Figure(fontsize = 20, size = (1000, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "y",
    title = "Density at t = $(round(t_final, digits = 3))",
    aspect = DataAspect()
)
hm1 = heatmap!(ax1, xc, yc, rho, colormap = :viridis)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(
    fig[1, 3], xlabel = "x", ylabel = "y",
    title = "Pressure",
    aspect = DataAspect()
)
hm2 = heatmap!(ax2, xc, yc, P, colormap = :inferno)
Colorbar(fig[1, 4], hm2)

resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "srmhd_cylindrical_blast.png") fig #src

# The blast wave is clearly elongated along the $x$-direction due to
# the uniform $B_x$ field, which channels the expansion. The
# $\nabla\cdot\vb B$ constraint is maintained at machine precision
# ($|\nabla\cdot\vb B|_{\max} = $ $(round(divB_max, sigdigits=2)))
# thanks to constrained transport.
all(isfinite, rho) || @warn("Non-finite densities detected") #hide
all(isfinite, P) || @warn("Non-finite pressures detected") #hide

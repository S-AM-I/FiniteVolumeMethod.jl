using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # MHD Rotor
# The MHD rotor problem (Balsara & Spicer, 1999) is a classic test for
# 2D MHD solvers. A dense, rapidly spinning disk is embedded in a
# uniform magnetic field. The magnetic pressure confines the rotor and
# launches torsional Alfvén waves that wrap around it, producing
# characteristic spiral structure.
#
# ## Problem Setup
# The domain is $[0, 1]^2$. A disk of radius $r_0 = 0.1$ centred at
# $(0.5, 0.5)$ has density $\rho = 10$ and toroidal velocity with
# $v_\phi = 2$. Outside a taper zone $r_1 = 0.115$ the fluid is at
# rest with $\rho = 1$. A uniform magnetic field $B_x = 5/\sqrt{4\pi}$
# threads the entire domain.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 1.4
eos = IdealGasEOS(gamma)
law = IdealMHDEquations{2}(eos)

r0 = 0.1
r1 = 0.115
B0 = 5.0 / sqrt(4.0 * π)
P0 = 1.0
rho_in = 10.0
rho_out = 1.0
vphi_max = 2.0

# The taper function smoothly transitions between the rotor interior
# and the ambient medium:
function rotor_ic(x, y)
    dx, dy = x - 0.5, y - 0.5
    r = sqrt(dx^2 + dy^2)
    if r < r0
        f = 1.0
    elseif r < r1
        f = (r1 - r) / (r1 - r0)
    else
        f = 0.0
    end
    rho = rho_in * f + rho_out * (1.0 - f)
    vphi = vphi_max * f
    ## Toroidal velocity: v_phi = (-sin(theta), cos(theta)) * vphi
    vx = r > 1.0e-14 ? -vphi * dy / r : 0.0
    vy = r > 1.0e-14 ? vphi * dx / r : 0.0
    return SVector(rho, vx, vy, 0.0, P0, B0, 0.0, 0.0)
end

# ## Vector Potential
# We initialise the uniform magnetic field via a vector potential
# $A_z(x,y) = B_0 \, y$ to ensure $\nabla\cdot\vb B = 0$ at machine
# precision through constrained transport.
Az_rotor(x, y) = B0 * y

# ## Solving
N = 100
mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, N)

prob = HyperbolicProblem2D(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    TransmissiveBC(), TransmissiveBC(),
    rotor_ic; final_time = 0.15, cfl = 0.3
)

coords, U, t_final, ct = solve_hyperbolic(prob; vector_potential = Az_rotor)
coords |> tc #hide

# ## Checking Divergence
divB_max = max_divB(ct, mesh)
@assert divB_max < 1.0e-12 #hide

# ## Visualisation
using CairoMakie

nx, ny = N, N
xc = [coords[i, 1][1] for i in 1:nx]
yc = [coords[1, j][2] for j in 1:ny]

rho = [conserved_to_primitive(law, U[i, j])[1] for i in 1:nx, j in 1:ny]
P = [conserved_to_primitive(law, U[i, j])[5] for i in 1:nx, j in 1:ny]
Bmag = [
    begin
            w = conserved_to_primitive(law, U[i, j])
            sqrt(w[6]^2 + w[7]^2)
        end for i in 1:nx, j in 1:ny
]

fig = Figure(fontsize = 20, size = (1400, 450))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "y",
    title = "Density", aspect = DataAspect()
)
hm1 = heatmap!(ax1, xc, yc, rho, colormap = :viridis)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(
    fig[1, 3], xlabel = "x", ylabel = "y",
    title = "Pressure", aspect = DataAspect()
)
hm2 = heatmap!(ax2, xc, yc, P, colormap = :inferno)
Colorbar(fig[1, 4], hm2)

ax3 = Axis(
    fig[1, 5], xlabel = "x", ylabel = "y",
    title = "|B|", aspect = DataAspect()
)
hm3 = heatmap!(ax3, xc, yc, Bmag, colormap = :plasma)
Colorbar(fig[1, 6], hm3)

resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "mhd_rotor.png") fig #src

# The density panel shows the spinning rotor being confined by the
# magnetic field. The pressure field reveals the magnetic pressure
# build-up around the rotor. The $|\vb B|$ panel shows the torsional
# Alfvén waves wrapping around the disk. The maximum
# $|\nabla\cdot\vb B| = $ $(round(divB_max, sigdigits=2)) confirms that
# constrained transport maintains the solenoidal constraint.
@assert all(rho .> 0) #hide
@assert all(P .> 0) #hide

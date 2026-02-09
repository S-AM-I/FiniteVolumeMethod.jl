using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Kelvin-Helmholtz Instability
# The Kelvin-Helmholtz instability (KHI) is a shear-driven hydrodynamic
# instability that develops at the interface between two fluid layers
# moving at different velocities. The instability rolls up into
# characteristic vortex structures.
#
# ## Problem Setup
# We solve the 2D Euler equations on $[0, 1]^2$ with periodic boundary
# conditions. The initial condition has two shear layers with a small
# velocity perturbation to seed the instability:

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{2}(eos)

## Shear layer parameters
rho1 = 1.0    ## density of the inner layer
rho2 = 2.0    ## density of the outer layers
v_shear = 0.5
P0 = 2.5
A_pert = 0.01 ## perturbation amplitude

function kh_ic(x, y)
    ## Two shear layers at y = 0.25 and y = 0.75
    if 0.25 < y < 0.75
        rho = rho2
        vx = v_shear
    else
        rho = rho1
        vx = -v_shear
    end
    ## Smooth the density transition
    sigma = 0.05
    rho = rho1 + (rho2 - rho1) * (
        0.5 * (1.0 + tanh((y - 0.25) / sigma)) -
        0.5 * (1.0 + tanh((y - 0.75) / sigma))
    )
    vx_smooth = -v_shear + 2 * v_shear * (
        0.5 * (1.0 + tanh((y - 0.25) / sigma)) -
        0.5 * (1.0 + tanh((y - 0.75) / sigma))
    )
    ## Sinusoidal perturbation in vy to seed the instability
    vy = A_pert * sin(4 * pi * x)
    return SVector(rho, vx_smooth, vy, P0)
end

# ## Solving at Two Resolutions
# We compare $N=64$ and $N=128$ to show how resolution affects
# the development of the instability.
t_final = 1.0

N_low = 64
mesh_low = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N_low, N_low)
prob_low = HyperbolicProblem2D(
    law, mesh_low, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    kh_ic; final_time=t_final, cfl=0.4
)
coords_low, U_low, _ = solve_hyperbolic(prob_low)

N_high = 128
mesh_high = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N_high, N_high)
prob_high = HyperbolicProblem2D(
    law, mesh_high, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    kh_ic; final_time=t_final, cfl=0.4
)
coords_high, U_high, t_end = solve_hyperbolic(prob_high)
coords_high |> tc #hide

# ## Visualisation
using CairoMakie

fig = Figure(fontsize=24, size=(1100, 500))

ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y",
           title="Density (N=$(N_low))", aspect=DataAspect())
xc_l = [coords_low[1][i] for i in 1:N_low]
yc_l = [coords_low[2][j] for j in 1:N_low]
rho_low = [conserved_to_primitive(law, U_low[i, j])[1] for i in 1:N_low, j in 1:N_low]
hm1 = heatmap!(ax1, xc_l, yc_l, rho_low, colormap=:viridis)

ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y",
           title="Density (N=$(N_high))", aspect=DataAspect())
xc_h = [coords_high[1][i] for i in 1:N_high]
yc_h = [coords_high[2][j] for j in 1:N_high]
rho_high = [conserved_to_primitive(law, U_high[i, j])[1] for i in 1:N_high, j in 1:N_high]
hm2 = heatmap!(ax2, xc_h, yc_h, rho_high, colormap=:viridis)
Colorbar(fig[1, 3], hm2)

resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "kelvin_helmholtz_instability.png") fig #src

# At higher resolution, the vortex roll-up is more finely resolved
# and secondary instabilities become visible. The KHI is an excellent
# test for the interplay between numerical dissipation and physical
# instability growth.
@assert all(rho_high .> 0) #hide

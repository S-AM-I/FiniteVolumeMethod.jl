using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Balsara SRMHD Shock Tube
# The Balsara 1 shock tube is the relativistic analogue of the Brio-Wu MHD
# test. It tests the special relativistic MHD (SRMHD) solver, including
# the iterative conserved-to-primitive conversion that accounts for the
# Lorentz factor.
#
# ## Problem Setup
# The SRMHD equations evolve 8 conserved variables
# $(D, S_x, S_y, S_z, \tau, B_x, B_y, B_z)$ where
# $D = \rho W$ (relativistic mass density), $W = 1/\sqrt{1-|\vb v|^2}$
# is the Lorentz factor, and $\tau$ is the energy density minus rest mass.
#
# The initial primitive states $(\rho, v_x, v_y, v_z, P, B_x, B_y, B_z)$ are:

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = SRMHDEquations{1}(eos)

wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)

N = 400
mesh = StructuredMesh1D(0.0, 1.0, N)
ic(x) = x < 0.5 ? wL : wR

# ## Solving
# We use the HLL solver with MUSCL reconstruction.
prob = HyperbolicProblem(
    law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.4, cfl = 0.4
)
x, U, t_final = solve_hyperbolic(prob)
x |> tc #hide

# ## Extracting Primitive Variables
# The SRMHD conserved-to-primitive conversion is iterative because the
# Lorentz factor $W$ depends on velocity, which depends on the conserved
# variables in a nonlinear way.
rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
vx = [conserved_to_primitive(law, U[i])[2] for i in eachindex(U)]
P = [conserved_to_primitive(law, U[i])[5] for i in eachindex(U)]
By = [conserved_to_primitive(law, U[i])[7] for i in eachindex(U)]

# ## Visualisation
using CairoMakie

fig = Figure(fontsize = 20, size = (1200, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho", title = "Density")
lines!(ax1, x, rho, color = :blue, linewidth = 1.5)

ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = L"v_x", title = "Velocity")
lines!(ax2, x, vx, color = :blue, linewidth = 1.5)

ax3 = Axis(fig[1, 3], xlabel = "x", ylabel = L"B_y", title = "Magnetic Field (y)")
lines!(ax3, x, By, color = :blue, linewidth = 1.5)

resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "balsara_srmhd_shock_tube.png") fig #src

# The solution shows the relativistic wave structure with compound MHD
# waves similar to the Brio-Wu test, but modified by relativistic effects.
# All velocities remain subluminal ($|v| < 1$) as required by special
# relativity.
@assert all(rho .> 0) #hide
@assert all(isfinite, rho) #hide
@assert all(P .> 0) #hide

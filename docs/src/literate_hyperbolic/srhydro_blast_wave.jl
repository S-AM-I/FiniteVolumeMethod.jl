using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Relativistic Sod Shock Tube (SR Hydro)
# The special relativistic Sod problem extends the classical
# Sod shock tube to relativistic fluid dynamics. It tests the
# ability of the solver to handle relativistic wave speeds and
# the iterative conservative-to-primitive (con2prim) recovery.
#
# ## Problem Setup
# The 1D special relativistic hydrodynamics equations conserve
# $(\rho W, \rho h W^2 v, \rho h W^2 - P - \rho W)$ where
# $W = 1/\sqrt{1 - v^2}$ is the Lorentz factor and $h$ is the
# specific enthalpy.
#
# Initial conditions (primitive $[\rho, v, P]$):
# ```math
# (\rho, v, P) = \begin{cases}(1, 0, 1) & x < 0.5,\\(0.125, 0, 0.1) & x \geq 0.5.\end{cases}
# ```

# We begin by loading the package and defining the conservation law.
using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

eos = IdealGasEOS(5.0 / 3.0)
law = SRHydroEquations{1}(eos)

# Define left and right primitive states $(\rho, v, P)$:
wL = SVector(1.0, 0.0, 1.0)
wR = SVector(0.125, 0.0, 0.1)

# Set up the mesh and boundary conditions:
N = 400
mesh = StructuredMesh1D(0.0, 1.0, N)

ic(x) = x < 0.5 ? wL : wR

# ## Solving with HLL
# We use the HLL Riemann solver with first-order reconstruction
# for robustness. The CFL is kept low (0.3) to respect the
# relativistic signal speeds.
prob = HyperbolicProblem(
    law, mesh, HLLSolver(), NoReconstruction(),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.2, cfl = 0.3,
)
x, U, t = solve_hyperbolic(prob)
x |> tc #hide

# ## Visualisation
# Extract primitive variables and compute the Lorentz factor
# $W = 1/\sqrt{1 - v^2}$.
using CairoMakie

W_prim = to_primitive(law, U)
rho_vals = [W_prim[i][1] for i in eachindex(W_prim)]
v_vals = [W_prim[i][2] for i in eachindex(W_prim)]
P_vals = [W_prim[i][3] for i in eachindex(W_prim)]
gamma_vals = [1.0 / sqrt(1.0 - W_prim[i][2]^2) for i in eachindex(W_prim)]

fig = Figure(fontsize = 24, size = (1200, 800))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho", title = "Density")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = "v", title = "Velocity")
ax3 = Axis(fig[2, 1], xlabel = "x", ylabel = "P", title = "Pressure")
ax4 = Axis(fig[2, 2], xlabel = "x", ylabel = "W", title = "Lorentz Factor")
scatter!(ax1, x, rho_vals, color = :blue, markersize = 3)
scatter!(ax2, x, v_vals, color = :red, markersize = 3)
scatter!(ax3, x, P_vals, color = :green, markersize = 3)
scatter!(ax4, x, gamma_vals, color = :purple, markersize = 3)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "srhydro_blast_wave.png") fig #src

# The wave structure is qualitatively similar to the classical case
# but the contact discontinuity and shock are compressed toward the
# right boundary due to relativistic effects.

# ## Physical Checks
# Density and pressure must be positive, and all velocities must
# remain subluminal ($|v| < 1$).
@test all(W_prim[i][1] > 0.0 for i in eachindex(W_prim)) #src
@test all(W_prim[i][3] > 0.0 for i in eachindex(W_prim)) #src
@test all(abs(W_prim[i][2]) < 1.0 for i in eachindex(W_prim)) #src

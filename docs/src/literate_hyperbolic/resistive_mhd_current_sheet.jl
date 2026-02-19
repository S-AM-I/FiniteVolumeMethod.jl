using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Resistive MHD Current Sheet
# This tutorial demonstrates the `ResistiveMHDEquations` type by
# setting up a current sheet — a thin layer where the magnetic
# field reverses direction. In fully resistive MHD this configuration
# would undergo reconnection on a diffusive time scale $\tau \sim L^2/\eta$.
#
# The `solve_hyperbolic` solver evolves only the ideal (hyperbolic) MHD
# part. The resistive terms (`resistive_flux_x`, `ohmic_heating`) are
# provided as standalone utility functions for operator-split or IMEX
# integration. Here we demonstrate both the ideal evolution and the
# resistive utility API.
#
# ## Problem Setup
# The 8-variable MHD system is used with a $B_y$ reversal at $x = 0.5$.
# Uniform density $\rho = 1$, pressure $P = 1$, and $B_x = 0.75$
# (a guide field).

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

eos = IdealGasEOS(5.0 / 3.0)
law = ResistiveMHDEquations{1}(eos; eta = 0.01)

# Initial condition: $B_y$ reverses across the midpoint.
function current_sheet_ic(x)
    rho = 1.0
    vx, vy, vz = 0.0, 0.0, 0.0
    P = 1.0
    Bx = 0.75
    By = x < 0.5 ? 1.0 : -1.0
    Bz = 0.0
    return SVector(rho, vx, vy, vz, P, Bx, By, Bz)
end

N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)

prob = HyperbolicProblem(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), current_sheet_ic;
    final_time = 0.1, cfl = 0.4,
)
x, U, t = solve_hyperbolic(prob)
x |> tc #hide

# ## Computing Current Density and Resistive Heating
# The current density $J_z \approx \partial B_y / \partial x$ and
# Ohmic heating rate $\eta |J|^2$ are computed from the cell data.
W = to_primitive(law, U)
By_vals = [W[i][7] for i in eachindex(W)]
dx = 1.0 / N

## Approximate current density at cell centers via central differences
J_vals = zeros(length(By_vals))
for i in 2:(length(By_vals) - 1)
    J_vals[i] = (By_vals[i + 1] - By_vals[i - 1]) / (2 * dx)
end
J_vals[1] = (By_vals[2] - By_vals[1]) / dx
J_vals[end] = (By_vals[end] - By_vals[end - 1]) / dx

## Ohmic heating rate at each cell
Q_vals = [ohmic_heating(law, J_vals[i]^2) for i in eachindex(J_vals)]

## Demonstrate resistive_flux_x between two adjacent cells
i_mid = N ÷ 2
flux_demo = resistive_flux_x(law, U[i_mid], U[i_mid + 1], dx)

# ## Visualisation
using CairoMakie

fig = Figure(fontsize = 24, size = (1200, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"B_y", title = "Magnetic Field By")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = L"J_z", title = "Current Density")
ax3 = Axis(fig[1, 3], xlabel = "x", ylabel = L"\eta |J|^2", title = "Ohmic Heating Rate")
scatter!(ax1, x, By_vals, color = :blue, markersize = 4)
scatter!(ax2, x, J_vals, color = :red, markersize = 4)
scatter!(ax3, x, Q_vals, color = :orange, markersize = 4)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "resistive_mhd_current_sheet.png") fig #src

# The ideal MHD evolution produces fast magnetosonic waves that
# propagate away from the current sheet. The current density $J_z$
# peaks at the reversal layer. The Ohmic heating rate $\eta |J|^2$
# shows where magnetic energy would be dissipated in a resistive run.

# ## Physical Checks
@test all(W[i][1] > 0.0 for i in eachindex(W)) #src
@test all(W[i][5] > 0.0 for i in eachindex(W)) #src
@test all(abs(W[i][7]) <= 1.5 for i in eachindex(W)) #src

using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Premixed Flame 1D
# This example demonstrates a one-dimensional premixed flame using
# the `ReactiveEulerEquations` with single-step Arrhenius chemistry
# and Strang operator splitting.
#
# ## Mathematical Setup
# A single irreversible reaction F → P with Arrhenius kinetics
# releases heat into the flow. The initial condition has fuel
# on the left (cold, unburnt) and product on the right (hot, burnt).
# The flame front propagates from right to left as fuel is consumed.
#
# ## Inputs
# - **Resolution**: $N = 200$
# - **Reaction**: $A = 5 \times 10^3$, $E_a = 8$, $q = 2$
# - **EOS**: Ideal gas, $\gamma = 1.4$
# - **Solver**: HLLC + MUSCL(Minmod) + Strang splitting

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = ReactiveEulerEquations{1}(eos, (:fuel, :product))

# ## Initial Condition
# Smooth transition from unburnt (fuel) to burnt (product) state.
function flame_ic(x)
    rho = 1.0
    v = 0.0
    ## Smooth pressure/temperature profile: hot on right
    P = 1.0 + 2.0 * 0.5 * (1.0 + tanh(50.0 * (x - 0.7)))
    ## Fuel mass fraction: 1 on left, 0 on right
    Y_fuel = 0.5 * (1.0 - tanh(50.0 * (x - 0.7)))
    Y_product = 1.0 - Y_fuel
    return SVector(rho, v, P, Y_fuel, Y_product)
end

# ## Chemistry
rxn = ArrheniusReaction{2}(
    5.0e3,          # Pre-exponential factor
    0.0,            # Temperature exponent
    8.0,            # Activation energy / R
    (1.0, 0.0),     # Reactant: fuel consumed
    (0.0, 1.0),     # Product: product created
    2.0,            # Heat release per unit mass
)
mech = ReactionMechanism{2, 1}((rxn,), (1.0, 1.0))
chem = ChemistrySource(mech; mu_mol = 1.0)

# ## Solve
N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)
t_final = 0.05
prob = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), flame_ic;
    final_time = t_final, cfl = 0.3
)

x, U, t_end = solve_coupled(prob, chem; splitting = StrangSplitting())

# ## Extract Profiles
rho_vals = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
v_vals = [conserved_to_primitive(law, U[i])[2] for i in eachindex(U)]
P_vals = [conserved_to_primitive(law, U[i])[3] for i in eachindex(U)]
T_vals = P_vals ./ rho_vals
Y_fuel_vals = [conserved_to_primitive(law, U[i])[4] for i in eachindex(U)]
Y_prod_vals = [conserved_to_primitive(law, U[i])[5] for i in eachindex(U)]

# ## Visualisation — Flame Structure
fig = Figure(fontsize = 20, size = (1200, 800))

ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho", title = "Density")
lines!(ax1, x, rho_vals, color = :blue, linewidth = 2)

ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = "T = P/ρ", title = "Temperature")
lines!(ax2, x, T_vals, color = :red, linewidth = 2)

ax3 = Axis(fig[2, 1], xlabel = "x", ylabel = "Y", title = "Species Mass Fractions")
lines!(ax3, x, Y_fuel_vals, color = :orange, linewidth = 2, label = "Fuel")
lines!(ax3, x, Y_prod_vals, color = :green, linewidth = 2, label = "Product")
axislegend(ax3, position = :rc)

ax4 = Axis(fig[2, 2], xlabel = "x", ylabel = "P", title = "Pressure")
lines!(ax4, x, P_vals, color = :purple, linewidth = 2)

resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "premixed_flame_1d.png") fig #src

# ## Test Assertions
## Density and pressure remain positive
@test all(rho_vals .> 0.0) #src
@test all(P_vals .> 0.0) #src
@assert all(rho_vals .> 0.0) #hide
@assert all(P_vals .> 0.0) #hide

## Some fuel has been consumed (Y_fuel < 1 somewhere in the domain)
@test minimum(Y_fuel_vals) < 0.5 #src
@assert minimum(Y_fuel_vals) < 0.5 #hide

## Temperature increased in the reaction zone
@test maximum(T_vals) > 2.0 #src
@assert maximum(T_vals) > 2.0 #hide

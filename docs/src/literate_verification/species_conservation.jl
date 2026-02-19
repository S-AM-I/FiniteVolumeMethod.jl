using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Species Conservation Verification
# This example verifies that the reactive Euler solver conserves the total
# mass of each species to machine precision when using periodic boundary
# conditions, both with and without chemical reactions.
#
# ## Mathematical Setup
# Three species (fuel, oxidizer, product) are advected on a periodic
# domain. Without reactions, each species' total mass $\int \rho Y_k\,dx$
# is individually conserved. With reactions, total mass $\int \rho\,dx$
# and total species mass $\sum_k \int \rho Y_k\,dx$ are conserved.
#
# ## Inputs
# - **Resolution**: $N = 100$
# - **3 species**: fuel, oxidizer, product
# - **Periodic BCs**
# - **Solver**: HLLC + MUSCL(Minmod)

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = ReactiveEulerEquations{1}(eos, (:fuel, :oxidizer, :product))

N = 100
mesh = StructuredMesh1D(0.0, 1.0, N)
dx = mesh.dx
t_final = 0.3

# ## Initial Condition with 3 Species
function three_species_ic(x)
    rho = 1.0 + 0.15 * sin(2 * pi * x)
    v = 0.5
    P = 1.0
    Y_fuel = 0.5 + 0.2 * cos(2 * pi * x)
    Y_ox = 0.3 + 0.1 * sin(4 * pi * x)
    Y_prod = 1.0 - Y_fuel - Y_ox
    return SVector(rho, v, P, Y_fuel, Y_ox, Y_prod)
end

# ## Case 1: No Reactions — Individual Conservation
prob_noreact = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), three_species_ic;
    final_time = t_final, cfl = 0.4
)

## Compute initial totals
U0 = FiniteVolumeMethod.initialize_1d(prob_noreact)
nc = ncells(mesh)
mass0_total = sum(U0[i + 2][1] * dx for i in 1:nc)
mass0_fuel = sum(U0[i + 2][4] * dx for i in 1:nc)
mass0_ox = sum(U0[i + 2][5] * dx for i in 1:nc)
mass0_prod = sum(U0[i + 2][6] * dx for i in 1:nc)

x, U_nr, t_nr = solve_hyperbolic(prob_noreact)

mass_total_nr = sum(U_nr[i][1] * dx for i in 1:nc)
mass_fuel_nr = sum(U_nr[i][4] * dx for i in 1:nc)
mass_ox_nr = sum(U_nr[i][5] * dx for i in 1:nc)
mass_prod_nr = sum(U_nr[i][6] * dx for i in 1:nc)

err_total_nr = abs(mass_total_nr - mass0_total) / abs(mass0_total)
err_fuel_nr = abs(mass_fuel_nr - mass0_fuel) / abs(mass0_fuel)
err_ox_nr = abs(mass_ox_nr - mass0_ox) / abs(mass0_ox)
err_prod_nr = abs(mass_prod_nr - mass0_prod) / abs(mass0_prod)

# ## Case 2: With Reactions — Total Mass Conservation
rxn = ArrheniusReaction{3}(
    500.0, 0.0, 5.0,
    (1.0, 0.5, 0.0),    # fuel + 0.5 oxidizer consumed
    (0.0, 0.0, 1.5),    # 1.5 product created (mass balance: 1 + 0.5 = 1.5)
    0.5,
)
mech = ReactionMechanism{3, 1}((rxn,), (1.0, 1.0, 1.0))
chem = ChemistrySource(mech; mu_mol = 1.0)

prob_react = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), three_species_ic;
    final_time = 0.1, cfl = 0.3
)

x_r, U_r, t_r = solve_coupled(prob_react, chem; splitting = StrangSplitting())

mass_total_r = sum(U_r[i][1] * dx for i in 1:nc)
## Sum of all species partial densities
mass_species_r = sum(U_r[i][4] + U_r[i][5] + U_r[i][6] for i in 1:nc) * dx

err_total_r = abs(mass_total_r - mass0_total) / abs(mass0_total)
err_species_r = abs(mass_species_r - mass0_total) / abs(mass0_total)

# ## Visualisation
fig = Figure(fontsize = 20, size = (1100, 450))

ax1 = Axis(
    fig[1, 1], xlabel = "Species", ylabel = "Relative error",
    yscale = log10, title = "No reactions: individual conservation"
)
errs_nr = [err_total_nr, err_fuel_nr, err_ox_nr, err_prod_nr]
errs_nr_plot = max.(errs_nr, 1.0e-16)
barplot!(ax1, 1:4, errs_nr_plot, color = :steelblue)
hlines!(ax1, [eps(Float64)], color = :red, linestyle = :dash, linewidth = 1, label = "Machine ε")
ax1.xticks = (1:4, ["Total ρ", "ρY_fuel", "ρY_ox", "ρY_prod"])
axislegend(ax1, position = :rt)

ax2 = Axis(
    fig[1, 2], xlabel = "Quantity", ylabel = "Relative error",
    yscale = log10, title = "With reactions: total mass conservation"
)
errs_r = [err_total_r, err_species_r]
errs_r_plot = max.(errs_r, 1.0e-16)
barplot!(ax2, 1:2, errs_r_plot, color = :coral)
hlines!(ax2, [eps(Float64)], color = :red, linestyle = :dash, linewidth = 1, label = "Machine ε")
ax2.xticks = (1:2, ["Total ρ", "Σ ρY_k"])
axislegend(ax2, position = :rt)

resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "species_conservation.png") fig #src

# ## Test Assertions
## No reactions: individual species conserved
@test err_total_nr < 1.0e-10 #src
@test err_fuel_nr < 1.0e-10 #src
@test err_ox_nr < 1.0e-10 #src
@test err_prod_nr < 1.0e-10 #src
@assert err_total_nr < 1.0e-10 #hide
@assert err_fuel_nr < 1.0e-10 #hide
@assert err_ox_nr < 1.0e-10 #hide
@assert err_prod_nr < 1.0e-10 #hide

## With reactions: total mass conserved
@test err_total_r < 1.0e-10 #src
@assert err_total_r < 1.0e-10 #hide

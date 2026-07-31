# # Eddy Dissipation Model Invariants
# This case benchmarks the turbulence-chemistry interaction source term of
# the Magnussen & Hjertager (1977) Eddy Dissipation Model,
# `compute_edm_reaction_rates`:
# ```math
# \dot\omega_{\text{fuel}} = -\rho A \frac{\varepsilon}{k}
# \min\!\left(Y_{\text{fuel}}, \frac{Y_{\text{ox}}}{s}\right), \qquad
# \dot\omega_{\text{ox}} = s\,\dot\omega_{\text{fuel}}, \qquad
# \dot\omega_{\text{prod}} = -(1 + s)\,\dot\omega_{\text{fuel}}.
# ```
# Given prescribed $Y$, $k$, $\varepsilon$, the rates are determined to
# machine precision, so five exact invariants can be checked:
# fuel-limited and oxidizer-limited branches, the stoichiometric mass
# balance $\sum_i \dot\omega_i \equiv 0$, the $\varepsilon/k$ mixing-rate
# proportionality, and the heat release
# $S_h = -\dot\omega_{\text{fuel}} \Delta H > 0$.
#
# ## Acceptance Gates
# - Both limiting branches match the closed form to relative $10^{-12}$
# - $|\sum_i \dot\omega_i| < 10^{-14}$ per cell
# - Doubling $\varepsilon/k$ doubles the rate to $10^{-12}$
# - $S_h = -\dot\omega_{\text{fuel}} \Delta H$ to $10^{-12}$, positive

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_edm_reaction_rates, compute_heat_release
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
n_cells = length(mesh.cell_volumes)
edm = EddyDissipationModel(; A_edm = 4.0, B_edm = 0.5)

# ## Fuel-Limited Branch
props = CombustionProperties(;
    species_names = (:fuel, :oxidizer, :product),
    molecular_weights = (16.0, 32.0, 44.0),
    stoich_ratio = 4.0,
    heat_of_combustion = 5.0e7,
)
species_fuel = SpeciesState(mesh, props; fuel = 0.01, oxidizer = 0.25, product = 0.0)
omega_fuel_branch = compute_edm_reaction_rates(
    edm, species_fuel, props, fill(1.0, n_cells), fill(0.5, n_cells), 1.2, mesh
)
fuel_expected = -1.2 * 4.0 * 0.5 * 0.01
fuel_branch_error = maximum(
    abs(omega_fuel_branch[1][c] - fuel_expected) / abs(fuel_expected) for c in 1:n_cells
)
stoich_error = maximum(1:n_cells) do c
    max(
        abs(omega_fuel_branch[2][c] - 4.0 * omega_fuel_branch[1][c]),
        abs(omega_fuel_branch[3][c] + 5.0 * omega_fuel_branch[1][c]),
    ) / abs(omega_fuel_branch[1][c])
end

# ## Oxidizer-Limited Branch
props_s4 = CombustionProperties(; stoich_ratio = 4.0)
species_ox = SpeciesState(mesh, props_s4; fuel = 0.5, oxidizer = 0.04, product = 0.0)
omega_ox_branch = compute_edm_reaction_rates(
    EddyDissipationModel(; A_edm = 4.0), species_ox, props_s4,
    fill(2.0, n_cells), fill(1.0, n_cells), 1.0, mesh
)
ox_expected = -1.0 * 4.0 * 0.5 * 0.01
ox_branch_error = maximum(
    abs(omega_ox_branch[1][c] - ox_expected) / abs(ox_expected) for c in 1:n_cells
)

# ## Mass Balance
species_mixed = SpeciesState(mesh, props_s4; fuel = 0.2, oxidizer = 0.3, product = 0.1)
omega_mixed = compute_edm_reaction_rates(
    edm, species_mixed, props_s4, fill(1.5, n_cells), fill(0.8, n_cells), 1.2, mesh
)
mass_balance = maximum(
    abs(omega_mixed[1][c] + omega_mixed[2][c] + omega_mixed[3][c]) for c in 1:n_cells
)

# ## Mixing-Rate Proportionality
species_mix = SpeciesState(mesh, props_s4; fuel = 0.05, oxidizer = 0.3, product = 0.0)
edm_a4 = EddyDissipationModel(; A_edm = 4.0)
omega_r1 = compute_edm_reaction_rates(
    edm_a4, species_mix, props_s4, fill(1.0, n_cells), fill(0.5, n_cells), 1.0, mesh
)
omega_r2 = compute_edm_reaction_rates(
    edm_a4, species_mix, props_s4, fill(1.0, n_cells), fill(1.0, n_cells), 1.0, mesh
)
proportionality_error = maximum(
    abs(omega_r2[1][c] / omega_r1[1][c] - 2.0) / 2.0 for c in 1:n_cells
)

# ## Heat Release
dH = 5.0e7
props_hr = CombustionProperties(; stoich_ratio = 4.0, heat_of_combustion = dH)
species_hr = SpeciesState(mesh, props_hr; fuel = 0.02, oxidizer = 0.2, product = 0.0)
omega_hr = compute_edm_reaction_rates(
    edm_a4, species_hr, props_hr, fill(1.0, n_cells), fill(0.5, n_cells), 1.0, mesh
)
S_h = compute_heat_release(omega_hr, props_hr)
heat_positive = all(>(0.0), S_h)
heat_identity_error = maximum(
    abs(S_h[c] + omega_hr[1][c] * dH) / abs(omega_hr[1][c] * dH) for c in 1:n_cells
)

# ## Visualisation — Rates and Balance
fig1 = Figure(fontsize = 24, size = (700, 450))
ax1 = Axis(
    fig1[1, 1], ylabel = "rate", xticks = (1:3, ["ω_fuel", "ω_ox", "ω_prod"]),
    title = "EDM stoichiometric rates (fuel-limited)"
)
barplot!(
    ax1, 1:3,
    [omega_fuel_branch[1][1], omega_fuel_branch[2][1], omega_fuel_branch[3][1]],
    color = [:firebrick, :steelblue, :seagreen]
)
hlines!(ax1, [0.0], color = :black)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("edm_invariants.png"), fig1)
end

# ## Acceptance
@test fuel_branch_error < 1.0e-12 #src
@test stoich_error < 1.0e-12 #src
@test ox_branch_error < 1.0e-12 #src
@test mass_balance < 1.0e-14 #src
@test proportionality_error < 1.0e-12 #src
@test heat_positive #src
@test heat_identity_error < 1.0e-12 #src
@assert fuel_branch_error < 1.0e-12 #hide
@assert stoich_error < 1.0e-12 #hide
@assert ox_branch_error < 1.0e-12 #hide
@assert mass_balance < 1.0e-14 #hide
@assert proportionality_error < 1.0e-12 #hide
@assert heat_positive #hide
@assert heat_identity_error < 1.0e-12 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "fuel_branch_error" => fuel_branch_error,
            "ox_branch_error" => ox_branch_error,
            "stoichiometry_error" => stoich_error,
            "mass_balance" => mass_balance,
            "proportionality_error" => proportionality_error,
            "heat_identity_error" => heat_identity_error,
        ),
        artifacts = ["edm_invariants.png"],
        notes = [
            "Benchmark-stage evidence for combustion: the Magnussen-Hjertager (1977) EDM closure satisfies its closed-form limiting branches, stoichiometric mass balance, mixing-rate proportionality, and heat-release identity to machine precision.",
        ],
        summary = Dict(
            "A_edm" => 4.0,
            "stoich_ratio" => 4.0,
        ),
    )
end

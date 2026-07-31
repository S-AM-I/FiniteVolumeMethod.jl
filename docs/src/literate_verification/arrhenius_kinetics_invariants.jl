# # Arrhenius Kinetics Invariants
# This case verifies the finite-rate chemistry closure
# `compute_arrhenius_reaction_rates`:
# ```math
# k_f(T) = A\,T^b e^{-E_a/(R T)}, \qquad
# \dot\omega_{\text{fuel}} = -\rho\,k_f\,Y_{\text{fuel}}^{n_f}\,Y_{\text{ox}}^{n_o},
# ```
# against five exact algebraic identities — including the exponential
# temperature sensitivity that distinguishes finite-rate kinetics from the
# mixing-limited EDM closure.
#
# ## Acceptance Gates
# - Zero fuel or oxidizer ⇒ $\dot\omega \equiv 0$ exactly
# - Closed-form rate and stoichiometric ratios to relative $10^{-12}$
# - Analytic $T$-sensitivity ratio
#   $\exp\!\left(\tfrac{E_a}{R}(1/T_1 - 1/T_2)\right)$ to $10^{-12}$
# - Linearity in the pre-exponential $A$ to $10^{-12}$
# - Low-temperature clamp: $T < 200$ K behaves as $T = 200$ K

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_arrhenius_reaction_rates
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

R_univ = 8.314
mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
n_cells = length(mesh.cell_volumes)
props = CombustionProperties(; stoich_ratio = 4.0)

T_at(value) = CollocatedScalarField(:T, mesh; value = value)

# ## Zero-Species Cutoff
reaction = CollocatedArrheniusReaction(; A = 1.0e10, E_a = 1.0e5)
omega_nofuel = compute_arrhenius_reaction_rates(
    reaction, SpeciesState(mesh, props; fuel = 0.0, oxidizer = 0.5, product = 0.0),
    props, T_at(1500.0), 1.0, mesh
)
omega_noox = compute_arrhenius_reaction_rates(
    reaction, SpeciesState(mesh, props; fuel = 0.5, oxidizer = 0.0, product = 0.0),
    props, T_at(1500.0), 1.0, mesh
)
cutoff_exact = all(omega_nofuel[1][c] == 0.0 for c in 1:n_cells) &&
    all(omega_noox[1][c] == 0.0 for c in 1:n_cells)

# ## Closed-Form Identity
A_pre = 1.0e10
b_exp = 0.5
E_a = 1.0e5
T_val = 1500.0
Y_f = 0.08
Y_o = 0.2
rho = 1.2
reaction_full = CollocatedArrheniusReaction(;
    A = A_pre, b = b_exp, E_a = E_a, n_fuel = 1.0, n_ox = 1.0,
)
omega = compute_arrhenius_reaction_rates(
    reaction_full, SpeciesState(mesh, props; fuel = Y_f, oxidizer = Y_o, product = 0.0),
    props, T_at(T_val), rho, mesh
)
k_f = A_pre * T_val^b_exp * exp(-E_a / (R_univ * T_val))
omega_expected = -rho * k_f * Y_f * Y_o
identity_error = maximum(
    abs(omega[1][c] - omega_expected) / abs(omega_expected) for c in 1:n_cells
)
stoich_error = maximum(1:n_cells) do c
    max(
        abs(omega[2][c] - 4.0 * omega[1][c]) / abs(4.0 * omega[1][c]),
        abs(omega[3][c] + 5.0 * omega[1][c]) / abs(5.0 * omega[1][c]),
    )
end

# ## Temperature Sensitivity
reaction_b0 = CollocatedArrheniusReaction(; A = 1.0e10, b = 0.0, E_a = 1.0e5)
state_T = SpeciesState(mesh, props; fuel = 0.1, oxidizer = 0.3, product = 0.0)
omega_low = compute_arrhenius_reaction_rates(reaction_b0, state_T, props, T_at(1000.0), 1.0, mesh)
omega_high = compute_arrhenius_reaction_rates(reaction_b0, state_T, props, T_at(2000.0), 1.0, mesh)
ratio_expected = exp(1.0e5 / R_univ * (1.0 / 1000.0 - 1.0 / 2000.0))
sensitivity_error = maximum(
    abs(omega_high[1][c] / omega_low[1][c] - ratio_expected) / ratio_expected
        for c in 1:n_cells
)

# ## A-Linearity and Low-T Clamp
omega_A1 = compute_arrhenius_reaction_rates(
    CollocatedArrheniusReaction(; A = 1.0e10, E_a = 1.0e5), state_T, props,
    T_at(1500.0), 1.0, mesh
)
omega_A2 = compute_arrhenius_reaction_rates(
    CollocatedArrheniusReaction(; A = 2.0e10, E_a = 1.0e5), state_T, props,
    T_at(1500.0), 1.0, mesh
)
linearity_error = maximum(
    abs(omega_A2[1][c] / omega_A1[1][c] - 2.0) / 2.0 for c in 1:n_cells
)

omega_T100 = compute_arrhenius_reaction_rates(reaction, state_T, props, T_at(100.0), 1.0, mesh)
omega_T200 = compute_arrhenius_reaction_rates(reaction, state_T, props, T_at(200.0), 1.0, mesh)
clamp_error = maximum(
    abs(omega_T100[1][c] - omega_T200[1][c]) / max(abs(omega_T200[1][c]), 1.0e-300)
        for c in 1:n_cells
)

# ## Visualisation — Arrhenius Line
T_sweep = range(800.0, 2400.0; length = 30)
rates = map(T_sweep) do T
    om = compute_arrhenius_reaction_rates(reaction_b0, state_T, props, T_at(T), 1.0, mesh)
    -om[1][1]
end

fig1 = Figure(fontsize = 24, size = (600, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "1000 / T [1/K]", ylabel = "|ω_fuel|",
    yscale = log10, title = "Arrhenius line (b = 0)"
)
scatterlines!(
    ax1, 1000.0 ./ collect(T_sweep), rates, marker = :circle, color = :blue,
    linewidth = 2, markersize = 8
)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("arrhenius_kinetics.png"), fig1)
end

# ## Acceptance
@test cutoff_exact #src
@test identity_error < 1.0e-12 #src
@test stoich_error < 1.0e-12 #src
@test sensitivity_error < 1.0e-12 #src
@test linearity_error < 1.0e-12 #src
@test clamp_error < 1.0e-14 #src
@assert cutoff_exact #hide
@assert identity_error < 1.0e-12 #hide
@assert stoich_error < 1.0e-12 #hide
@assert sensitivity_error < 1.0e-12 #hide
@assert linearity_error < 1.0e-12 #hide
@assert clamp_error < 1.0e-14 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "identity_relative_error" => identity_error,
            "stoichiometry_relative_error" => stoich_error,
            "sensitivity_relative_error" => sensitivity_error,
            "linearity_relative_error" => linearity_error,
            "clamp_relative_error" => clamp_error,
        ),
        artifacts = ["arrhenius_kinetics.png"],
        notes = [
            "Verification-stage exact-algebra evidence for combustion: the finite-rate Arrhenius closure matches its closed form, stoichiometry, exponential temperature sensitivity, pre-exponential linearity, and low-T clamp to 1e-12.",
        ],
        summary = Dict(
            "E_a" => E_a,
            "stoich_ratio" => 4.0,
        ),
    )
end

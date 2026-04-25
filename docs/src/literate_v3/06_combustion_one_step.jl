# # Tutorial 06 — One-Step EDM Combustion (Species Transport)
#
# Demonstrates the v3 combustion stack on a small 2D mixing layer
# using Eddy Dissipation Model (EDM) with a one-step
# `fuel + oxidizer -> product` reaction.
#
# Runtime budget: ~5 s on a laptop (8×4 mesh, 3 SIMPLE iterations).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/06_combustion_one_step.jl
# ```
#
# What it demonstrates:
# - `CombustionProperties` + `EddyDissipationModel` construction
# - Passing `bcs_species` as a nested dict (per species, per patch)
# - `solve_simple_reacting` returning a `SpeciesState{N, T}`
#
# KNOWN ISSUE: EDM is one-step only with a Lewis-unity assumption.
# For multi-step chemistry use `MultiStepMechanism` +
# `CollocatedArrheniusReaction` or the FGM table path; see
# `src/combustion/arrhenius.jl` and `src/combustion/fgm.jl`.

using FiniteVolumeMethod
using LinearSolve
using StaticArrays
using Printf

include(joinpath(@__DIR__, "..", "..", "..", "test", "TestHelpers.jl"))

mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)

bcs = Dict{Symbol, AbstractBoundaryCondition}(
    :left => FixedVelocityBC((0.1, 0.0)),
    :right => FixedPressureBC(0.0),
    :bottom => NoSlipWallBC(),
    :top => NoSlipWallBC(),
)

algo = SIMPLE(; max_iterations = 3, tolerance = 1.0e-12)
prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

thermal_props = FluidThermalProperties{2}(; k = 0.6, Cp = 4000.0)
combustion_props = CombustionProperties()
edm = EddyDissipationModel()

# Inlet carries pure fuel from the left, oxidiser enters from the right.
bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
    :left => thermal_inlet_bc(350.0),
    :right => thermal_insulated_bc(),
    :bottom => thermal_inlet_bc(300.0),
    :top => thermal_inlet_bc(300.0),
)

bcs_species = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
    :fuel => Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(1.0),
        :right => ParabolicNeumann(0.0),
        :bottom => ParabolicNeumann(0.0),
        :top => ParabolicNeumann(0.0),
    ),
    :oxidizer => Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(0.0),
        :right => ParabolicDirichlet(0.233),
        :bottom => ParabolicNeumann(0.0),
        :top => ParabolicNeumann(0.0),
    ),
    :product => Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(0.0),
        :right => ParabolicNeumann(0.0),
        :bottom => ParabolicNeumann(0.0),
        :top => ParabolicNeumann(0.0),
    ),
)

result, thermal_state, species_state = solve_simple_reacting(
    prob, thermal_props, combustion_props, edm;
    bcs_T = bcs_T,
    bcs_species = bcs_species,
    Y_init = Dict(:fuel => 0.5, :oxidizer => 0.1),
)

println("=== One-step EDM combustion ===")
@printf "SIMPLE outer its  : %d\n" result.iterations
@printf "T min / max       : %.2f / %.2f K\n" minimum(thermal_state.T_field.internal) maximum(thermal_state.T_field.internal)
for (i, name) in enumerate([:fuel, :oxidizer, :product])
    Y = species_state.Y[i].internal
    @printf "Y(%s) min / max   : %.4f / %.4f\n" name minimum(Y) maximum(Y)
end

# Manifest feature  : phase8.edm_reaction (experimental)
# V&V tests         : test/combustion.jl, test/v_and_v_combustion_props.jl

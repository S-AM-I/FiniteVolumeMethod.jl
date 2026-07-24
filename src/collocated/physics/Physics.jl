# ============================================================
# Collocated.Physics — optional physics for the collocated solvers
# ============================================================
#
# RANS/LES/hybrid turbulence, thermal energy + conjugate heat transfer,
# P1/fvDOM/WSGGM radiation, and combustion/species transport. These
# consume the collocated operator/equation types and the incompressible
# solver entry points, so the module nests inside Collocated and loads
# after the Collocated core (but before the CommonSolve façade, which
# late-binds into the six solve_* entry points defined here).
module Physics

using ...Geometry
using ...Numerics
# _dispatch_solve is unexported as of Stage 4c; the turbulence/thermal/
# radiation/combustion solver files call it bare.
using ...Numerics: _dispatch_solve
using ...Parabolic
# physics/turbulence/k_epsilon.jl defines the converting constructor
# KEpsilon(::StandardKEpsilon) — import so it extends the
# Parabolic type's constructor instead of shadowing the type name.
import ...Parabolic: KEpsilon
# The momentum/pressure operators (assemble_momentum!, correct_*, etc.) are no
# longer imported here: Stage 5e moved those calls into the shared solver cores
# (`_simple_outer_step!`, `_piso_step!`, `_pimple_step!`), which live in the
# parent `Collocated` module. The physics loops now call the cores, so only the
# setup/residual/transport helpers below remain in use.
using ..Collocated: add_diag!, CollocatedEquation,
    CollocatedScalarField,
    CollocatedVectorField,
    FaceFluxField,
    IncompressibleProblem,
    IncompressibleState,
    MRFZone,
    NoSlipWallBC,
    PIMPLE,
    PISO,
    PorousZone,
    SIMPLE,
    SolveResult,
    WallFunctionBC,
    assemble_convection!,
    assemble_ddt_euler!,
    assemble_laplacian!,
    build_boundary_map,
    continuity_residual,
    gradient,
    reset!,
    to_linear_problem,
    _face_tag,
    _make_incompressible_workspace,
    _print_simple_residuals,
    _velocity_labels,
    collect_cyclic_pairs,
    update_boundary_pressure!,
    update_boundary_velocity!,
    _piso_step!,
    _pimple_step!,
    _simple_outer_step!
using LinearAlgebra: LinearAlgebra, dot, norm
using Printf: @sprintf
using StaticArrays: SVector

include("turbulence/k_epsilon.jl")
include("turbulence/interface.jl")
include("turbulence/strain_rate.jl")
include("turbulence/wall_distance.jl")
include("turbulence/k_epsilon_rans.jl")
include("turbulence/k_omega.jl")
include("turbulence/k_omega_sst.jl")
include("turbulence/spalart_allmaras.jl")
include("turbulence/wall_functions.jl")
include("turbulence/solvers.jl")
include("turbulence/les_types.jl")
include("turbulence/smagorinsky.jl")
include("turbulence/wale.jl")
include("turbulence/dynamic_smagorinsky.jl")
include("turbulence/ddes.jl")
include("turbulence/wmles.jl")
include("turbulence/sa_ddes.jl")
include("turbulence/iddes.jl")
include("thermal/types.jl")
include("thermal/energy_equation.jl")
include("thermal/enthalpy_equation.jl")
include("thermal/buoyancy.jl")
include("thermal/solid_conduction.jl")
include("thermal/conjugate.jl")
include("thermal/solvers.jl")
include("radiation/types.jl")
include("radiation/p1.jl")
include("radiation/solvers.jl")
include("radiation/fvdom.jl")
include("radiation/wsggm.jl")
include("combustion/types.jl")
include("combustion/variable_lewis.jl")
include("combustion/species_transport.jl")
include("combustion/edm.jl")
include("combustion/edc.jl")
include("combustion/arrhenius.jl")
include("combustion/multi_step.jl")
include("combustion/fgm.jl")
include("combustion/solvers.jl")

# Supported internals consumed by tests/docs as FiniteVolumeMethod.<name>
# (reached via the main module's qualified passthrough imports).
export
    T_from_h,
    h_from_T,
    patankar_interface_coupling,
    EquilibriumWMLES,
    IDDES,
    WSGGMModel,
    compute_band_emissivity,
    compute_band_weight,
    enthalpy_bcs_from_temperature,
    enthalpy_field_from_temperature,
    iddes_blended_length,
    scattering_phase_value,
    scattering_source_contribution,
    solve_wsggm_radiation,
    temperature_from_enthalpy!,
    turbulent_viscosity_sa!,
    wmles_wall_nut,
    wmles_wall_shear,
    wsggm_effective_absorption

export
    AbstractHybridModel, AbstractLESModel, AbstractRANSModel, AbstractRadiationModel,
    CollocatedArrheniusReaction, CombustionProperties, ConjugateHeatTransferProblem, DDES,
    DynamicSmagorinsky, EddyDissipationConcept, EddyDissipationModel, FGMTable,
    FluidThermalProperties, FvDOMModel, KOmega, KOmegaSSTModel, KappaOmegaSST,
    LESTurbulenceState, MultiStepMechanism, P1Model, RANSTurbulenceState, RadiationState,
    STEFAN_BOLTZMANN, Smagorinsky, SolidThermalProperties, SpalartAllmaras, SpeciesState,
    StandardKEpsilon, ThermalState, TurbulentWallBC, VariableLewis, WALE,
    apply_wall_functions!, assemble_energy!, assemble_p1!, assemble_solid_conduction!,
    assemble_species!, build_fgm_table_from_callback, compute_alpha_eff,
    compute_arrhenius_reaction_rates, compute_buoyancy_source, compute_edc_reaction_rates,
    compute_edm_reaction_rates, compute_filter_width, compute_fred_reaction_rates,
    compute_friction_velocity, compute_heat_release, compute_interface_heat_flux,
    compute_multi_step_rates, compute_multi_step_rates!, compute_nu_eff, compute_nut_wall,
    compute_production, compute_radiation_source, compute_strain_rate,
    compute_strain_rate_magnitude, compute_turbulent_viscosity, compute_wall_distance,
    epsilon_wall_value, equilibrium_epsilon_wall, equilibrium_k_wall,
    equilibrium_omega_wall, has_buoyancy, k_wall_value, lewis_number, lookup_fgm,
    lookup_fgm!, marshak_wall_bc, n_turbulence_fields, one_step_arrhenius_mechanism,
    radiation_inlet_bc, solve_conjugate_ht, solve_fvdom_radiation,
    solve_incompressible_thermal, solve_incompressible_turbulent, solve_p1_radiation,
    solve_simple_reacting, solve_simple_thermal, solve_simple_thermal_radiation,
    solve_simple_turbulent, solve_solid_conduction, solve_species!, solve_turbulence!,
    spalding_u_tau, species_diffusivity, thermal_convective_bc, thermal_heated_wall_bc,
    thermal_inlet_bc, thermal_insulated_bc, turbulence_field_names, turbulence_inlet_bc,
    turbulence_wall_bc, turbulent_viscosity!, update_k_eff!

end # module Physics

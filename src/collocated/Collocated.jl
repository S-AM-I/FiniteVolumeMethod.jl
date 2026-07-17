# ============================================================
# Collocated — the cell-centered collocated (OpenFOAM-style) family
# ============================================================
#
# Owns the Phase-0 collocated operators, the incompressible
# SIMPLE/PISO/PIMPLE solvers, multiphase VOF + Eulerian two-fluid,
# Lagrangian DPM, dynamic mesh/ALE, function objects, collocated AMR,
# post-processing, and the upstream zone models (MRF, porous,
# cavitation) whose types appear in incompressible method signatures.
# Optional physics (turbulence/thermal/radiation/combustion) nests in
# Collocated.Physics. The IncompressibleProblem remake/SciMLStructures
# methods stay in the flat parent (core/sciml_structures.jl, remake.jl)
# until the sciml/ relocation step.
module Collocated

using ..Geometry
# collocated/types.jl extends these Geometry generics on the collocated
# field types — import so the methods land on the Geometry functions.
import ..Geometry: ncells, nfaces
using ..Numerics
# incompressible/simple.jl defines the SOLE method of the Numerics
# _solve_linear extension point (called by Numerics._dispatch_solve);
# a bare `using` would shadow it and break every field sub-solve.
import ..Numerics: _solve_linear
# _dispatch_solve is unexported as of Stage 4c; the incompressible,
# dynamic-mesh, and multiphase solver files call it bare.
using ..Numerics: _dispatch_solve
using ..Parabolic
# lagrangian/injection.jl adds DPM injection methods to the particle
# verb owned by Parabolic (parabolic/particles.jl) — one shared generic.
import ..Parabolic: inject_particles!
using CommonSolve: CommonSolve, solve
using SciMLBase: SciMLBase, LinearProblem, remake
using LinearAlgebra: LinearAlgebra, inertia, normalize
using Printf: @sprintf
using SparseArrays: sparse
using StaticArrays: StaticArrays, SVector, SMatrix

# Upstream zone models: their types appear in incompressible method
# signatures (Vector{PorousZone{T}}, Vector{MRFZone{T}},
# CavitationProperties{T}), so they must precede the solver core.
include("mrf/types.jl")
include("mrf/momentum_source.jl")
include("mrf/multi_zone.jl")
include("porous/types.jl")
include("porous/darcy_forchheimer.jl")
include("cavitation/types.jl")
include("cavitation/kunz.jl")
include("cavitation/schnerr_sauer.jl")
include("cavitation/merkle.jl")
include("cavitation/solvers.jl")

# Phase-0 collocated operators.
include("types.jl")
include("interpolation.jl")
include("gradient.jl")
include("laplacian.jl")
include("divergence.jl")
include("ddt.jl")
include("cyclic.jl")

include("function_objects/types.jl")
include("function_objects/expression_bc.jl")

include("amr/adapt.jl")
include("amr/error_indicators.jl")
include("amr/refinement.jl")
include("amr/coarsening.jl")

include("incompressible/types.jl")
include("incompressible/boundary_conditions.jl")
include("incompressible/momentum.jl")
include("incompressible/pressure.jl")
include("incompressible/correction.jl")
include("incompressible/residuals.jl")
include("incompressible/simple.jl")
include("incompressible/piso.jl")
include("incompressible/pimple.jl")

include("multiphase/types.jl")
include("multiphase/mixture.jl")
include("multiphase/boundedness.jl")
include("multiphase/alpha_transport.jl")
include("multiphase/surface_tension.jl")
include("multiphase/iso_advector.jl")
include("multiphase/solvers.jl")
include("multiphase/drag_closures.jl")
include("multiphase/two_fluid.jl")
include("multiphase/mass_transfer.jl")
include("multiphase/two_fluid_solver.jl")

include("lagrangian/drag_models.jl")
include("lagrangian/heat_transfer.jl")
include("lagrangian/two_way_coupling.jl")
include("lagrangian/particle_solver.jl")
include("lagrangian/spray.jl")
include("lagrangian/collisions.jl")
include("lagrangian/agglomeration.jl")
include("lagrangian/primary_breakup.jl")
include("lagrangian/injection.jl")

include("dynamic_mesh/types.jl")
include("dynamic_mesh/solid_body.jl")
include("dynamic_mesh/laplacian_motion.jl")
include("dynamic_mesh/mesh_update.jl")
include("dynamic_mesh/ale.jl")
include("dynamic_mesh/six_dof.jl")
include("dynamic_mesh/topo_changer.jl")
include("dynamic_mesh/overset.jl")
include("dynamic_mesh/ami.jl")
# Needs MeshMotionState from dynamic_mesh/types.jl.
include("lagrangian/primary_breakup_fsi.jl")

include("postprocessing/field_operations.jl")
include("postprocessing/wall_quantities.jl")
include("postprocessing/forces.jl")
include("postprocessing/sampling.jl")
include("postprocessing/field_statistics.jl")

include("physics/Physics.jl")
using .Physics
# The CommonSolve façade late-binds into six Physics solve_* entry
# points, so it must load after `using .Physics`.
include("incompressible/solution.jl")
include("incompressible/sciml_interface.jl")

export
    ConvectionScheme, CONV_UPWIND, CONV_LINEAR, CONV_BLENDED,
    TimeScheme, TIME_EULER, TIME_BDF2, TIME_CRANK_NICOLSON,
    NonOrthoCorrectionMode, NON_ORTHO_NONE, NON_ORTHO_MINIMUM,
    NON_ORTHO_ORTHOGONAL, NON_ORTHO_OVER_RELAXED,
    add_diag!, add_offdiag_NP!, add_offdiag_PN!, add_face_coeffs_PN!,
    add_block_diag!, add_block_offdiag_NP!, add_block_offdiag_PN!,
    angular_velocity_vector, evaluate_expression_bc, mrf_frame_flux,
    mrf_frame_velocity,
    AbstractBreakupModel, AbstractCavitationModel, AbstractCavitationVaporModel,
    AbstractCollisionModel, AbstractCollocatedField, AbstractDragClosure, AbstractDragModel,
    AbstractFVMSolution, AbstractFunctionObject, AbstractMRFZone, AbstractMotionSolver,
    AbstractPVCoupling, AbstractParticleHeatTransfer, AbstractPorousModel,
    AtmosphericBLProfileBC, BlockCollocatedEquation, BlockSparsityPattern, BoundaryPatch,
    CavitationProperties, CoarseningPlan, CodedFixedValueBC, CollocatedEquation,
    CollocatedScalarField, CollocatedVectorField, ConvectiveOutletBC, CustomBC, CyclicBC,
    DarcyForchheimerPorous, DarcyPorous, ExpressionBC, FaceFluxField, FieldStatistics,
    FixedPressureBC, FixedVelocityBC, FlowRateInletBC, ForceProbe, GibilaroDrag,
    IncompressibleProblem, IncompressibleSolution, IncompressibleState, InletOutletBC,
    IshiiZuberDrag, KHRTBreakup, KunzCavitation, KunzModel, LaplacianMotion, MRFZone,
    MerkleCavitation, MerkleModel, MeshMotionState, MultiMRF, NoSlipWallBC,
    ORourkeCollision, OrthotropicPorous, PIMPLE, PISO, PointProbe, PorousJumpBC, PorousZone,
    PressureInletVelocityBC, RanzMarshall, RefinementMarker, RefinementPlan,
    RotationalMRFZone, SIMPLE, SchillerNaumann, SchnerrSauerCavitation, SchnerrSauerModel,
    SlipWallBC, SolidBodyMotion, SolveResult, SparsityPattern, SpatialVelocityBC,
    StokesDrag, StringExpressionBC, SymmetryBC, TABBreakup, TimeDependentVelocityBC,
    TotalPressureBC, TwoFluidProperties, TwoFluidSolver, TwoFluidState, TwoPhaseProperties,
    UniformFixedValueBC, VOFState, WallFunctionBC, WaveTransmissiveBC, ZeroGradientBC,
    advance_particles!, ale_corrected_flux, apply_breakup!, apply_coarsening!,
    apply_collisions!, apply_cyclic_bc!, apply_refinement!, assemble_alpha!,
    assemble_convection!, assemble_ddt!, assemble_ddt_bdf2!, assemble_ddt_crank_nicolson!,
    assemble_ddt_euler!, assemble_laplacian, assemble_laplacian!, assemble_momentum!,
    assemble_pressure!, breakup_diameter, bubble_reynolds, build_block_collocated_sparsity,
    build_boundary_map, build_collocated_sparsity, build_multi_mrf_from_zones,
    cavitation_source, clip_alpha!, collocated_to_odefunction, compute_compression_flux,
    compute_courant_number, compute_curvature, compute_displacement!,
    compute_distance_diffusivity, compute_drag_force, compute_energy_source,
    compute_enstrophy, compute_face_flux!, compute_forces, compute_max_courant,
    compute_mesh_flux!, compute_momentum_source, compute_nusselt_number,
    compute_particle_heat_transfer, compute_q_criterion, compute_surface_tension_force,
    compute_vapor_source, compute_vorticity, compute_wall_heat_flux,
    compute_wall_shear_stress, compute_y_plus, continuity_residual,
    continuity_residual_interior, correct_fluxes!, correct_velocity!,
    darcy_forchheimer_source, density_ratio, divergence, divergence!, drag_coefficient,
    drag_force_density, enforce_volume_fraction_sum!, evaluate, extract_boundary_patches,
    extract_momentum_operators!, face_value, field_average, field_min_max,
    find_nearest_cell, flux_correction_factor, force_coefficients, gradient, gradient!,
    green_gauss_gradient, has_surface_tension, interphase_drag, interpolate_blended,
    interpolate_linear, interpolate_upwind, is_fvm_solution, mark_cells_by_gradient,
    mark_for_refinement, match_cyclic_faces, momentum_residual, mrf_make_absolute!,
    mrf_make_relative!, mrf_momentum_source, mrf_momentum_source_2d_planar,
    mules_limit_flux!, n_boundary_faces, nblocks, porous_momentum_source, reset!,
    residual_error_indicator, rhie_chow_correction!, sample_field_at_point, sample_line,
    set_particle_properties!, should_breakup, solve_ale, solve_incompressible, solve_simple,
    solve_vof, stokes_limit_drag, to_linear_problem, turbulence_intensity, update_mesh!,
    update_mixture_properties!, verify_gcl, weber_number, zz_error_indicator
# Re-exported Collocated.Physics public API (Stage 4c: the underscore
# internals are no longer exported anywhere; they stay reachable in this
# module via the selective `using .Physics:` block below, which the main
# module's qualified passthrough imports resolve against).
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
    wsggm_effective_absorption,
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

end # module Collocated

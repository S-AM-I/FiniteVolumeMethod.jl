# ============================================================
# Hyperbolic — the cell-centered hyperbolic solver family
# ============================================================
#
# Owns the conservation laws (Euler, MHD variants, Navier-Stokes, shallow
# water, SR/GR (M)HD, two-fluid, reactive Euler), Riemann solvers,
# reconstruction (MUSCL/PPM/WENO), hyperbolic boundary conditions,
# constrained transport, spacetime metrics, block-structured AMR, and the
# semidiscrete SciML bridge (core/: caches, state mapping, CFL callback,
# ODE/SplitODE construction, the sciml_problem contract, and solution
# accessors). The cross-family glue (core/symbolic_indexing.jl,
# core/sciml_structures.jl, remake.jl) and solve.jl stay in the flat parent
# until the sciml/ relocation step; this module owns the
# fvm_symbolic_index/_amr_symbolic_index/variable_names generics they extend.
module Hyperbolic

using ..Geometry
using ..Numerics: AbstractEOS, IdealGasEOS, StiffenedGasEOS, pressure,
    sound_speed, internal_energy, AbstractLimiter, MinmodLimiter,
    SuperbeeLimiter, VanLeerLimiter, VenkatakrishnanLimiter, KorenLimiter,
    OspreLimiter, minmod, superbee, van_leer, venkatakrishnan, koren, ospre,
    compute_slope_ratio, AbstractBackend, CPUBackend, to_backend,
    _cpu_backend_only, _unsupported_backend
# euler_3d.jl defines the 5-argument 3D total_energy methods — import so they
# extend the Numerics generic rather than shadow it (Stage-3 recipe).
import ..Numerics: total_energy
# Supported-but-unexported internal drivers behind the SciML solve surface
# (threaded/GPU/parity paths; ext/FVMCUDAExt extends _solve_hyperbolic).
using Compat: @compat
@compat public solve_hyperbolic, solve_hyperbolic_imex
# core/sciml_contract.jl and core/results.jl dispatch on the parabolic
# problem types (sciml_problem/solution_accessor are cross-family verbs).
using ..Parabolic: FVMProblem, FVMSystem, SteadyFVMProblem, AbstractFVMTemplate

using CommonSolve: CommonSolve
using SciMLBase: SciMLBase, ODEProblem, ODEFunction, SplitODEProblem,
    SteadyStateProblem, DiscreteCallback, CallbackSet, set_proposed_dt!
using StaticArrays: StaticArrays, SVector
using PreallocationTools: PreallocationTools, DiffCache
using DelaunayTriangulation: DelaunayTriangulation, get_point
using Base.Threads

# Owned cross-family generics (GAP 1): core/ode_construction.jl attaches
# symbolic-index metadata through these; the methods for both solver families
# live in the flat parent (core/symbolic_indexing.jl), which extends them via
# the layer-file `import .Hyperbolic:` guard.
function fvm_symbolic_index end
function _amr_symbolic_index end
function _mhd_ct_2d_symbolic_index end
function _mhd_ct_3d_symbolic_index end

include("conservation_laws.jl")
include("euler.jl")
include("riemann_solvers.jl")
include("reconstruction.jl")
include("boundary_conditions_hyp.jl")
include("hllc_solver.jl")
include("mhd.jl")
include("hlld_solver.jl")
include("hyperbolic_problem.jl")
include("hyperbolic_solve.jl")
include("hyperbolic_problem_2d.jl")
include("boundary_conditions_2d.jl")
include("hyperbolic_solve_2d.jl")

include("constrained_transport/ct_data.jl")
include("constrained_transport/emf.jl")
include("constrained_transport/ct_update.jl")
include("constrained_transport/divb.jl")

include("mhd_solve_2d.jl")

include("navier_stokes.jl")
include("viscous_flux.jl")
include("noslip_bc.jl")
include("navier_stokes_solve.jl")
include("navier_stokes_solve_2d.jl")

include("resistive_mhd.jl")
include("hall_mhd.jl")
include("shallow_water.jl")
include("srhydro.jl")
include("two_fluid.jl")

include("con2prim.jl")
include("srmhd.jl")
include("srmhd_solve.jl")
include("srmhd_solve_2d.jl")

include("metric/abstract_metric.jl")
include("metric/minkowski.jl")
include("metric/schwarzschild.jl")
include("metric/kerr.jl")
include("metric/metric_data.jl")

include("grmhd.jl")
include("grmhd_con2prim.jl")
include("grmhd_solve_2d.jl")

include("euler_3d.jl")
include("mhd_3d.jl")

include("hyperbolic_problem_3d.jl")
include("boundary_conditions_3d.jl")
include("hyperbolic_solve_3d.jl")

include("constrained_transport/ct_data_3d.jl")
include("constrained_transport/emf_3d.jl")
include("constrained_transport/ct_update_3d.jl")
include("constrained_transport/divb_3d.jl")

include("mhd_solve_3d.jl")

include("amr/amr_grid.jl")
include("amr/refinement.jl")
include("amr/prolongation.jl")
include("amr/restriction.jl")
include("amr/flux_correction.jl")
include("amr/amr_solve.jl")

include("multirate.jl")

include("ppm.jl")

include("weno3.jl")
include("weno.jl")
include("characteristic_projection.jl")
include("stiff_sources.jl")
include("imex.jl")
include("imex_solve.jl")

include("reactive_euler.jl")
include("chemistry.jl")

include("positivity_limiter.jl")
include("threading.jl")

include("unstructured_problem.jl")
include("unstructured_solve.jl")

include("variable_names.jl")

include("core/cache.jl")
include("core/state_mapping.jl")
include("core/cfl_callback.jl")
include("core/callback_merge.jl")
include("core/ode_construction.jl")
include("core/split_construction.jl")
include("core/sciml_contract.jl")
include("core/results.jl")

# Multi-physics operator splitting (Stage 3f: moved in from flat
# coupling/ — it consumes only Hyperbolic + Geometry names).
include("coupling/abstract_coupling.jl")
include("coupling/operators.jl")
include("coupling/data_transfer.jl")
include("coupling/coupled_solve.jl")

# Public API (re-exported from the main module).
export
    # Conservation laws
    AbstractConservationLaw, EulerEquations, NavierStokesEquations,
    IdealMHDEquations, fast_magnetosonic_speed, slow_magnetosonic_speed,
    nvariables, physical_flux, max_wave_speed, wave_speeds,
    conserved_to_primitive, primitive_to_conserved,
    # Riemann solvers
    AbstractRiemannSolver, LaxFriedrichsSolver, HLLSolver, HLLCSolver,
    HLLDSolver, solve_riemann,
    # Reconstruction
    CellCenteredMUSCL, NoReconstruction, reconstruct_interface,
    PPMReconstruction, WENO3, WENO5, nghost, reconstruct_interface_weno5,
    CharacteristicWENO, left_eigenvectors, right_eigenvectors,
    # Hyperbolic boundary conditions
    AbstractHyperbolicBC, TransmissiveBC, ReflectiveBC, InflowBC,
    PeriodicHyperbolicBC, DirichletHyperbolicBC, NoSlipBC,
    # Problems and solver interface
    HyperbolicProblem, HyperbolicProblem2D, HyperbolicProblem3D,
    initialize_2d, compute_dt, compute_dt_2d, compute_dt_3d,
    hyperbolic_rhs!, hyperbolic_rhs_2d!, hyperbolic_rhs_3d!, to_primitive,
    # Navier-Stokes
    thermal_conductivity, viscous_flux_1d, viscous_flux_x_2d,
    viscous_flux_y_2d,
    # Resistive MHD
    ResistiveMHDEquations, resistive_flux_x, resistive_flux_y, ohmic_heating,
    resistive_dt,
    # Hall MHD
    HallMHDEquations, whistler_speed, hall_flux_x, hall_flux_y, hall_dt,
    # Shallow water
    ShallowWaterEquations, BottomTopography, topography_source_1d,
    # SR hydro
    SRHydroEquations, srhydro_con2prim,
    # Two-fluid plasma
    TwoFluidEquations, ion_primitive, electron_primitive, ion_conserved,
    electron_conserved, lorentz_source_1d, lorentz_source_2d,
    # Reactive Euler and chemistry
    ReactiveEulerEquations, euler_primitive, euler_conserved,
    species_mass_fractions, species_partial_densities, temperature,
    ArrheniusReaction, ReactionMechanism, ChemistrySource,
    # Stiff sources and IMEX
    AbstractStiffSource, ResistiveSource, CoolingSource, NullSource,
    evaluate_stiff_source, stiff_source_jacobian, AbstractIMEXScheme,
    IMEX_SSP3_433, IMEX_ARS222, IMEX_Midpoint, imex_tableau, imex_nstages,
    # Positivity limiter
    PositivityLimiter, apply_positivity_limiter!, apply_positivity_limiter_2d!,
    limit_reconstructed_states,
    # Constrained transport (2D)
    CTData2D, initialize_ct!, initialize_ct_from_potential!, face_to_cell_B!,
    copy_ct, copyto_ct!, compute_emf_2d!, ct_update!, compute_divB, max_divB,
    l2_divB,
    # Constrained transport (3D)
    CTData3D, initialize_ct_3d!, initialize_ct_3d_from_potential!,
    face_to_cell_B_3d!, ct_update_3d!, ct_weighted_update_3d!, compute_divB_3d,
    max_divB_3d, l2_divB_3d,
    # AMR
    AMRBlock, AMRGrid, AbstractRefinementCriterion, GradientRefinement,
    CurrentSheetRefinement, active_blocks, blocks_at_level, max_active_level,
    block_cell_center, needs_refinement, needs_coarsening, refine_block!,
    coarsen_block!, regrid!, prolongate!,
    prolongate_B_divergence_preserving_2d!, restrict!, restrict_B_face_2d!,
    restrict_B_face_3d!, FluxRegister, reset_flux_register!,
    accumulate_fine_flux!, store_coarse_flux!, apply_flux_correction_2d!,
    apply_flux_correction_3d!, AMRProblem, solve_amr, compute_dt_amr,
    advance_level!,
    # Multi-rate subcycling
    SubcyclingScheme, solve_amr_subcycled, advance_level_subcycled!,
    compute_dt_subcycled, total_substeps,
    # SRMHD
    SRMHDEquations, lorentz_factor, srmhd_b_quantities, srmhd_con2prim,
    Con2PrimResult,
    # Spacetime metrics
    AbstractMetric, MinkowskiMetric, SchwarzschildMetric, KerrMetric, lapse,
    shift, spatial_metric, sqrt_gamma, inv_spatial_metric, MetricData2D,
    precompute_metric, precompute_metric_at_faces,
    # GRMHD
    GRMHDEquations, grmhd_con2prim, grmhd_con2prim_cached,
    grmhd_primitive_to_conserved_densitized, grmhd_prim2con_densitized_cached,
    grmhd_max_wave_speed_coord, grmhd_source_terms,
    # Unstructured hyperbolic solver
    UnstructuredHyperbolicProblem, rotate_to_normal, rotate_flux_from_normal,
    boundary_ghost_state, get_bc,
    # Variable naming for laws (owned here; dashboard/symbolic indexing consume)
    variable_names,
    # Semidiscrete caches
    AbstractSemidiscreteCache, HyperbolicCache1D, HyperbolicCache2D,
    HyperbolicCache3D, UnstructuredCache, MHDCTCache2D, MHDCTCache3D,
    GRMHDCTCache2D, AMRCache, build_cache, build_mhd_ct_cache,
    build_grmhd_ct_cache, build_amr_cache,
    # State mapping
    unfold_to_padded!, fold_from_padded!, unfold_mhd_augmented!,
    fold_mhd_augmented!, initial_state_flat, initial_mhd_augmented_state,
    flatten_amr_state, unfold_amr!, fold_amr!,
    # CFL callback
    cfl_stepsize_callback, compute_initial_dt,
    # MHD stage limiter
    mhd_stage_limiter,
    # Solution accessors
    FVMSolutionAccessor, HyperbolicSolutionAccessor, MHDSolutionAccessor,
    MHD3DSolutionAccessor, AMRODESolutionAccessor, AMRSolution, get_conserved,
    get_primitive, get_coordinates, get_ct_state,
    # Canonical SciML contract
    sciml_problem, fvm_symbolic_index, solution_accessor, solution_snapshot,
    solution_coordinates, solution_state_layout, solution_variables,
    # Multi-physics coupling (operator splitting)
    AbstractOperator, AbstractSplittingScheme, LieTrotterSplitting,
    StrangSplitting, CoupledProblem, HyperbolicOperator, SourceOperator,
    advance!, compute_operator_dt, solve_coupled, cell_to_vertex,
    vertex_to_cell

end # module Hyperbolic

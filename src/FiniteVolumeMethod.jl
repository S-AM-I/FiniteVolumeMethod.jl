module FiniteVolumeMethod

using ChunkSplitters: ChunkSplitters, chunks
using CommonSolve: CommonSolve, solve
using DelaunayTriangulation: DelaunayTriangulation, Triangulation,
    add_ghost_triangles!,
    convert_boundary_points_to_indices,
    delete_ghost_triangles!, each_solid_triangle,
    each_solid_vertex, get_adjacent, get_area,
    get_boundary_edge_map, get_boundary_nodes,
    get_ghost_vertex_map, get_neighbours, get_point,
    getxy, lock_convex_hull!, num_boundary_edges,
    num_solid_triangles, refine!, statistics,
    triangle_vertices, triangulate,
    triangulate_rectangle, unlock_convex_hull!
using LinearAlgebra: LinearAlgebra, I, dot, norm
using PreallocationTools: PreallocationTools, DiffCache, get_tmp
using SciMLBase: SciMLBase, CallbackSet, DiscreteCallback, LinearProblem,
    MatrixOperator, ODEFunction, ODEProblem, SplitODEProblem,
    SteadyStateProblem, remake, set_proposed_dt!
using SparseArrays: SparseArrays, sparse
using StaticArrays: StaticArrays, SVector
using Base.Threads

# Module loading order: types → solvers → SciML bridge → I/O
include("layers/domain_problem_definitions.jl")
include("layers/discretization_assembly_kernels.jl")
include("layers/sciml_adapters_and_accessors.jl")
include("layers/extensions_tooling_output.jl")

# --- Parabolic Core Types ---
export
    # Geometry and Mesh
    AbstractParabolicMesh,
    AbstractNode,
    AbstractCell,
    AbstractFace,
    CellType,
    CT_Tetrahedron,
    CT_Hexahedron,
    CT_Prism,
    CT_Pyramid,
    CT_Polyhedron,
    # Boundary and Initial Conditions
    AbstractFVMBoundaryCondition,
    AbstractBoundaryCondition,
    AbstractInitialCondition,
    UnsupportedBCError,
    ParabolicDirichlet,
    ParabolicNeumann,
    ParabolicRobin,
    # Variables and Fields
    AbstractVariable,
    VariableRole,
    STATEVAR,
    Variable,
    CellField,
    SimulationState,
    validate_state,
    update_field,
    # Discretization
    AbstractDiscretization,
    AbstractSemidiscretization,
    AbstractFluxCalculator,
    AbstractReconstruction,
    # Structured Mesh Types (Parabolic)
    Node1D,
    Cell1D,
    Face1D,
    Mesh1D,
    Node2D,
    Cell2D,
    Face2D,
    Mesh2D,
    Node3D,
    Cell3D,
    Face3D,
    Mesh3D,
    # Structured Mesh Generation
    generate_mesh_1d,
    generate_mesh_1d_nonuniform,
    generate_mesh_2d,
    generate_mesh_2d_nonuniform,
    generate_mesh_3d,
    generate_mesh_3d_nonuniform,
    # Curvilinear Mesh
    CurvilinearMesh2D,
    CurvilinearMesh3D,
    get_cell_center,
    get_face_geo,
    # Unstructured Mesh (Parabolic)
    UnstructuredFace2D,
    UnstructuredCell2D,
    UnstructuredMesh2D,
    UnstructuredFace3D,
    UnstructuredCell3D,
    UnstructuredMesh3D,
    convert_to_unstructured,
    check_mesh_quality,
    refine_uniform,
    # FVM Mesh Wrappers
    AbstractFiniteVolumeMesh,
    AbstractFVMMesh,
    dim_of,
    n_cells,
    n_faces,
    StructuredFVMMesh,
    CurvilinearFVMMesh,
    UnstructuredFVMMesh,
    validate_mesh,
    build_structured_mesh3d,
    build_axisymmetric_rz_mesh,
    structured_boundary_tags,
    build_curvilinear_mesh,
    polygon_area,
    parse_ply,
    parse_vtk,
    tag_unstructured_faces_by_bounds,
    build_unstructured_from_polygons,
    load_unstructured_mesh,
    # Mesh I/O
    read_gmsh,
    volume_tet,
    volume_hex,
    build_faces_from_cells,
    get_cell_faces,
    write_vtk_unstructured,
    # Mesh I/O (Phase 4)
    read_openfoam_polymesh,
    write_openfoam_field,
    convert_to_fvm_mesh,
    volume_prism,
    volume_pyramid,
    MeshQualityReport,
    check_mesh_quality,
    print_mesh_quality,
    # Mesh Partitioning
    PartitionedMesh,
    partition_mesh_rcb,
    recursive_bisection,
    extract_submesh

# --- Parabolic Solver (from Simu.jl SimuFVM migration) ---
export
    # Equation Models
    AbstractEquationModel,
    AbstractDiffusion,
    AbstractAdvection,
    AbstractAdvectionDiffusion,
    Diffusion1D,
    Diffusion2D,
    Diffusion3D,
    VariableDiffusion1D,
    VariableDiffusion2D,
    VariableDiffusion3D,
    AnisotropicDiffusion1D,
    AnisotropicDiffusion2D,
    AnisotropicDiffusion3D,
    CylindricalDiffusion1D,
    CylindricalDiffusion2D,
    SphericalDiffusion1D,
    SphericalAdvection1D,
    SphericalAdvectionDiffusion1D,
    CylindricalAdvection1D,
    CylindricalAdvection2D,
    Advection1D,
    Advection2D,
    Advection3D,
    VariableAdvection1D,
    VariableAdvection2D,
    VariableAdvection3D,
    AdvectionDiffusion1D,
    AdvectionDiffusion2D,
    AdvectionDiffusion3D,
    VariableAdvectionDiffusion1D,
    VariableAdvectionDiffusion2D,
    VariableAdvectionDiffusion3D,
    CylindricalAdvectionDiffusion1D,
    CylindricalAdvectionDiffusion2D,
    # Source Terms
    AbstractSourceTerm,
    ConstantSource,
    SpatialSource,
    FunctionSource,
    LinearizedSource,
    evaluate_source,
    # Turbulence (Parabolic)
    AbstractTurbulenceModel,
    ParabolicKEpsilon,
    update_turbulent_viscosity!,
    compute_production_k,
    assemble_k_source,
    assemble_epsilon_source,
    parabolic_compute_friction_velocity,
    update_wall_bcs!,
    ParabolicTurbulentWall,
    # Assembly
    assemble_system,
    assemble_mass_matrix,
    assemble_deferred_correction,
    # Coupled System Assembly
    AbstractCoupling,
    LinearCoupling,
    assemble_coupled_system,
    build_linear_coupling_block,
    # Boundary Conditions (Parabolic Solver)
    InterfaceBC,
    ParabolicPeriodicBC,
    ParabolicNonlinearDirichlet,
    ParabolicNonlinearNeumann,
    ParabolicCoupledBC,
    OutflowBC,
    # Gradients (Parabolic)
    reconstruct_gradient_green_gauss_2d,
    reconstruct_gradient_green_gauss_3d,
    reconstruct_gradient_least_squares_1d,
    reconstruct_gradient_least_squares_2d,
    # Schemes (Parabolic)
    muscl_reconstruction_1d,
    quick_reconstruction_1d,
    second_order_diffusion_flux_1d,
    muscl_advection_flux_1d,
    quick_advection_flux_1d,
    weno5_reconstruction_1d,
    weno5_advection_flux_1d,
    weno5_reconstruction_right_biased,
    muscl_reconstruction_2d,
    quick_reconstruction_2d,
    muscl_advection_flux_2d,
    quick_advection_flux_2d,
    # Compressible Fluxes (Parabolic)
    ideal_gas_pressure,
    parabolic_sound_speed,
    hllc_flux_1d,
    # Particles (Parabolic)
    AbstractParticle,
    LagrangianParticle,
    ParticleTracker,
    inject_particles!,
    find_cell_index,
    is_point_in_cell,
    advect_particles!,
    # FSI (Parabolic)
    AbstractStructuralModel,
    SpringMassSystem,
    update_structure!,
    deform_mesh!,
    update_mesh_geometry!,
    # Kernels
    compute_fluxes_cpu!,
    # Utils (Parabolic)
    add_entry!,
    apply_source_term!,
    get_diffusion_coefficient_at_face_2d,
    TimeDependentDirichlet,
    TimeDependentNeumann,
    TimeDependentRobin

# --- Collocated Cell-Centered FVM Operators (Phase 0) ---
export
    # Fields
    AbstractCollocatedField,
    CollocatedScalarField,
    CollocatedVectorField,
    FaceFluxField,
    CollocatedEquation,
    BoundaryPatch,
    extract_boundary_patches,
    ncells,
    n_boundary_faces,
    nfaces,
    # Mesh helpers
    is_internal_face,
    face_normal_area,
    face_weight,
    find_nearest_cell,
    # Interpolation
    interpolate_linear,
    interpolate_upwind,
    interpolate_blended,
    build_boundary_map,
    face_value,
    compute_face_flux!,
    rhie_chow_correction!,
    # Gradient
    gradient!,
    gradient,
    green_gauss_gradient,
    # Laplacian
    assemble_laplacian!,
    assemble_laplacian,
    NonOrthoCorrectionMode,
    NON_ORTHO_MINIMUM,
    NON_ORTHO_ORTHOGONAL,
    NON_ORTHO_OVER_RELAXED,
    # Divergence
    divergence!,
    divergence,
    ConvectionScheme,
    CONV_UPWIND,
    CONV_LINEAR,
    CONV_BLENDED,
    assemble_convection!,
    # Temporal
    TimeScheme,
    TIME_EULER,
    TIME_BDF2,
    TIME_CRANK_NICOLSON,
    assemble_ddt!,
    assemble_ddt_euler!,
    assemble_ddt_bdf2!,
    assemble_ddt_crank_nicolson!,
    # Cyclic BC
    match_cyclic_faces,
    apply_cyclic_bc!,
    # SciML bridge
    to_linear_problem,
    collocated_to_odefunction,
    reset!,
    # Tunable schema (Stage 1e)
    register_tunable!,
    tunable_schema,
    tunable_names,
    tunable_namedtuple,
    # Unified solution wrapper (Stage 1f)
    AbstractFVMSolution,
    is_fvm_solution,
    # Linear operator abstraction (Stage 1h)
    AbstractLinearOperator,
    SparseMatrixLinearOperator,
    MatrixFreeError,
    underlying_matrix,
    as_linear_operator,
    # Sparsity pattern + fast-path assembly helpers (Stage 1a)
    SparsityPattern,
    build_collocated_sparsity,
    add_diag!,
    add_offdiag_PN!,
    add_offdiag_NP!,
    add_face_coeffs_PN!,
    # Block-coupled equation + assembly helpers (Stage 1c)
    BlockCollocatedEquation,
    BlockSparsityPattern,
    build_block_collocated_sparsity,
    add_block_diag!,
    add_block_offdiag_PN!,
    add_block_offdiag_NP!,
    nblocks

# --- Incompressible Navier-Stokes (Phase 1) ---
export
    # Algorithm types
    AbstractPVCoupling,
    SIMPLE,
    PISO,
    PIMPLE,
    # Problem and state
    IncompressibleProblem,
    IncompressibleState,
    IncompressibleSolution,
    SolveResult,
    # Boundary conditions
    FixedVelocityBC,
    SpatialVelocityBC,
    FixedPressureBC,
    NoSlipWallBC,
    SlipWallBC,
    InletOutletBC,
    ZeroGradientBC,
    TotalPressureBC,
    SymmetryBC,
    FlowRateInletBC,
    TimeDependentVelocityBC,
    WallFunctionBC,
    ConvectiveOutletBC,
    PressureInletVelocityBC,
    CyclicBC,
    CustomBC,
    UniformFixedValueBC,
    CodedFixedValueBC,
    WaveTransmissiveBC,
    AtmosphericBLProfileBC,
    PorousJumpBC,
    # Solvers
    solve_simple,
    solve_incompressible,
    # Assembly (advanced)
    assemble_momentum!,
    assemble_pressure!,
    extract_momentum_operators!,
    correct_velocity!,
    correct_fluxes!,
    continuity_residual_interior,
    momentum_residual,
    continuity_residual,
    compute_max_courant

# --- Linear Solver Infrastructure (Phase 5) ---
export
    FVMSolverConfig,
    FieldSolverConfig,
    default_solver_config,
    build_preconditioner

# --- RANS Turbulence Models (Phase 2a) ---
export
    # Abstract types
    AbstractRANSModel,
    # Model types
    KOmega,
    KOmegaSSTModel,
    SpalartAllmaras,
    # State
    RANSTurbulenceState,
    # Interface
    turbulent_viscosity!,
    solve_turbulence!,
    n_turbulence_fields,
    turbulence_field_names,
    # Solvers
    solve_simple_turbulent,
    solve_incompressible_turbulent,
    # Utilities
    compute_wall_distance,
    compute_strain_rate,
    compute_nu_eff,
    turbulence_inlet_bc,
    turbulence_wall_bc,
    # Wall functions
    spalding_u_tau,
    compute_nut_wall,
    equilibrium_k_wall,
    equilibrium_epsilon_wall,
    equilibrium_omega_wall,
    apply_wall_functions!

# --- LES & Hybrid Turbulence Models (Phase 2b) ---
export
    AbstractLESModel,
    AbstractHybridModel,
    LESTurbulenceState,
    Smagorinsky,
    WALE,
    DynamicSmagorinsky,
    DDES,
    compute_filter_width

# --- Conjugate Heat Transfer & Buoyancy (Phase 3) ---
export
    # Types
    FluidThermalProperties,
    SolidThermalProperties,
    ThermalState,
    ConjugateHeatTransferProblem,
    # Energy equation
    assemble_energy!,
    update_k_eff!,
    compute_alpha_eff,
    # Buoyancy
    compute_buoyancy_source,
    has_buoyancy,
    # Solid conduction
    assemble_solid_conduction!,
    solve_solid_conduction,
    # Conjugate
    solve_conjugate_ht,
    compute_interface_heat_flux,
    # Solver wrappers
    solve_simple_thermal,
    solve_incompressible_thermal,
    # BC convenience
    thermal_inlet_bc,
    thermal_insulated_bc,
    thermal_heated_wall_bc,
    thermal_convective_bc

# --- Post-Processing (Phase 12) ---
export
    # Field operations
    compute_vorticity,
    compute_q_criterion,
    compute_enstrophy,
    compute_courant_number,
    # Field statistics
    field_average,
    field_min_max,
    turbulence_intensity,
    # Wall quantities
    compute_wall_shear_stress,
    compute_y_plus,
    compute_wall_heat_flux,
    compute_nusselt_number,
    # Forces
    compute_forces,
    force_coefficients,
    # Sampling
    sample_line,
    sample_field_at_point

# --- Multiphase VOF (Phase 7) ---
export
    TwoPhaseProperties,
    VOFState,
    has_surface_tension,
    assemble_alpha!,
    compute_compression_flux,
    clip_alpha!,
    update_mixture_properties!,
    compute_curvature,
    compute_surface_tension_force,
    solve_vof

# --- Radiation (Phase 9) ---
export
    AbstractRadiationModel,
    P1Model,
    FvDOMModel,
    RadiationState,
    STEFAN_BOLTZMANN,
    assemble_p1!,
    solve_p1_radiation,
    solve_fvdom_radiation,
    compute_radiation_source,
    marshak_wall_bc,
    radiation_inlet_bc,
    solve_simple_thermal_radiation

# --- Combustion & Species Transport (Phase 8) ---
export
    CombustionProperties,
    SpeciesState,
    EddyDissipationModel,
    EddyDissipationConcept,
    assemble_species!,
    solve_species!,
    compute_edm_reaction_rates,
    compute_edc_reaction_rates,
    compute_heat_release,
    solve_simple_reacting,
    # Arrhenius finite-rate chemistry
    CollocatedArrheniusReaction,
    compute_arrhenius_reaction_rates,
    compute_fred_reaction_rates,
    # Multi-step mechanism (generalised Arrhenius)
    MultiStepMechanism,
    one_step_arrhenius_mechanism,
    compute_multi_step_rates,
    compute_multi_step_rates!,
    read_chemkin_mechanism,
    # Variable Lewis number species transport
    VariableLewis,
    species_diffusivity,
    lewis_number,
    # Flamelet-Generated Manifold (FGM) tabulated chemistry
    FGMTable,
    build_fgm_table_from_callback,
    lookup_fgm,
    lookup_fgm!,
    compute_fgm_table_from_cantera

# --- Lagrangian DPM (Phase 11) ---
export
    AbstractDragModel,
    StokesDrag,
    SchillerNaumann,
    compute_drag_force,
    AbstractParticleHeatTransfer,
    RanzMarshall,
    compute_particle_heat_transfer,
    compute_momentum_source,
    compute_energy_source,
    set_particle_properties!,
    advance_particles!,
    # Spray breakup
    AbstractBreakupModel,
    TABBreakup,
    KHRTBreakup,
    weber_number,
    should_breakup,
    breakup_diameter,
    apply_breakup!,
    # Collisions
    AbstractCollisionModel,
    ORourkeCollision,
    apply_collisions!

# --- Dynamic/Moving Mesh (Phase 10) ---
export
    AbstractMotionSolver, SolidBodyMotion, LaplacianMotion, MeshMotionState,
    compute_displacement!, update_mesh!, compute_mesh_flux!,
    ale_corrected_flux, solve_ale,
    compute_distance_diffusivity

# --- Pressure-Based Thermo + Rheology (Stage 3) ---
export
    AbstractThermoModel,
    IncompressibleThermo,
    IdealGas,
    BoussinesqThermo,
    SutherlandGas,
    SutherlandViscosity,
    density_at,
    viscosity_at,
    cp_at,
    beta_at,
    is_compressible,
    # Rheology
    AbstractRheology,
    NewtonianRheology,
    PowerLawRheology,
    BirdCarreauRheology,
    HerschelBulkleyRheology,
    CassonRheology

# --- Stage 5b MULES ---
export mules_limit_flux!
# --- Stage 5d GCL verification ---
export verify_gcl

# --- Stage 6a Moving Reference Frame ---
export
    AbstractMRFZone,
    RotationalMRFZone,
    angular_velocity_vector,
    mrf_momentum_source,
    mrf_momentum_source_2d_planar

# --- Stage 6c Porous media ---
export
    AbstractPorousModel,
    DarcyPorous,
    DarcyForchheimerPorous,
    OrthotropicPorous,
    porous_momentum_source

# --- Stage 6d Cavitation ---
export
    AbstractCavitationModel,
    KunzCavitation,
    SchnerrSauerCavitation,
    MerkleCavitation,
    cavitation_source

# --- Stage 6f Aeroacoustics ---
export
    FWHSurface,
    FWHObserver,
    curle_dipole_pressure,
    fwh_monopole_pressure

# --- Stage 6g Population balance moment methods ---
export
    qmom_recover_abscissae_weights,
    qmom_moment_source_growth,
    qmom_moment_source_aggregation,
    qmom_moment_source_breakage

# --- Stage 7a Solid mechanics ---
export
    IsotropicElastic,
    SolidDisplacementProblem,
    stress_tensor,
    small_strain_tensor,
    cantilever_tip_deflection

# --- Stage 7b FSI ---
export
    AitkenRelaxation,
    update_aitken!,
    FSIInterface,
    interface_residual_norm

# --- Stage 7d Function objects ---
export
    AbstractFunctionObject,
    PointProbe,
    ForceProbe,
    ExpressionBC,
    evaluate_expression_bc,
    FieldStatistics

# --- Stage 8a Mesh generation ---
export
    Octree,
    is_leaf,
    subdivide!,
    build_octree,
    count_leaves,
    intersects_sphere,
    refine_near_sphere!

# --- Stage 8c/d Collocated AMR + indicators ---
export
    RefinementMarker,
    mark_cells_by_gradient,
    flux_correction_factor,
    zz_error_indicator

# --- Stage 9e Matrix-free operator ---
export MatrixFreeLinearOperator

# --- Stage 9f Units integration ---
export
    strip_units,
    is_dimensionless,
    as_si_velocity,
    as_si_density,
    as_si_viscosity,
    as_si_temperature

# --- MPI Parallelism (Phase 6 / Stage 2) ---
export
    distribute_mesh,
    halo_exchange!,
    solve_simple_distributed,
    # Stage 2b/2c: dep-free geometric partitioning + local submesh extraction.
    # These run purely in the base module so the test suite can exercise the
    # partitioning logic without needing MPI loaded.
    partition_rcb,
    extract_local_mesh,
    LocalMeshData

export FVMGeometry,
    FVMProblem,
    FVMSystem,
    SteadyFVMProblem,
    BoundaryConditions,
    InternalConditions,
    Conditions,
    Neumann,
    Dudt,
    Dirichlet,
    Constrained,
    Robin,
    solve,
    remake,
    compute_flux,
    pl_interpolate,
    # Coordinate systems
    AbstractCoordinateSystem,
    Cartesian,
    Cylindrical,
    Spherical,
    geometric_volume_weight,
    geometric_flux_weight,
    get_coordinate_system,
    # Flux limiters
    AbstractLimiter,
    MinmodLimiter,
    SuperbeeLimiter,
    VanLeerLimiter,
    VenkatakrishnanLimiter,
    BarthJespersenLimiter,
    KorenLimiter,
    OspreLimiter,
    minmod,
    superbee,
    van_leer,
    venkatakrishnan,
    barth_jespersen,
    koren,
    ospre,
    apply_limiter,
    select_limiter,
    # Gradient reconstruction
    AbstractGradientMethod,
    GreenGaussGradient,
    LeastSquaresGradient,
    reconstruct_gradient,
    reconstruct_gradient_at_edge,
    reconstruct_gradient_at_point,
    reconstruct_all_gradients,
    # MUSCL scheme
    MUSCLScheme,
    muscl_reconstruct_face_value,
    muscl_reconstruct_edge_values,
    muscl_advective_flux,
    muscl_diffusive_flux,
    MUSCLFluxFunction,
    create_muscl_problem,
    # Advection-diffusion
    AdvectionDiffusionEquation,
    # Nonlinear BCs
    NonlinearDirichlet,
    NonlinearNeumann,
    NonlinearRobin,
    linearize_bc,
    compute_boundary_gradient,
    evaluate_nonlinear_bc,
    # Periodic BCs
    PeriodicBC,
    PeriodicNodeMapping,
    PeriodicConditions,
    compute_periodic_mapping,
    apply_periodic_constraints!,
    has_periodic_conditions,
    # Coupled multi-field BCs
    CoupledBC,
    CoupledDirichlet,
    CoupledNeumann,
    CoupledRobin,
    CoupledBoundaryConditions,
    evaluate_coupled_bc,
    add_coupled_bc!,
    get_coupled_bc,
    has_coupled_bc,
    get_target_field,
    # Anisotropic diffusion
    AnisotropicDiffusionEquation,
    make_rotation_tensor,
    make_spatially_varying_tensor,
    # Turbulence models
    StandardKEpsilon,
    KappaOmegaSST,
    compute_turbulent_viscosity,
    compute_strain_rate_magnitude,
    compute_production,
    compute_friction_velocity,
    k_wall_value,
    epsilon_wall_value,
    TurbulentWallBC,
    # Mesh abstractions
    AbstractBackend,
    CPUBackend,
    CUDASolverBackend,
    to_backend,
    to_host,
    supports_backend,
    backend_summary,
    AbstractMesh,
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
    ncells,
    nfaces,
    cell_center,
    cell_volume,
    face_area,
    face_owner,
    face_neighbor,
    ndims_mesh,
    cell_ijk,
    cell_idx_3d,
    # Equations of state
    AbstractEOS,
    IdealGasEOS,
    StiffenedGasEOS,
    pressure,
    sound_speed,
    internal_energy,
    total_energy,
    # Conservation laws
    AbstractConservationLaw,
    EulerEquations,
    NavierStokesEquations,
    IdealMHDEquations,
    fast_magnetosonic_speed,
    slow_magnetosonic_speed,
    nvariables,
    physical_flux,
    max_wave_speed,
    wave_speeds,
    conserved_to_primitive,
    primitive_to_conserved,
    # Riemann solvers
    AbstractRiemannSolver,
    LaxFriedrichsSolver,
    HLLSolver,
    HLLCSolver,
    HLLDSolver,
    solve_riemann,
    # Reconstruction
    CellCenteredMUSCL,
    NoReconstruction,
    reconstruct_interface,
    # PPM reconstruction
    PPMReconstruction,
    # WENO reconstruction
    WENO3,
    WENO5,
    nghost,
    reconstruct_interface_weno5,
    CharacteristicWENO,
    left_eigenvectors,
    right_eigenvectors,
    # Hyperbolic boundary conditions
    AbstractHyperbolicBC,
    TransmissiveBC,
    ReflectiveBC,
    InflowBC,
    PeriodicHyperbolicBC,
    DirichletHyperbolicBC,
    NoSlipBC,
    # Hyperbolic problem and solver
    HyperbolicProblem,
    HyperbolicProblem2D,
    HyperbolicProblem3D,
    solve_hyperbolic,
    initialize_2d,
    compute_dt,
    compute_dt_2d,
    compute_dt_3d,
    hyperbolic_rhs!,
    hyperbolic_rhs_2d!,
    hyperbolic_rhs_3d!,
    to_primitive,
    # Navier-Stokes
    thermal_conductivity,
    viscous_flux_1d,
    viscous_flux_x_2d,
    viscous_flux_y_2d,
    # Resistive MHD
    ResistiveMHDEquations,
    resistive_flux_x,
    resistive_flux_y,
    ohmic_heating,
    resistive_dt,
    # Hall MHD
    HallMHDEquations,
    whistler_speed,
    hall_flux_x,
    hall_flux_y,
    hall_dt,
    # Shallow Water
    ShallowWaterEquations,
    BottomTopography,
    topography_source_1d,
    # SR Hydro
    SRHydroEquations,
    srhydro_con2prim,
    # Two-Fluid Plasma
    TwoFluidEquations,
    ion_primitive,
    electron_primitive,
    ion_conserved,
    electron_conserved,
    lorentz_source_1d,
    lorentz_source_2d,
    # Reactive Euler
    ReactiveEulerEquations,
    euler_primitive,
    euler_conserved,
    species_mass_fractions,
    species_partial_densities,
    temperature,
    ArrheniusReaction,
    ReactionMechanism,
    ChemistrySource,
    # 2D mesh helpers
    cell_ij,
    cell_idx,
    # Stiff sources and IMEX
    AbstractStiffSource,
    ResistiveSource,
    CoolingSource,
    NullSource,
    evaluate_stiff_source,
    stiff_source_jacobian,
    AbstractIMEXScheme,
    IMEX_SSP3_433,
    IMEX_ARS222,
    IMEX_Midpoint,
    imex_tableau,
    imex_nstages,
    solve_hyperbolic_imex,
    # Positivity limiter
    PositivityLimiter,
    apply_positivity_limiter!,
    apply_positivity_limiter_2d!,
    limit_reconstructed_states,
    # Constrained transport (2D)
    CTData2D,
    initialize_ct!,
    initialize_ct_from_potential!,
    face_to_cell_B!,
    copy_ct,
    copyto_ct!,
    compute_emf_2d!,
    ct_update!,
    compute_divB,
    max_divB,
    l2_divB,
    # Constrained transport (3D)
    CTData3D,
    initialize_ct_3d!,
    initialize_ct_3d_from_potential!,
    face_to_cell_B_3d!,
    ct_update_3d!,
    ct_weighted_update_3d!,
    compute_divB_3d,
    max_divB_3d,
    l2_divB_3d,
    # AMR
    AMRBlock,
    AMRGrid,
    AbstractRefinementCriterion,
    GradientRefinement,
    CurrentSheetRefinement,
    is_leaf,
    active_blocks,
    blocks_at_level,
    max_active_level,
    block_cell_center,
    needs_refinement,
    needs_coarsening,
    refine_block!,
    coarsen_block!,
    regrid!,
    prolongate!,
    prolongate_B_divergence_preserving_2d!,
    restrict!,
    restrict_B_face_2d!,
    restrict_B_face_3d!,
    FluxRegister,
    reset_flux_register!,
    accumulate_fine_flux!,
    store_coarse_flux!,
    apply_flux_correction_2d!,
    apply_flux_correction_3d!,
    AMRProblem,
    solve_amr,
    compute_dt_amr,
    advance_level!,
    # Multi-rate subcycling
    SubcyclingScheme,
    solve_amr_subcycled,
    advance_level_subcycled!,
    compute_dt_subcycled,
    total_substeps,
    # SRMHD
    SRMHDEquations,
    lorentz_factor,
    srmhd_b_quantities,
    srmhd_con2prim,
    Con2PrimResult,
    # Spacetime metrics
    AbstractMetric,
    MinkowskiMetric,
    SchwarzschildMetric,
    KerrMetric,
    lapse,
    shift,
    spatial_metric,
    sqrt_gamma,
    inv_spatial_metric,
    MetricData2D,
    precompute_metric,
    precompute_metric_at_faces,
    # GRMHD
    GRMHDEquations,
    grmhd_con2prim,
    grmhd_con2prim_cached,
    grmhd_primitive_to_conserved_densitized,
    grmhd_prim2con_densitized_cached,
    grmhd_max_wave_speed_coord,
    grmhd_source_terms,
    # Unstructured hyperbolic solver
    UnstructuredHyperbolicMesh,
    UnstructuredHyperbolicProblem,
    rotate_to_normal,
    rotate_flux_from_normal,
    boundary_ghost_state,
    get_bc,
    # Multi-physics coupling
    AbstractOperator,
    AbstractSplittingScheme,
    LieTrotterSplitting,
    StrangSplitting,
    CoupledProblem,
    HyperbolicOperator,
    SourceOperator,
    advance!,
    compute_operator_dt,
    solve_coupled,
    cell_to_vertex,
    vertex_to_cell,
    # Dashboard data export
    FVMSnapshot,
    FVMSessionData,
    variable_names,
    mesh_to_dict,
    conserved_totals,
    snapshot_to_dict,
    session_to_dict,
    add_convergence_point!,
    hyperbolic_monitor,
    create_session_data,
    FVMMonitorCallback,
    export_session,
    import_session,
    serve_dashboard

# --- Semidiscrete Core (ODEProblem integration for hyperbolic solvers) ---
export
    # Cache types
    AbstractSemidiscreteCache,
    HyperbolicCache1D,
    HyperbolicCache2D,
    HyperbolicCache3D,
    UnstructuredCache,
    MHDCTCache2D,
    MHDCTCache3D,
    GRMHDCTCache2D,
    AMRCache,
    # Cache construction
    build_cache,
    build_mhd_ct_cache,
    build_grmhd_ct_cache,
    build_amr_cache,
    # State mapping
    unfold_to_padded!,
    fold_from_padded!,
    unfold_mhd_augmented!,
    fold_mhd_augmented!,
    initial_state_flat,
    initial_mhd_augmented_state,
    flatten_amr_state,
    unfold_amr!,
    fold_amr!,
    # CFL callback
    cfl_stepsize_callback,
    compute_initial_dt,
    # MHD stage limiter
    mhd_stage_limiter,
    # Solution accessors
    FVMSolutionAccessor,
    HyperbolicSolutionAccessor,
    MHDSolutionAccessor,
    MHD3DSolutionAccessor,
    AMRODESolutionAccessor,
    AMRSolution,
    get_conserved,
    get_primitive,
    get_coordinates,
    get_ct_state,
    # Canonical SciML contract
    sciml_problem,
    FVMSymbolicIndex,
    FVMVar,
    fvm_symbolic_index,
    solution_accessor,
    solution_snapshot,
    solution_coordinates,
    solution_state_layout,
    solution_variables

# --- I/O (from Simu.jl SimuIO migration) ---
export
    # Output Manager
    OutputSchedule,
    OutputTarget,
    Diagnostic,
    SimulationConfig,
    Provenance,
    OutputManager,
    validate_schedule,
    next_write_time,
    run_diagnostics,
    # Diagnostics
    volume_integral,
    conservation_summary,
    boundary_fluxes,
    flux_inout,
    write_boundary_flux_csv,
    write_operator_splits_csv,
    # VTK
    write_line_vtk,
    write_structured_vtk_3d,
    # Utils
    ensure_output_dirs,
    write_csv,
    stringify_keys,
    write_metadata_toml,
    print_scientific,
    print_with_units,
    print_table_header,
    print_table_row,
    print_progress,
    ensure_extension,
    safe_filename,
    # In-situ monitoring
    AbstractMonitor,
    Probe,
    IntegralMonitor,
    find_cell_containing,
    sample_probe,
    compute_integral,
    # Registry
    save_model_package,
    load_model_package,
    # HDF5 stubs
    write_solution_hdf5,
    read_solution_hdf5,
    # Checkpointing
    CheckpointManager,
    save_checkpoint,
    load_checkpoint

export
    supported_features,
    feature_maturity,
    feature_validation_status,
    feature_role,
    feature_solver_family,
    feature_required_ladder_stages,
    feature_claim_policy,
    feature_limitations,
    capability_matrix

# --- SciML Bridge (parabolic assembly → SciMLBase) ---
export
    parabolic_to_odefunction,
    parabolic_to_linearproblem

# --- Wave 4 fast-path (v3.105) — mesh gen / AMR / adjoint / KA-GPU / unit + prop extensions ---
export
    # Mesh generation (Stage 8)
    GmshPipeline,
    run_gmsh_pipeline,
    auto_remediate!,
    SnappyMesher,
    # Collocated AMR + error indicators
    residual_error_indicator,
    zz_error_indicator,
    mark_for_refinement,
    RefinementPlan,
    CoarseningPlan,
    apply_refinement!,
    apply_coarsening!,
    # Adjoint
    AbstractAdjointAlgorithm,
    SteadyAdjoint,
    TransientAdjoint,
    solve_adjoint,
    solve_steady_adjoint,
    solve_transient_adjoint,
    verify_adjoint_gradient,
    # Kernel backend trait
    KernelBackend,
    CPUBackend,
    KABackend,
    kernel_backend,
    per_term_ad,
    # Runtime function objects (string-DSL expression BC lives alongside the
    # closure-based `ExpressionBC` exported above — they are different types)
    StringExpressionBC,
    evaluate,
    Probe,
    Force,
    SamplingPlane,
    trigger_probe,
    # Unitful hook
    strip_units,
    annotate_units,
    is_unitful,
    # External property / solver stubs
    CoolPropFluid,
    PETScLinearSolver,
    # Wave 5 — true MPI decomposition + Eulerian two-fluid (experimental)
    LocalFVMMesh,
    build_local_mesh,
    partition_mesh_metis,
    TwoFluidProperties,
    TwoFluidState,
    TwoFluidSolver,
    AbstractDragClosure,
    IshiiZuberDrag,
    GibilaroDrag,
    drag_coefficient,
    drag_force_density,
    bubble_reynolds,
    stokes_limit_drag,
    density_ratio,
    enforce_volume_fraction_sum!,
    interphase_drag

using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload
@setup_workload begin
    @compile_workload begin
        # Compile a non-steady problem
        n = 5
        α = π / 4
        x₁ = [0.0, 1.0]
        y₁ = [0.0, 0.0]
        r₂ = fill(1, n)
        θ₂ = LinRange(0, α, n)
        x₂ = @. r₂ * cos(θ₂)
        y₂ = @. r₂ * sin(θ₂)
        x₃ = [cos(α), 0.0]
        y₃ = [sin(α), 0.0]
        x = [x₁, x₂, x₃]
        y = [y₁, y₂, y₃]
        boundary_nodes, points = convert_boundary_points_to_indices(x, y)
        tri = triangulate(points; boundary_nodes)
        A = get_area(tri)
        refine!(tri)
        mesh = FVMGeometry(tri)
        lower_bc = arc_bc = upper_bc = (x, y, t, u, p) -> zero(u)
        types = (Neumann, Dirichlet, Neumann)
        BCs = BoundaryConditions(mesh, (lower_bc, arc_bc, upper_bc), types)
        f = (x, y) -> 1 - sqrt(x^2 + y^2)
        D = (x, y, t, u, p) -> one(u)
        initial_condition = [
            f(x, y)
                for (x, y) in
                DelaunayTriangulation.DelaunayTriangulation.each_point(tri)
        ]
        final_time = 0.1
        prob = FVMProblem(mesh, BCs; diffusion_function = D, initial_condition, final_time)
        ode_prob = ODEProblem(prob)
        steady_prob = SteadyFVMProblem(prob)
        nl_prob = SteadyStateProblem(steady_prob)

        # Compile a system
        tri = triangulate_rectangle(0, 100, 0, 100, 5, 5, single_boundary = true)
        mesh = FVMGeometry(tri)
        bc_u = (x, y, t, (u, v), p) -> zero(u)
        bc_v = (x, y, t, (u, v), p) -> zero(v)
        BCs_u = BoundaryConditions(mesh, bc_u, Neumann)
        BCs_v = BoundaryConditions(mesh, bc_v, Neumann)
        q_u = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
            u = αu * x + βu * y + γu
            ∇u = (αu, βu)
            ∇v = (αv, βv)
            χu = p.c * u / (1 + u^2)
            _q = χu .* ∇v .- ∇u
            return _q
        end
        q_v = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
            ∇v = (αv, βv)
            _q = -p.D .* ∇v
            return _q
        end
        S_u = (x, y, t, (u, v), p) -> begin
            return u * (1 - u)
        end
        S_v = (x, y, t, (u, v), p) -> begin
            return u - p.a * v
        end
        q_u_parameters = (c = 4.0,)
        q_v_parameters = (D = 1.0,)
        S_v_parameters = (a = 0.1,)
        u_initial_condition = 0.01rand(DelaunayTriangulation.num_solid_vertices(tri))
        v_initial_condition = zeros(DelaunayTriangulation.num_solid_vertices(tri))
        final_time = 1000.0
        u_prob = FVMProblem(
            mesh, BCs_u;
            flux_function = q_u, flux_parameters = q_u_parameters,
            source_function = S_u,
            initial_condition = u_initial_condition, final_time = final_time
        )
        v_prob = FVMProblem(
            mesh, BCs_v;
            flux_function = q_v, flux_parameters = q_v_parameters,
            source_function = S_v, source_parameters = S_v_parameters,
            initial_condition = v_initial_condition, final_time = final_time
        )
        prob = FVMSystem(u_prob, v_prob)
        ode_prob = ODEProblem(prob)
        steady_prob = SteadyFVMProblem(prob)
        nl_prob = SteadyStateProblem(steady_prob)
    end
end
end

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
    MatrixOperator, ODEFunction, ODEProblem, SteadyStateProblem, remake
using SparseArrays: SparseArrays, sparse
using StaticArrays: StaticArrays, SVector
using Base.Threads

include("backends.jl")

# --- Parabolic Types & Mesh (from Simu.jl migration) ---
include("parabolic/types.jl")
include("parabolic/mesh/types.jl")
include("parabolic/mesh/structured.jl")
include("parabolic/mesh/curvilinear.jl")
include("parabolic/mesh/unstructured.jl")
include("parabolic/mesh/fvm_mesh.jl")
include("parabolic/mesh/io.jl")
include("parabolic/mesh/partitioning.jl")

# Flux limiters (canonical implementations — used by both vertex-centered and parabolic solvers)
include("schemes/limiters.jl")

# --- Parabolic Solver (from Simu.jl SimuFVM migration) ---
include("parabolic/models.jl")
include("parabolic/utils.jl")
include("parabolic/boundary_conditions.jl")
include("parabolic/gradients.jl")
include("parabolic/limiters.jl")   # ParabolicLimiters delegates to schemes/limiters.jl
include("parabolic/schemes.jl")
include("parabolic/compressible_fluxes.jl")
include("parabolic/turbulence.jl")
include("parabolic/particles.jl")
include("parabolic/fsi.jl")
include("parabolic/kernels.jl")
include("parabolic/assembly/assembly_1d.jl")
include("parabolic/assembly/assembly_2d.jl")
include("parabolic/assembly/assembly_3d.jl")
include("parabolic/assembly/assembly_cylindrical.jl")
include("parabolic/assembly/assembly_spherical.jl")
include("parabolic/assembly/assembly_unstructured.jl")
include("parabolic/assembly/assembly_curvilinear.jl")
include("parabolic/assembly/assembly_system.jl")

include("coordinate_systems.jl")
include("geometry.jl")
include("conditions.jl")
include("problem.jl")
include("equations/boundary_edge_contributions.jl")
include("equations/control_volumes.jl")
include("equations/dirichlet.jl")
include("equations/individual_flux_contributions.jl")
include("equations/main_equations.jl")
include("equations/shape_functions.jl")
include("equations/source_contributions.jl")
include("equations/triangle_contributions.jl")
include("solve.jl")
include("utils.jl")

# Schemes for higher-order methods (limiters moved earlier, before parabolic/limiters.jl)
include("schemes/gradients.jl")
include("schemes/muscl.jl")

# Advanced boundary conditions
include("conditions/nonlinear.jl")
include("conditions/periodic.jl")
include("conditions/coupled.jl")

include("specific_problems/abstract_templates.jl")
include("specific_problems/advection_diffusion_equation.jl")
include("specific_problems/anisotropic_diffusion.jl")

# Physics models
include("physics/turbulence/k_epsilon.jl")

# Hyperbolic solver framework (cell-centered FVM)
include("mesh/abstract_mesh.jl")
include("mesh/structured_mesh.jl")
include("eos/eos_interface.jl")
include("eos/ideal_gas.jl")
include("eos/stiffened_gas.jl")
include("hyperbolic/conservation_laws.jl")
include("hyperbolic/euler.jl")
include("hyperbolic/riemann_solvers.jl")
include("hyperbolic/reconstruction.jl")
include("hyperbolic/boundary_conditions_hyp.jl")
include("hyperbolic/hllc_solver.jl")
include("hyperbolic/mhd.jl")
include("hyperbolic/hlld_solver.jl")
include("hyperbolic/hyperbolic_problem.jl")
include("hyperbolic/hyperbolic_solve.jl")
include("hyperbolic/hyperbolic_problem_2d.jl")
include("hyperbolic/boundary_conditions_2d.jl")
include("hyperbolic/hyperbolic_solve_2d.jl")

# Constrained transport for MHD
include("constrained_transport/ct_data.jl")
include("constrained_transport/emf.jl")
include("constrained_transport/ct_update.jl")
include("constrained_transport/divb.jl")

# 2D MHD solver with constrained transport
include("hyperbolic/mhd_solve_2d.jl")

# Navier-Stokes (viscous terms)
include("hyperbolic/navier_stokes.jl")
include("hyperbolic/viscous_flux.jl")
include("hyperbolic/noslip_bc.jl")
include("hyperbolic/navier_stokes_solve.jl")
include("hyperbolic/navier_stokes_solve_2d.jl")

# Resistive MHD (magnetic diffusivity)
include("hyperbolic/resistive_mhd.jl")

# Hall MHD (whistler waves, ion-scale dynamics)
include("hyperbolic/hall_mhd.jl")

# Shallow Water Equations
include("hyperbolic/shallow_water.jl")

# SR Hydro (relativistic hydro without B)
include("hyperbolic/srhydro.jl")

# Two-Fluid Plasma (separate ion/electron fluids)
include("hyperbolic/two_fluid.jl")

# SRMHD (Special Relativistic MHD)
include("hyperbolic/con2prim.jl")
include("hyperbolic/srmhd.jl")
include("hyperbolic/srmhd_solve.jl")
include("hyperbolic/srmhd_solve_2d.jl")

# Spacetime metrics for GRMHD
include("metric/abstract_metric.jl")
include("metric/minkowski.jl")
include("metric/schwarzschild.jl")
include("metric/kerr.jl")
include("metric/metric_data.jl")

# GRMHD (General Relativistic MHD)
include("hyperbolic/grmhd.jl")
include("hyperbolic/grmhd_con2prim.jl")
include("hyperbolic/grmhd_solve_2d.jl")

# 3D mesh and Euler/MHD extensions
include("mesh/structured_mesh_3d.jl")
include("hyperbolic/euler_3d.jl")
include("hyperbolic/mhd_3d.jl")

# 3D Hyperbolic solver
include("hyperbolic/hyperbolic_problem_3d.jl")
include("hyperbolic/boundary_conditions_3d.jl")
include("hyperbolic/hyperbolic_solve_3d.jl")

# 3D Constrained transport for MHD
include("constrained_transport/ct_data_3d.jl")
include("constrained_transport/emf_3d.jl")
include("constrained_transport/ct_update_3d.jl")
include("constrained_transport/divb_3d.jl")

# 3D MHD solver with constrained transport
include("hyperbolic/mhd_solve_3d.jl")

# Block-structured AMR
include("amr/amr_grid.jl")
include("amr/refinement.jl")
include("amr/prolongation.jl")
include("amr/restriction.jl")
include("amr/flux_correction.jl")
include("amr/amr_solve.jl")

# Multi-rate (subcycling) time stepping for AMR
include("hyperbolic/multirate.jl")

# PPM reconstruction (Colella & Woodward 1984)
include("hyperbolic/ppm.jl")

# WENO reconstruction and IMEX time integration (Phase 8)
include("hyperbolic/weno3.jl")
include("hyperbolic/weno.jl")
include("hyperbolic/characteristic_projection.jl")
include("hyperbolic/stiff_sources.jl")
include("hyperbolic/imex.jl")
include("hyperbolic/imex_solve.jl")

# Reactive Euler (multi-species with Arrhenius chemistry)
include("hyperbolic/reactive_euler.jl")
include("hyperbolic/chemistry.jl")

# Positivity-preserving limiter (Zhang & Shu 2010)
include("hyperbolic/positivity_limiter.jl")

# Threading support for 2D solvers (Phase 12)
include("hyperbolic/threading.jl")

# Unstructured mesh hyperbolic solver (Phase 10)
include("mesh/unstructured_hyperbolic_mesh.jl")
include("hyperbolic/unstructured_problem.jl")
include("hyperbolic/unstructured_solve.jl")

# Multi-physics coupling via operator splitting (Phase 11)
include("coupling/abstract_coupling.jl")
include("coupling/operators.jl")
include("coupling/data_transfer.jl")
include("coupling/coupled_solve.jl")

# SciMLBase.remake for all problem types
include("remake.jl")

# Dashboard data export and monitoring
include("dashboard_types.jl")

# --- SciML Bridge (parabolic assembly → SciMLBase problem types) ---
# Use OrdinaryDiffEq.jl for time stepping, LinearSolve.jl for linear systems,
# NonlinearSolve.jl for Newton/JFNK, SparseDiffTools.jl for colored Jacobians,
# and SciMLSensitivity.jl for adjoint analysis.
include("parabolic/sciml_bridge.jl")

# --- I/O (from Simu.jl SimuIO migration) ---
include("io/utils.jl")
include("io/manager.jl")
include("io/diagnostics.jl")
include("io/vtk.jl")
include("io/insitu.jl")
include("io/registry.jl")
include("io/hdf5.jl")
include("io/checkpointing.jl")
include("capabilities.jl")

# --- Parabolic Core Types (from Simu.jl migration) ---
export
    # Tags
    AbstractTag,
    AbstractTagSpatial,
    AbstractTagTime,
    AbstractTagSteady,
    AbstractTagIVP,
    # Simulation, Problem, Solution
    AbstractSimulation,
    AbstractProblem,
    AbstractSolution,
    AbstractProblemIVP,
    AbstractProblemSteady,
    AbstractProblemPDE,
    # Geometry and Mesh
    AbstractGeometry,
    AbstractGeometryComponent,
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
    # Physics and Discretization
    AbstractField,
    AbstractPhysicsOperator,
    AbstractBoundaryCondition,
    AbstractInitialCondition,
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
    # Materials
    AbstractMaterialProperty,
    AbstractMaterial,
    AbstractMaterialModel,
    # Discretization
    AbstractDiscretization,
    AbstractSemidiscretization,
    AbstractFluxCalculator,
    AbstractReconstruction,
    # Solvers and Algorithms
    AbstractAlgorithm,
    # Callbacks and Diagnostics
    AbstractCallback,
    AbstractDiagnostic,
    # Post-processing and I/O
    AbstractOutputManager,
    AbstractConfig,
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
    AbstractFVMMesh,
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
    # Limiters (Parabolic submodule)
    ParabolicLimiters,
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
    get_velocity_at_face_1d,
    get_velocity_at_face_2d,
    compute_cfl_1d,
    compute_cfl_2d,
    assemble_mass_matrix_1d,
    assemble_mass_matrix_2d,
    assemble_mass_matrix_3d,
    TimeDependentDirichlet,
    TimeDependentNeumann,
    TimeDependentRobin

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
    capability_matrix

# --- SciML Bridge (parabolic assembly → SciMLBase) ---
export
    parabolic_to_odefunction,
    parabolic_to_linearproblem

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

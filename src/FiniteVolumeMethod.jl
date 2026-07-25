module FiniteVolumeMethod

using ChunkSplitters: ChunkSplitters, index_chunks
using CommonSolve: CommonSolve, solve, init, solve!, step!
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

# Module loading order: Geometry submodule → types → solvers → SciML bridge → I/O
include("geometry/Geometry.jl")
using .Geometry
# Explicit imports so the flat remainder can EXTEND these Geometry generics
# with unqualified `function ncells(...)` definitions (a bare `using` would
# silently shadow them with new local functions and fracture dispatch).
import .Geometry: ncells, nfaces, cell_center
include("numerics/Numerics.jl")
using .Numerics
# euler_3d.jl extends `total_energy` and incompressible/simple.jl extends
# `_solve_linear` with unqualified definitions; the underscored names are
# qualified as `FiniteVolumeMethod.NAME` inside package extensions and must
# resolve to the Numerics function objects.
import .Numerics: total_energy, _solve_linear, _unsupported_backend,
    autodiff_forward_step, _extension_preconditioner, _try_krylov_solver,
    # ext/FVMMPIExt and tests call the field-solve dispatcher (and tests
    # its solver-resolution helper) as FiniteVolumeMethod.<name>
    # (unexported as of Stage 4c)
    _dispatch_solve, _resolve_solver,
    # KA CPU-fallback kernels: reached as FiniteVolumeMethod.<name> by
    # test/v_and_v_ka_backend.jl (unresolvable since the Stage-3c wrap)
    interpolate_face_ka!, elementwise_sum_ka!
# ============================================================
# Domain / Problem Definitions
# ============================================================
#
# As of Stage 3d the parabolic family lives in
# the Parabolic submodule; this section wires the cell-vertex conditions
# engine, the Parabolic family, and the collocated Phase-0 operators.

include("vertex_conditions/VertexConditions.jl")
using .VertexConditions
# `ParametrisedFunction` is documented public API (docs/src/interface.md) but
# unexported by VertexConditions; the guard keeps `FiniteVolumeMethod.ParametrisedFunction`
# resolving for the docs `@docs`/`@ref` machinery.
import .VertexConditions: ParametrisedFunction

include("parabolic/Parabolic.jl")
using .Parabolic
# sciml/solve.jl (flat) and qualified test access use these Parabolic
# internals, unexported as of Stage 4c.
import .Parabolic: _neqs, _get_boundary_flux
# WYOS ("writing your own solver") helpers: documented public API referenced as
# `FiniteVolumeMethod.X` in docs/src/wyos/overview.md, unexported by Parabolic.
import .Parabolic: create_rhs_b, two_point_interpolant,
    neumann_boundary_edge_contributions!, non_neumann_boundary_edge_contributions!

# Stage-4b prefix resolution: the Simu.jl-migration names lost their interim
# `Parabolic` prefix (canonical access is module-qualified, e.g.
# `Parabolic.DirichletBC`). The old names remain as deprecated top-level
# aliases until Stage 8. Stage 5b cleared the six holdouts that kept the
# prefix: ParabolicNonlinear{Dirichlet,Neumann,Robin} and ParabolicCoupledBC
# were deleted outright — they were unreachable stubs whose only consumers
# (linearize_nonlinear_bc) were themselves never called, and VertexConditions
# owns the working implementations. ParabolicPeriodicBC became
# StructuredPeriodicBC: it pairs structured-mesh face symbols, so it was never
# a twin of the unstructured segment-index PeriodicBC. parabolic_sound_speed
# folded into Numerics.sound_speed, and parabolic_compute_friction_velocity
# became rough_wall_friction_velocity (its roughness argument is genuinely
# absent from the collocated compute_friction_velocity).
Base.@deprecate_binding ParabolicDirichlet DirichletBC
Base.@deprecate_binding ParabolicNeumann NeumannBC
Base.@deprecate_binding ParabolicRobin RobinBC
Base.@deprecate_binding ParabolicKEpsilon KEpsilon
Base.@deprecate_binding ParabolicTurbulentWall TurbulentWall
Base.@deprecate_binding parabolic_to_odefunction to_odefunction
Base.@deprecate_binding parabolic_to_linearproblem to_linearproblem

# The collocated (OpenFOAM-style) family: operators, incompressible
# SIMPLE/PISO/PIMPLE, multiphase, DPM, dynamic mesh, collocated AMR,
# post-processing, zone models, and nested Collocated.Physics
# (turbulence/thermal/radiation/combustion). Loads after Parabolic:
# its BC handling dispatches on AbstractBoundaryCondition.
include("collocated/Collocated.jl")
using .Collocated
# Dispatch-fracture guards + qualified-internal passthroughs (Stage-3
# recipe): the flat pressure_based/ family calls unexported
# incompressible/cyclic internals, solid_mechanics uses _face_tag, and
# tests/validation/docs reach the remaining internals as
# FiniteVolumeMethod.<name> — all must resolve to the Collocated
# bindings (temporary over-import, curated in Stage 4).
import .Collocated: DynamicContactAngle, KHACTBreakup, LISABreakup,
    NoMassTransfer, StaticContactAngle, TwoFluidProblem,
    _cyclic_cell_pairs, _extract_component, _face_diffusivity, _face_tag,
    _make_incompressible_workspace, _make_scalar_field,
    _needs_pressure_reference, _particle_reynolds, _pimple_step!,
    _set_component!, _snapshot_old_time!, _velocity_labels,
    _non_ortho_E_magnitude, _zz_indicator_smoothed, add_darcy_forchheimer_source!,
    apply_contact_angle, apply_cyclic_to_equation!,
    assemble_isoadvector_flux!, collect_cyclic_pairs, compute_HbyA_flux,
    couple_primary_breakup_fsi!, cox_voinov_angle, expand_bcs_pressure,
    expand_pressure_bc, expand_velocity_bc, fix_pressure_reference!,
    kunz_cond_rate, kunz_rate, kunz_vap_rate, least_squares_gradient,
    merkle_cond_rate, merkle_rate, merkle_vap_rate, run!,
    schnerr_sauer_bubble_radius, schnerr_sauer_rate, solve_two_fluid,
    two_fluid_mixture_continuity_residual, under_relax_momentum!,
    update!, update_boundary_cyclic!, update_boundary_pressure!,
    update_boundary_velocity!, warn_experimental!,
    # Collocated.Physics public passthroughs (exported by Physics)
    T_from_h, h_from_T, patankar_interface_coupling,
    EquilibriumWMLES, IDDES, WSGGMModel, compute_band_emissivity,
    compute_band_weight, enthalpy_bcs_from_temperature,
    enthalpy_field_from_temperature, iddes_blended_length,
    scattering_phase_value, scattering_source_contribution,
    solve_wsggm_radiation, temperature_from_enthalpy!,
    turbulent_viscosity_sa!, wmles_wall_nut, wmles_wall_shear,
    wsggm_effective_absorption,
    # MRF internals reached via `using FiniteVolumeMethod:` in the
    # v_and_v_mrf_* tests (unresolvable since the Stage-3f wrap)
    add_mrf_source!, add_multi_mrf_source!, centrifugal_force,
    coriolis_force, mrf_cell_source
# Collocated.Physics underscore internals (unexported as of Stage 4c);
# tests/docs reach them as FiniteVolumeMethod.<name>. Imported straight
# from the nested module: Collocated no longer carries pass-through
# bindings for them (ExplicitImports would flag those as stale there).
import .Collocated.Physics: _sym_self_magnitude_sq, _durbin_C_T,
    _wall_projection, _EDC_FALLBACK_MIXING_RATE, _apply_durbin_cap!,
    _blend, _cell_absorption, _ddes_length_scale, _ddes_shielding,
    _iddes_alpha, _iddes_f_B, _iddes_f_d_tilde, _iddes_f_dt, _iddes_f_e,
    _iddes_r_dl, _iddes_r_dt, _s12_quadrature, _s2_quadrature,
    _s4_quadrature, _s6_quadrature, _s8_quadrature, _sa_fv1,
    _species_index, _sst_F1, _sst_F2, _sym_contract, _test_filter,
    _update_turbulence!

# ============================================================
# Discretization / Assembly Kernels
# ============================================================
#
# This section retains the current include order for reconstruction,
# assembly, update kernels, and legacy solve paths while making the
# ownership boundary explicit for the v2 refactor.

# The cell-centered hyperbolic family (conservation laws, Riemann solvers,
# reconstruction, CT, metrics, AMR, and the semidiscrete SciML bridge) lives
# in the Hyperbolic submodule. It must precede solve.jl, which calls
# sciml_problem/_merge_problem_callbacks from the module.
include("hyperbolic/Hyperbolic.jl")
using .Hyperbolic
# Dispatch-fracture guards + qualified-internal passthroughs (Stage-3 recipe):
# the flat remainder extends fvm_symbolic_index/_amr_symbolic_index
# (sciml/symbolic_indexing.jl) and variable_names (dashboard consumers) with
# unqualified definitions, ext/FVMCUDAExt extends FVM._solve_hyperbolic, and
# tests/validation/docs call the remaining unexported internals as
# FiniteVolumeMethod.<name> — all must resolve to the Hyperbolic bindings.
import .Hyperbolic: fvm_symbolic_index, _amr_symbolic_index,
    _mhd_ct_2d_symbolic_index, _mhd_ct_3d_symbolic_index, variable_names,
    solve_hyperbolic, _solve_hyperbolic, solve_hyperbolic_imex,
    initialize_1d, initialize_3d, _cell_center_coords_2d, _reconstruct_face,
    _reconstruct_face_2d, _reconstruct_face_2d_y, reconstruct_interface_1d,
    _reflect_primitive, _nghost_for_reconstruction, _merge_problem_callbacks,
    _problem_callback, _compute_dt_2d_threaded, _hyperbolic_rhs_2d_threaded!,
    _implicit_solve_1d!, _implicit_solve_2d!, _implicit_solve_2d_threaded!,
    apply_bc_left!,
    apply_bc_right!, apply_bc_2d_left!, apply_bc_2d_bottom!,
    apply_boundary_conditions!, apply_boundary_conditions_2d!,
    apply_boundary_conditions_3d!, apply_periodic_bcs!,
    grmhd_recover_primitive_field, _grmhd_coord_wave_speeds,
    _grmhd_wave_speeds, _grmhd_valencia_flux, _grmhd_stage_rhs!,
    _grmhd_initialize_densitized_2d!, _weno3_reconstruct_left,
    _weno3_reconstruct_right, _weno5_reconstruct_left,
    _weno5_reconstruct_right,
    # mesh_generation/octree.jl (flat, later in this section) adds an
    # is_leaf(::Octree) method to what is one shared generic — import so it
    # extends the Hyperbolic (AMRBlock) function instead of shadowing it.
    is_leaf

include("sciml/solve.jl")

# Experimental scaffolds (Stage 3h): quarantined in the Experimental
# submodule — every dir is either manifest-`experimental` (mpi_parallel;
# pressure_based/adjoint/mesh_generation as smoke-only evidence) or has
# no manifest coverage (aeroacoustics, population_balance,
# solid_mechanics, fsi). Entry points warn once per feature on first
# use. The hyperbolic coupling/ files and the whole collocated family
# moved into the Hyperbolic and Collocated submodules in Stages 3e/3f.
include("experimental/Experimental.jl")
using .Experimental
# Qualified-internal passthroughs (Stage-3 recipe; temporary until the
# Stage-4 export curation): the MPI/Metis/CoolProp/Gmsh package
# extensions add methods as `function FiniteVolumeMethod.NAME(...)` to
# the Experimental stub seams, and tests/docs reach the remaining
# unexported internals as FiniteVolumeMethod.<name> — every one must
# resolve to the Experimental binding or extension dispatch fractures.
import .Experimental: solve_simple_distributed, distribute_mesh,
    halo_exchange!, partition_mesh_metis, partition_rcb,
    extract_local_mesh, build_local_mesh, LocalFVMMesh, LocalMeshData,
    CoolPropFluid, coolprop_density, coolprop_viscosity,
    coolprop_specific_heat, coolprop_thermal_conductivity, GmshPipeline,
    run_gmsh_pipeline,
    # test/docs-qualified internals
    CompressibleSIMPLE, CompressiblePIMPLE, CompressibleProblem,
    CompressibleState, solve_compressible, PengRobinson, RedlichKwong,
    TabulatedProperties, compute_face_densities!, update_density!,
    update_viscosity!, update_mass_flux!, face_density,
    solve_linear_elasticity, solve_finite_strain, SolidMechanicsState,
    SolidProperties, solve_partitioned_fsi, update_aitken_omega!,
    leaves, _uniform_refine!, read_stl_ascii, write_stl_ascii,
    solve_transient_adjoint_linear, UniformCheckpoint, add_checkpoint!,
    nearest_checkpoint, restore_between, Sutherland,
    build_castellated_mesh, build_snappy_mesh, cell_count,
    snap_to_surface!, triangle_intersects_aabb,
    # standalone-shim support: v_and_v_compressible.jl may eval the
    # compressible sources into this module, whose drivers call it
    _experimental_warn

# ============================================================
# SciML Adapters / Accessors
# ============================================================
#
# All canonical SciML problem construction, remake behavior, cache
# layout, and solution-accessor logic is collected here.

# The semidiscrete cache/state-mapping/ODE-construction/contract/accessor
# files moved into the Hyperbolic submodule in Stage 3e; the
# incompressible solution/façade moved into Collocated in Stage 3f. The
# cross-family glue below stays flat.
# sciml_structures.jl and remake.jl define the SciMLStructures/remake
# methods for IncompressibleProblem — import so they extend against the
# Collocated-owned type.
import .Collocated: IncompressibleProblem, SteadyIncompressibleProblem, AnyIncompressibleProblem,
    CollocatedSymbolicIndex
include("sciml/symbolic_indexing.jl")
include("sciml/sciml_structures.jl")
include("sciml/remake.jl")

# ============================================================
# Extensions / Tooling / Output
# ============================================================
#
# The dashboard/output/diagnostics family lives in the FVMIO submodule.
# capabilities.jl stays flat: it is coupled to validation/manifest.{jl,toml}.

include("io/FVMIO.jl")
using .FVMIO
# Qualified-internal passthroughs (Stage-3 recipe): the six package
# extensions add methods as `function FiniteVolumeMethod.NAME(...)` to the
# FVMIO extension stubs (dashboard monitors/session IO, HDF5, checkpointing,
# VTK 3D) and reference the session types and dict helpers the same way —
# every one of these must resolve to the FVMIO binding, or the extension
# method-adds silently create shadow generics and dispatch fractures.
import .FVMIO: CheckpointManager, FVMMonitorCallback, FVMSessionData,
    FVMSnapshot, conserved_totals, create_session_data, export_session,
    hyperbolic_monitor, import_session, load_checkpoint, mesh_to_dict,
    read_solution_hdf5, save_checkpoint, serve_dashboard, session_to_dict,
    snapshot_to_dict, stringify_keys, write_solution_hdf5,
    write_structured_vtk_3d

include("capabilities.jl")

# The curated export surface (Stage 4c) — the only export site of this module.
include("api.jl")

# Submodules are deliberately unexported (access is qualified, e.g.
# `FiniteVolumeMethod.Parabolic`); mark them API so tooling and docs treat
# the qualified paths as supported.
using Compat: @compat
@compat public Geometry, Numerics, VertexConditions, Parabolic, Collocated,
    Hyperbolic, Experimental, FVMIO,
    # supported-but-unexported flat API: the named-tunable registry
    # (sciml/sciml_structures.jl, Stage 1e)
    register_tunable!, tunable_schema

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

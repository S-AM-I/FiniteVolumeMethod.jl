# ============================================================
# Layer 2: Discretization / Assembly Kernels
# ============================================================
#
# This layer retains the current include order for reconstruction,
# assembly, update kernels, and legacy solve paths while making the
# ownership boundary explicit for the v2 refactor.

# The cell-centered hyperbolic family (conservation laws, Riemann solvers,
# reconstruction, CT, metrics, AMR, and the semidiscrete SciML bridge) lives
# in the Hyperbolic submodule. It must precede solve.jl, which calls
# sciml_problem/_merge_problem_callbacks from the module.
include("../hyperbolic/Hyperbolic.jl")
using .Hyperbolic
# Dispatch-fracture guards + qualified-internal passthroughs (Stage-3 recipe):
# the flat remainder extends fvm_symbolic_index/_amr_symbolic_index
# (core/symbolic_indexing.jl) and variable_names (dashboard consumers) with
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
    # mesh_generation/octree.jl (flat, later in this layer) adds an
    # is_leaf(::Octree) method to what is one shared generic — import so it
    # extends the Hyperbolic (AMRBlock) function instead of shadowing it.
    is_leaf

include("../solve.jl")

# Flat remainder of Layer 2 (experimental scaffolds; Stage 3h candidates).
# The hyperbolic coupling/ files and the whole collocated family
# (operators, incompressible, physics, multiphase, lagrangian,
# dynamic_mesh, function_objects, collocated AMR, zone models,
# postprocessing) moved into the Hyperbolic and Collocated submodules
# in Stages 3e/3f.
include("../pressure_based/thermo_models.jl")
include("../pressure_based/rheology.jl")
include("../pressure_based/coolprop_stub.jl")
include("../aeroacoustics/fwh.jl")
include("../aeroacoustics/pml.jl")
include("../population_balance/qmom.jl")
include("../population_balance/types.jl")
include("../population_balance/dqmom.jl")
include("../population_balance/class_method.jl")
include("../solid_mechanics/types.jl")
include("../solid_mechanics/linear_elasticity.jl")
include("../solid_mechanics/finite_strain.jl")
include("../solid_mechanics/solvers.jl")
include("../fsi/coupling.jl")
include("../fsi/interface.jl")
include("../fsi/partitioned.jl")
include("../mesh_generation/octree.jl")
include("../mesh_generation/stl_reader.jl")
include("../mesh_generation/snap.jl")
include("../mesh_generation/snappy.jl")
include("../mesh_generation/gmsh_pipeline.jl")

# Compressible pressure-based family (Wave 1)
# Extends SIMPLE/PIMPLE with density coupling + EOS dispatch; calls
# unexported Collocated internals via the Layer-1 import guard.
include("../pressure_based/eos_coupling.jl")
include("../pressure_based/compressible_simple.jl")
include("../pressure_based/compressible_pimple.jl")

# Discrete adjoint (Wave 4) — steady linear-system identity; transient stubbed
include("../adjoint/types.jl")
include("../adjoint/steady.jl")
include("../adjoint/checkpointing.jl")
include("../adjoint/reverse_sweep.jl")
include("../adjoint/transient.jl")
include("../adjoint/solvers.jl")

# MPI Parallelism stubs (Phase 6)
# Concrete implementations live in ext/FVMMPIExt/.
include("../parallel/stubs.jl")
include("../parallel/rcb_partitioner.jl")
include("../parallel/local_mesh.jl")
include("../parallel/metis_stub.jl")

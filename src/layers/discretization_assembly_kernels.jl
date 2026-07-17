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


include("../physics/turbulence/k_epsilon.jl")

include("../coupling/abstract_coupling.jl")
include("../coupling/operators.jl")
include("../coupling/data_transfer.jl")
include("../coupling/coupled_solve.jl")

# Incompressible Navier-Stokes — SIMPLE/PISO/PIMPLE (Phase 1)
# Depends on Phase 0 collocated operators from Layer 1.
include("../pressure_based/thermo_models.jl")
include("../pressure_based/rheology.jl")
include("../pressure_based/coolprop_stub.jl")
# Stage 6: greenfield physics modules. Ordered before incompressible/types
# to keep all physics-trait supertypes available when IncompressibleProblem
# and its helpers first refer to them.
include("../mrf/types.jl")
include("../mrf/momentum_source.jl")
include("../mrf/multi_zone.jl")
include("../porous/types.jl")
include("../porous/darcy_forchheimer.jl")
include("../cavitation/types.jl")
include("../cavitation/kunz.jl")
include("../cavitation/schnerr_sauer.jl")
include("../cavitation/merkle.jl")
include("../cavitation/solvers.jl")
include("../aeroacoustics/fwh.jl")
include("../aeroacoustics/pml.jl")
include("../population_balance/qmom.jl")
include("../population_balance/types.jl")
include("../population_balance/dqmom.jl")
include("../population_balance/class_method.jl")
# Stage 7: solid mechanics, FSI, function objects.
include("../solid_mechanics/types.jl")
include("../solid_mechanics/linear_elasticity.jl")
include("../solid_mechanics/finite_strain.jl")
include("../solid_mechanics/solvers.jl")
include("../fsi/coupling.jl")
include("../fsi/interface.jl")
include("../fsi/partitioned.jl")
include("../function_objects/types.jl")
include("../function_objects/expression_bc.jl")
# Stage 8: mesh generation + collocated AMR + error indicators.
include("../mesh_generation/octree.jl")
include("../mesh_generation/stl_reader.jl")
include("../mesh_generation/snap.jl")
include("../mesh_generation/snappy.jl")
include("../mesh_generation/gmsh_pipeline.jl")
include("../amr_collocated/adapt.jl")
include("../amr_collocated/error_indicators.jl")
include("../amr_collocated/refinement.jl")
include("../amr_collocated/coarsening.jl")
include("../incompressible/types.jl")
include("../incompressible/boundary_conditions.jl")
include("../incompressible/momentum.jl")
include("../incompressible/pressure.jl")
include("../incompressible/correction.jl")
include("../incompressible/residuals.jl")
include("../incompressible/simple.jl")
include("../incompressible/piso.jl")
include("../incompressible/pimple.jl")

# Compressible pressure-based family (Wave 1)
# Extends SIMPLE/PIMPLE with density coupling + EOS dispatch.
include("../pressure_based/eos_coupling.jl")
include("../pressure_based/compressible_simple.jl")
include("../pressure_based/compressible_pimple.jl")

# Linear Solver Infrastructure (Phase 5)
# Must come after incompressible (provides _solve_linear) and before turbulence.
# Kernel dispatch for KA/Enzyme extensions (v3.0 ships CPU path only)
# Discrete adjoint (Wave 4) — steady linear-system identity; transient stubbed
include("../adjoint/types.jl")
include("../adjoint/steady.jl")
include("../adjoint/checkpointing.jl")
include("../adjoint/reverse_sweep.jl")
include("../adjoint/transient.jl")
include("../adjoint/solvers.jl")

# RANS Turbulence Models (Phase 2a)
# Depends on Phase 0 operators + Phase 1 incompressible solver.
include("../turbulence/interface.jl")
include("../turbulence/strain_rate.jl")
include("../turbulence/wall_distance.jl")
include("../turbulence/k_epsilon_rans.jl")
include("../turbulence/k_omega.jl")
include("../turbulence/k_omega_sst.jl")
include("../turbulence/spalart_allmaras.jl")
include("../turbulence/wall_functions.jl")
include("../turbulence/solvers.jl")

# LES & Hybrid Turbulence Models (Phase 2b)
include("../turbulence/les_types.jl")
include("../turbulence/smagorinsky.jl")
include("../turbulence/wale.jl")
include("../turbulence/dynamic_smagorinsky.jl")
include("../turbulence/ddes.jl")
include("../turbulence/wmles.jl")
include("../turbulence/sa_ddes.jl")
include("../turbulence/iddes.jl")

# Conjugate Heat Transfer & Buoyancy (Phase 3)
# Depends on Phase 0 operators + Phase 1 incompressible + Phase 2a turbulence.
include("../thermal/types.jl")
include("../thermal/energy_equation.jl")
include("../thermal/enthalpy_equation.jl")
include("../thermal/buoyancy.jl")
include("../thermal/solid_conduction.jl")
include("../thermal/conjugate.jl")
include("../thermal/solvers.jl")

# Multiphase VOF (Phase 7)
# Depends on Phase 0 operators + Phase 1 incompressible.
include("../multiphase/types.jl")
include("../multiphase/mixture.jl")
include("../multiphase/boundedness.jl")
include("../multiphase/alpha_transport.jl")
include("../multiphase/surface_tension.jl")
include("../multiphase/iso_advector.jl")
include("../multiphase/solvers.jl")
# Eulerian two-fluid (Wave 5, experimental) + Ishii-Zuber / Gibilaro drag closures
include("../multiphase/drag_closures.jl")
include("../multiphase/two_fluid.jl")
include("../multiphase/mass_transfer.jl")
include("../multiphase/two_fluid_solver.jl")

# Radiation (Phase 9)
# Depends on Phase 0 operators + Phase 3 thermal.
include("../radiation/types.jl")
include("../radiation/p1.jl")
include("../radiation/solvers.jl")
include("../radiation/fvdom.jl")
include("../radiation/wsggm.jl")

# Combustion & Species Transport (Phase 8)
# Depends on Phase 0 operators + Phase 1 incompressible + Phase 2a turbulence + Phase 3 thermal.
include("../combustion/types.jl")
include("../combustion/variable_lewis.jl")
include("../combustion/species_transport.jl")
include("../combustion/edm.jl")
include("../combustion/edc.jl")
include("../combustion/arrhenius.jl")
include("../combustion/multi_step.jl")
include("../combustion/fgm.jl")
include("../combustion/solvers.jl")

# Lagrangian DPM (Phase 11)
include("../lagrangian/drag_models.jl")
include("../lagrangian/heat_transfer.jl")
include("../lagrangian/two_way_coupling.jl")
include("../lagrangian/particle_solver.jl")
include("../lagrangian/spray.jl")
include("../lagrangian/collisions.jl")
include("../lagrangian/agglomeration.jl")
include("../lagrangian/primary_breakup.jl")
include("../lagrangian/injection.jl")

# Dynamic/Moving Mesh (Phase 10)
include("../dynamic_mesh/types.jl")
include("../dynamic_mesh/solid_body.jl")
include("../dynamic_mesh/laplacian_motion.jl")
include("../dynamic_mesh/mesh_update.jl")
include("../dynamic_mesh/ale.jl")
include("../dynamic_mesh/six_dof.jl")
include("../dynamic_mesh/topo_changer.jl")
include("../dynamic_mesh/overset.jl")
include("../dynamic_mesh/ami.jl")
# Primary-breakup FSI coupling needs MeshMotionState (from dynamic_mesh/types.jl)
include("../lagrangian/primary_breakup_fsi.jl")

# MPI Parallelism stubs (Phase 6)
# Concrete implementations live in ext/FVMMPIExt/.
include("../parallel/stubs.jl")
include("../parallel/rcb_partitioner.jl")
include("../parallel/local_mesh.jl")
include("../parallel/metis_stub.jl")

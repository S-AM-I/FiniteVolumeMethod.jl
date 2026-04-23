# ============================================================
# Layer 2: Discretization / Assembly Kernels
# ============================================================
#
# This layer retains the current include order for reconstruction,
# assembly, update kernels, and legacy solve paths while making the
# ownership boundary explicit for the v2 refactor.

include("../equations/boundary_edge_contributions.jl")
include("../equations/control_volumes.jl")
include("../equations/dirichlet.jl")
include("../equations/individual_flux_contributions.jl")
include("../equations/main_equations.jl")
include("../equations/shape_functions.jl")
include("../equations/source_contributions.jl")
include("../equations/triangle_contributions.jl")
include("../solve.jl")
include("../utils.jl")

include("../schemes/gradients.jl")
include("../schemes/muscl.jl")

include("../conditions/nonlinear.jl")
include("../conditions/periodic.jl")
include("../conditions/coupled.jl")

include("../specific_problems/abstract_templates.jl")
include("../specific_problems/advection_diffusion_equation.jl")
include("../specific_problems/anisotropic_diffusion.jl")

include("../physics/turbulence/k_epsilon.jl")

include("../mesh/structured_mesh.jl")
include("../eos/eos_interface.jl")
include("../eos/ideal_gas.jl")
include("../eos/stiffened_gas.jl")
include("../hyperbolic/conservation_laws.jl")
include("../hyperbolic/euler.jl")
include("../hyperbolic/riemann_solvers.jl")
include("../hyperbolic/reconstruction.jl")
include("../hyperbolic/boundary_conditions_hyp.jl")
include("../hyperbolic/hllc_solver.jl")
include("../hyperbolic/mhd.jl")
include("../hyperbolic/hlld_solver.jl")
include("../hyperbolic/hyperbolic_problem.jl")
include("../core/legacy_api.jl")
include("../hyperbolic/hyperbolic_solve.jl")
include("../hyperbolic/hyperbolic_problem_2d.jl")
include("../hyperbolic/boundary_conditions_2d.jl")
include("../hyperbolic/hyperbolic_solve_2d.jl")

include("../constrained_transport/ct_data.jl")
include("../constrained_transport/emf.jl")
include("../constrained_transport/ct_update.jl")
include("../constrained_transport/divb.jl")

include("../hyperbolic/mhd_solve_2d.jl")

include("../hyperbolic/navier_stokes.jl")
include("../hyperbolic/viscous_flux.jl")
include("../hyperbolic/noslip_bc.jl")
include("../hyperbolic/navier_stokes_solve.jl")
include("../hyperbolic/navier_stokes_solve_2d.jl")

include("../hyperbolic/resistive_mhd.jl")
include("../hyperbolic/hall_mhd.jl")
include("../hyperbolic/shallow_water.jl")
include("../hyperbolic/srhydro.jl")
include("../hyperbolic/two_fluid.jl")

include("../hyperbolic/con2prim.jl")
include("../hyperbolic/srmhd.jl")
include("../hyperbolic/srmhd_solve.jl")
include("../hyperbolic/srmhd_solve_2d.jl")

include("../metric/abstract_metric.jl")
include("../metric/minkowski.jl")
include("../metric/schwarzschild.jl")
include("../metric/kerr.jl")
include("../metric/metric_data.jl")

include("../hyperbolic/grmhd.jl")
include("../hyperbolic/grmhd_con2prim.jl")
include("../hyperbolic/grmhd_solve_2d.jl")

include("../mesh/structured_mesh_3d.jl")
include("../hyperbolic/euler_3d.jl")
include("../hyperbolic/mhd_3d.jl")

include("../hyperbolic/hyperbolic_problem_3d.jl")
include("../hyperbolic/boundary_conditions_3d.jl")
include("../hyperbolic/hyperbolic_solve_3d.jl")

include("../constrained_transport/ct_data_3d.jl")
include("../constrained_transport/emf_3d.jl")
include("../constrained_transport/ct_update_3d.jl")
include("../constrained_transport/divb_3d.jl")

include("../hyperbolic/mhd_solve_3d.jl")

include("../amr/amr_grid.jl")
include("../amr/refinement.jl")
include("../amr/prolongation.jl")
include("../amr/restriction.jl")
include("../amr/flux_correction.jl")
include("../amr/amr_solve.jl")

include("../hyperbolic/multirate.jl")

include("../hyperbolic/ppm.jl")

include("../hyperbolic/weno3.jl")
include("../hyperbolic/weno.jl")
include("../hyperbolic/characteristic_projection.jl")
include("../hyperbolic/stiff_sources.jl")
include("../hyperbolic/imex.jl")
include("../hyperbolic/imex_solve.jl")

include("../hyperbolic/reactive_euler.jl")
include("../hyperbolic/chemistry.jl")

include("../hyperbolic/positivity_limiter.jl")
include("../hyperbolic/threading.jl")

include("../mesh/unstructured_hyperbolic_mesh.jl")
include("../hyperbolic/unstructured_problem.jl")
include("../hyperbolic/unstructured_solve.jl")

include("../coupling/abstract_coupling.jl")
include("../coupling/operators.jl")
include("../coupling/data_transfer.jl")
include("../coupling/coupled_solve.jl")

# Incompressible Navier-Stokes — SIMPLE/PISO/PIMPLE (Phase 1)
# Depends on Phase 0 collocated operators from Layer 1.
include("../pressure_based/thermo_models.jl")
include("../pressure_based/rheology.jl")
# Stage 6: greenfield physics modules. Ordered before incompressible/types
# to keep all physics-trait supertypes available when IncompressibleProblem
# and its helpers first refer to them.
include("../mrf/types.jl")
include("../porous/types.jl")
include("../cavitation/types.jl")
include("../aeroacoustics/fwh.jl")
include("../population_balance/qmom.jl")
# Stage 7: solid mechanics, FSI, function objects.
include("../solid_mechanics/types.jl")
include("../fsi/coupling.jl")
include("../function_objects/types.jl")
# Stage 8: mesh generation + collocated AMR + error indicators.
include("../mesh_generation/octree.jl")
include("../amr_collocated/adapt.jl")
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
include("../linear_solvers/abstract_operator.jl")
include("../linear_solvers/matrix_free.jl")
include("../linear_solvers/preconditioners.jl")
include("../linear_solvers/solver_config.jl")
include("../units/units.jl")

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

# Conjugate Heat Transfer & Buoyancy (Phase 3)
# Depends on Phase 0 operators + Phase 1 incompressible + Phase 2a turbulence.
include("../thermal/types.jl")
include("../thermal/energy_equation.jl")
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
include("../multiphase/solvers.jl")

# Radiation (Phase 9)
# Depends on Phase 0 operators + Phase 3 thermal.
include("../radiation/types.jl")
include("../radiation/p1.jl")
include("../radiation/solvers.jl")
include("../radiation/fvdom.jl")

# Combustion & Species Transport (Phase 8)
# Depends on Phase 0 operators + Phase 1 incompressible + Phase 2a turbulence + Phase 3 thermal.
include("../combustion/types.jl")
include("../combustion/species_transport.jl")
include("../combustion/edm.jl")
include("../combustion/solvers.jl")
include("../combustion/edc.jl")
include("../combustion/arrhenius.jl")

# Lagrangian DPM (Phase 11)
include("../lagrangian/drag_models.jl")
include("../lagrangian/heat_transfer.jl")
include("../lagrangian/two_way_coupling.jl")
include("../lagrangian/particle_solver.jl")
include("../lagrangian/spray.jl")
include("../lagrangian/collisions.jl")

# Dynamic/Moving Mesh (Phase 10)
include("../dynamic_mesh/types.jl")
include("../dynamic_mesh/solid_body.jl")
include("../dynamic_mesh/laplacian_motion.jl")
include("../dynamic_mesh/mesh_update.jl")
include("../dynamic_mesh/ale.jl")

# MPI Parallelism stubs (Phase 6)
# Concrete implementations live in ext/FVMMPIExt/.
include("../parallel/stubs.jl")
include("../parallel/rcb_partitioner.jl")
include("../parallel/local_mesh.jl")

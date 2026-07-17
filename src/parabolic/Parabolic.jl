# ============================================================
# Parabolic — the cell-vertex parabolic/elliptic solver family
# ============================================================
#
# Owns the FVMProblem/FVMSystem/SteadyFVMProblem problem types, the
# cell-vertex assembly kernels (equations/), the structured-mesh equation
# models and assembly (models.jl + assembly/), the specific-problem
# templates, and the parabolic SciML bridge. `solve.jl` intentionally stays
# in the flat parent until the sciml/ relocation step (it is bidirectionally
# coupled to core/symbolic_indexing + core/sciml_contract).
module Parabolic

using ..Geometry
using ..VertexConditions
using ..Numerics
# problem.jl / root_utils.jl EXTEND these Geometry generics with unqualified
# definitions — import so they extend rather than shadow (Stage-3 recipe).
import ..Geometry: get_triangle_props, get_volume, _safe_get_triangle_props
# problem.jl extends the 14 conditions accessors (moved from Layer 1).
import ..VertexConditions: get_dudt_fidx, get_neumann_fidx, get_robin_fidx,
    get_dirichlet_fidx, get_constrained_fidx, eval_condition_fnc,
    is_dudt_node, is_neumann_edge, is_robin_edge, is_dirichlet_node,
    is_constrained_edge, has_condition, has_dirichlet_nodes, get_dirichlet_nodes

using SciMLBase: SciMLBase, ODEProblem, ODEFunction, MatrixOperator,
    LinearProblem
using DelaunayTriangulation: DelaunayTriangulation, each_solid_triangle,
    each_solid_vertex, get_adjacent, get_boundary_edge_map, get_point, getxy,
    triangle_vertices
using LinearAlgebra: LinearAlgebra, norm, dot
using SparseArrays: SparseArrays, sparse
using StaticArrays: StaticArrays, SVector
using PreallocationTools: PreallocationTools, get_tmp
using Base.Threads

include("types.jl")
include("models.jl")
include("utils.jl")
include("boundary_conditions.jl")
include("gradients.jl")
include("schemes.jl")
include("compressible_fluxes.jl")
include("turbulence.jl")
include("particles.jl")
include("fsi.jl")
include("kernels.jl")
include("assembly/assembly_1d.jl")
include("assembly/assembly_2d.jl")
include("assembly/assembly_3d.jl")
include("assembly/assembly_cylindrical.jl")
include("assembly/assembly_spherical.jl")
include("assembly/assembly_unstructured.jl")
include("assembly/assembly_curvilinear.jl")
include("assembly/assembly_system.jl")
include("problem.jl")
include("equations/boundary_edge_contributions.jl")
include("equations/control_volumes.jl")
include("equations/dirichlet.jl")
include("equations/individual_flux_contributions.jl")
include("equations/main_equations.jl")
include("equations/shape_functions.jl")
include("equations/source_contributions.jl")
include("equations/triangle_contributions.jl")
include("root_utils.jl")
include("muscl_problem.jl")
include("specific_problems/abstract_templates.jl")
include("specific_problems/diffusion_equation.jl")
include("specific_problems/linear_reaction_diffusion_equations.jl")
include("specific_problems/mean_exit_time.jl")
include("specific_problems/poissons_equation.jl")
include("specific_problems/laplaces_equation.jl")
include("specific_problems/advection_diffusion_equation.jl")
include("specific_problems/anisotropic_diffusion.jl")
include("sciml_bridge.jl")

export
    AbstractProblemPDE, AbstractBoundaryCondition, AbstractInitialCondition, UnsupportedBCError,
    ParabolicDirichlet, ParabolicNeumann, ParabolicRobin, AbstractVariable,
    VariableRole, STATEVAR, Variable, CellField,
    make_cell_field, SimulationState, validate_state, update_field,
    AbstractDiscretization, AbstractSemidiscretization, AbstractFluxCalculator, AbstractReconstruction,
    AbstractEquationModel, AbstractDiffusion, AbstractAdvection, AbstractAdvectionDiffusion,
    Diffusion1D, Diffusion2D, Diffusion3D, VariableDiffusion1D,
    VariableDiffusion2D, VariableDiffusion3D, AnisotropicDiffusion1D, AnisotropicDiffusion2D,
    AnisotropicDiffusion3D, CylindricalDiffusion1D, CylindricalDiffusion2D, VariableCylindricalDiffusion1D,
    VariableCylindricalDiffusion2D, SphericalDiffusion1D, SphericalAdvection1D, SphericalAdvectionDiffusion1D,
    CylindricalAdvection1D, CylindricalAdvection2D, VariableCylindricalAdvection2D, Advection1D,
    Advection2D, Advection3D, VariableAdvection1D, VariableAdvection2D,
    VariableAdvection3D, AdvectionDiffusion1D, AdvectionDiffusion2D, AdvectionDiffusion3D,
    VariableAdvectionDiffusion1D, VariableAdvectionDiffusion2D, VariableAdvectionDiffusion3D, CylindricalAdvectionDiffusion1D,
    CylindricalAdvectionDiffusion2D, VariableCylindricalAdvectionDiffusion2D, AbstractSourceTerm, ConstantSource,
    SpatialSource, FunctionSource, LinearizedSource, evaluate_source,
    AbstractTurbulenceModel, ParabolicKEpsilon, update_turbulent_viscosity!, compute_production_k,
    assemble_k_source, assemble_epsilon_source, parabolic_compute_friction_velocity, update_wall_bcs!,
    ParabolicTurbulentWall, assemble_system, assemble_mass_matrix, assemble_deferred_correction,
    AbstractCoupling, LinearCoupling, assemble_coupled_system, build_linear_coupling_block,
    InterfaceBC, ParabolicPeriodicBC, ParabolicNonlinearDirichlet, ParabolicNonlinearNeumann,
    ParabolicCoupledBC, OutflowBC, reconstruct_gradient_green_gauss_2d, reconstruct_gradient_green_gauss_3d,
    reconstruct_gradient_least_squares_1d, reconstruct_gradient_least_squares_2d, muscl_reconstruction_1d, quick_reconstruction_1d,
    second_order_diffusion_flux_1d, muscl_advection_flux_1d, quick_advection_flux_1d, weno5_reconstruction_1d,
    weno5_advection_flux_1d, weno5_reconstruction_right_biased, muscl_reconstruction_2d, quick_reconstruction_2d,
    muscl_advection_flux_2d, quick_advection_flux_2d, ideal_gas_pressure, parabolic_sound_speed,
    hllc_flux_1d, AbstractParticle, LagrangianParticle, ParticleTracker,
    inject_particles!, find_cell_index, is_point_in_cell, advect_particles!,
    AbstractStructuralModel, SpringMassSystem, update_structure!, deform_mesh!,
    update_mesh_geometry!, compute_fluxes_cpu!, add_entry!, apply_source_term!,
    get_diffusion_coefficient_at_face_2d, TimeDependentDirichlet, TimeDependentNeumann, TimeDependentRobin,
    FVMProblem, FVMSystem, SteadyFVMProblem, compute_flux,
    pl_interpolate, create_muscl_problem, DiffusionEquation, LinearReactionDiffusionEquation,
    MeanExitTimeProblem, PoissonsEquation, LaplacesEquation, AdvectionDiffusionEquation,
    AnisotropicDiffusionEquation, make_rotation_tensor, make_spatially_varying_tensor, parabolic_to_odefunction,
    parabolic_to_linearproblem, AbstractFVMTemplate, fvm_eqs!, _neqs,
    update_dirichlet_nodes!,
    AbstractField, AbstractConfig, AbstractOutputManager,
    get_coordinate_system,
    InvalidFluxError, _get_boundary_flux, apply_dirichlet_conditions!, apply_dudt_conditions!, apply_steady_dirichlet_conditions!, boundary_edge_contributions!, construct_flux_function, eval_all_fncs_in_tuple, eval_condition_fnc, eval_flux_function, eval_source_fnc, fix_missing_vertices!, flatten_tuples, fvm_eqs!, fvm_eqs_single_source_contribution!, fvm_eqs_single_triangle!, get_boundary_cv_components, get_boundary_fluxes, get_conditions, get_constrained_fidx, get_cv_components, get_dirichlet_fidx, get_dirichlet_nodes, get_dudt_fidx, get_flux, get_fluxes, get_neumann_fidx, get_shape_function_coefficients, get_source_contribution, get_triangle_props, get_volume, has_condition, has_dirichlet_nodes, is_constrained_edge, is_dirichlet_node, is_dudt_node, is_neumann_edge, is_system, map_fidx, triangle_contributions!

end # module Parabolic

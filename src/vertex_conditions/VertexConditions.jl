# ============================================================
# VertexConditions — the cell-vertex (FVMGeometry/FVMProblem) conditions engine
# ============================================================
#
# Owns the ConditionType enum, BoundaryConditions/InternalConditions and their
# assembled Conditions/SimpleConditions forms, plus the nonlinear, periodic,
# and coupled boundary-condition extensions. Concrete per-family BC structs
# (hyperbolic, incompressible, parabolic Parabolic*) stay with their families.
#
# The module is named VertexConditions (not Conditions) because it exports a
# `Conditions` struct, which could not live inside a module of the same name.
module VertexConditions

using ..Geometry: FVMGeometry, get_triangle_props, _safe_get_triangle_props
using DelaunayTriangulation: DelaunayTriangulation, get_point, getxy, get_adjacent,
    is_ghost_vertex, num_ghost_vertices, num_segments, num_solid_vertices,
    has_ghost_triangles, has_boundary_nodes, add_ghost_triangles!,
    delete_ghost_triangles!, lock_convex_hull!, unlock_convex_hull!,
    get_ghost_vertex_map, get_boundary_nodes, num_boundary_edges
using SparseArrays: sparse

include("conditions.jl")
include("nonlinear.jl")
include("periodic.jl")
include("coupled.jl")

# Public API (re-exported from the main module)
export Conditions, BoundaryConditions, InternalConditions,
    Neumann, Dudt, Dirichlet, Constrained, Robin,
    NonlinearDirichlet, NonlinearNeumann, NonlinearRobin,
    linearize_bc, compute_boundary_gradient, evaluate_nonlinear_bc,
    PeriodicBC, PeriodicNodeMapping, PeriodicConditions,
    compute_periodic_mapping, apply_periodic_constraints!, has_periodic_conditions,
    CoupledBC, CoupledDirichlet, CoupledNeumann, CoupledRobin,
    CoupledBoundaryConditions, evaluate_coupled_bc, add_coupled_bc!,
    get_coupled_bc, has_coupled_bc, get_target_field

# Internal surface consumed by the still-flat remainder (problem.jl, equations/,
# specific_problems/, hyperbolic BC helpers) — temporary over-export, curated in
# Stage 4.
export SimpleConditions, wrap_functions, eval_fnc_in_het_tuple, eval_condition_fnc, get_f,
    get_dudt_fidx, get_neumann_fidx, get_dirichlet_fidx, get_constrained_fidx,
    get_robin_fidx, is_dudt_node, is_neumann_edge, is_dirichlet_node,
    is_constrained_edge, is_robin_edge, has_condition, has_dirichlet_nodes,
    has_dudt_nodes, has_neumann_edges, has_constrained_edges, has_robin_edges,
    get_dirichlet_nodes, get_dudt_nodes, get_neumann_edges, get_constrained_edges,
    get_robin_edges,
    ConditionType, get_segment_nodes

end # module

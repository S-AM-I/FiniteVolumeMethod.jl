# geometry/parabolic_abstract_types.jl — abstract geometry/mesh types from the
# Simu.jl parabolic migration (split out of parabolic/types.jl in Stage 3a).

# --- Geometry and Mesh Abstract Types ---

"""
    AbstractGeometry

Base type for all geometric representations in the simulation.
"""
abstract type AbstractGeometry end

"""
    AbstractGeometryComponent

Base type for components of a geometry (e.g., nodes, cells, faces).
"""
abstract type AbstractGeometryComponent end

"""
    AbstractParabolicMesh{Dim} <: AbstractFVMesh{Dim}

Abstract representation of a computational mesh for parabolic/elliptic solvers,
re-rooted (v4.0) under the unified mesh hierarchy [`AbstractFVMesh`](@ref).
The bare name `AbstractParabolicMesh` (a `UnionAll`) still matches all
dimensions in signatures.
"""
abstract type AbstractParabolicMesh{Dim} <: AbstractFVMesh{Dim} end

"""
    AbstractNode <: AbstractGeometryComponent

Abstract representation of a point in space (node/vertex).
"""
abstract type AbstractNode <: AbstractGeometryComponent end

"""
    AbstractCell <: AbstractGeometryComponent

Abstract representation of a computational cell (element).
"""
abstract type AbstractCell <: AbstractGeometryComponent end

"""
    AbstractFace <: AbstractGeometryComponent

Abstract representation of an interface between cells or a boundary.
"""
abstract type AbstractFace <: AbstractGeometryComponent end

"""
    CellType

Enum identifying the geometric shape of an unstructured 3D cell.

Instances: `CT_Tetrahedron`, `CT_Hexahedron`, `CT_Prism`, `CT_Pyramid`, `CT_Polyhedron`.
"""
@enum CellType begin
    CT_Tetrahedron   # 4 triangular faces
    CT_Hexahedron    # 6 quadrilateral faces
    CT_Prism         # 2 triangular + 3 quadrilateral faces
    CT_Pyramid       # 1 quadrilateral base + 4 triangular faces
    CT_Polyhedron    # generic polyhedral (fallback)
end

@doc "Tetrahedral cell (4 triangular faces)." CT_Tetrahedron
@doc "Hexahedral cell (6 quadrilateral faces)." CT_Hexahedron
@doc "Prismatic / wedge cell (2 triangular + 3 quadrilateral faces)." CT_Prism
@doc "Pyramidal cell (1 quadrilateral base + 4 triangular faces)." CT_Pyramid
@doc "Generic polyhedral cell (fallback for non-standard shapes)." CT_Polyhedron

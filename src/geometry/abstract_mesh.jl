"""
    AbstractFVMesh{Dim}

Umbrella supertype for every mesh type used by any solver family in this
repository (parabolic vertex-centered, hyperbolic cell-centered, and
collocated unstructured). `Dim` is the spatial dimension.

Concrete hierarchies that subtype this:
- `AbstractMesh{Dim}` — hyperbolic structured / unstructured cell-centered
  (Stage 1d retrofit).
- `AbstractFVMMesh{Dim, T}` — parabolic cell-centered + collocated
  `UnstructuredFVMMesh` (declared in `src/parabolic/mesh/fvm_mesh.jl`,
  retrofitted in Stage 1d).
- `FVMGeometry` — parabolic cell-vertex over a `DelaunayTriangulation`
  (declared in `src/geometry.jl`, retrofitted in Stage 1d; `Dim = 2`).

The umbrella type exists purely to let downstream library code write one
`f(mesh::AbstractFiniteVolumeMesh)` method instead of three overloads.
Stage 1d generic fallbacks (`n_cells`, `n_faces`, `dim_of`) are defined
below; concrete types override as needed.
"""
abstract type AbstractFVMesh{Dim} end

# Retired v3 name for the unified root (kept as an alias; remove in Stage 8).
const AbstractFiniteVolumeMesh = AbstractFVMesh

"""
    AbstractMesh{Dim}

Abstract supertype for all mesh types in the hyperbolic solver framework.
`Dim` is the spatial dimension (1, 2, or 3).
"""
abstract type AbstractMesh{Dim} <: AbstractFVMesh{Dim} end

"""
    ndims_mesh(mesh::AbstractMesh{Dim}) -> Int

Return the spatial dimension of the mesh.
"""
ndims_mesh(::AbstractMesh{Dim}) where {Dim} = Dim

"""
    ncells(mesh::AbstractMesh)

Return the number of cells in the mesh.
"""
function ncells end

"""
    nfaces(mesh::AbstractMesh)

Return the number of internal faces in the mesh.
"""
function nfaces end

"""
    cell_center(mesh::AbstractMesh, i::Int)

Return the coordinates of the center of cell `i`.
"""
function cell_center end

"""
    cell_volume(mesh::AbstractMesh, i::Int)

Return the volume (or area in 2D, length in 1D) of cell `i`.
"""
function cell_volume end

"""
    face_normal(mesh::AbstractMesh, f::Int)

Return the outward-pointing normal vector of face `f` (pointing from owner to neighbor).
"""
function face_normal end

"""
    face_area(mesh::AbstractMesh, f::Int)

Return the area (or length in 2D, 1.0 in 1D) of face `f`.
"""
function face_area end

"""
    face_owner(mesh::AbstractMesh, f::Int)

Return the index of the cell that owns face `f` (the "left" cell).
"""
function face_owner end

"""
    face_neighbor(mesh::AbstractMesh, f::Int)

Return the index of the neighbor cell across face `f` (the "right" cell).
"""
function face_neighbor end

# ── Stage 1d umbrella-interface helpers ─────────────────────────────
#
# Generic accessors defined on the umbrella supertype. Concrete mesh
# types override these as they see fit; the defaults here give a usable
# but error-verbose fallback so that mistakes surface loudly.

"""
    dim_of(mesh::AbstractFVMesh{Dim}) -> Int

Spatial dimension of the mesh. Matches the type parameter `Dim`.
"""
dim_of(::AbstractFVMesh{Dim}) where {Dim} = Dim

"""
    n_cells(mesh::AbstractFVMesh) -> Int

Total number of cells (control volumes) in the mesh. Concrete types must
implement this.
"""
function n_cells(mesh::AbstractFVMesh)
    return error(
        "n_cells(::$(typeof(mesh))) not implemented; ",
        "concrete mesh types must override this generic method."
    )
end

"""
    n_faces(mesh::AbstractFVMesh) -> Int

Total number of faces (both internal and boundary) in the mesh.
"""
function n_faces(mesh::AbstractFVMesh)
    return error(
        "n_faces(::$(typeof(mesh))) not implemented; ",
        "concrete mesh types must override this generic method."
    )
end

# Concrete-mesh overloads are defined in `src/mesh/generic_interface.jl`,
# which is loaded once every concrete mesh type has been declared.

# ── Stage 1d umbrella BC supertype ───────────────────────────────────

"""
    AbstractFVMBoundaryCondition

Umbrella supertype for every boundary condition across this repository's
solver families. Concrete sub-hierarchies:

- `AbstractBoundaryCondition` — parabolic + collocated incompressible
  (matrix-based handlers, see `src/parabolic/boundary_conditions.jl` and
  `src/incompressible/boundary_conditions.jl`).
- `AbstractHyperbolicBC` — hyperbolic ghost-cell BCs (see
  `src/hyperbolic/boundary_conditions_hyp.jl`).

The umbrella exists so downstream library code (tests, solver-agnostic
utilities, introspection tooling) can dispatch on
`bc::AbstractFVMBoundaryCondition` and match either family. There is no
unified `apply_bc!` contract yet — the two hierarchies use different
evaluation models (ghost-cell filling vs. matrix assembly) so they keep
their own apply methods. Stage 1f will revisit whether a unified
`apply_bc!(state, bc, mesh, patch_idx)` can be provided as an adapter
layer.
"""
abstract type AbstractFVMBoundaryCondition end

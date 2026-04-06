# collocated/types.jl — Core types for collocated cell-centered FVM operators
#
# These types provide the OpenFOAM-style field abstractions needed for
# pressure-velocity coupling (SIMPLE/PISO/PIMPLE) on unstructured polyhedral
# meshes.  They extend the existing AbstractField / CellField hierarchy and
# produce SciMLBase-compatible LinearProblem / ODEFunction outputs.

using LinearAlgebra: norm
using SparseArrays: SparseArrays, spzeros, SparseMatrixCSC

# ── Abstract hierarchy ───────────────────────────────────────────────

"""
    AbstractCollocatedField <: AbstractField

Supertype for cell-centered fields on `UnstructuredFVMMesh`.
"""
abstract type AbstractCollocatedField <: AbstractField end

# ── Boundary patch ───────────────────────────────────────────────────

"""
    BoundaryPatch

Named region of boundary faces on an `UnstructuredFVMMesh`.

`face_indices` are 1-based indices into the mesh's face arrays where
`face_cells[2, f] == 0` (i.e. boundary faces).
"""
struct BoundaryPatch
    name::Symbol
    face_indices::Vector{Int}
end

"""
    extract_boundary_patches(mesh::UnstructuredFVMMesh)

Build `BoundaryPatch` vector from `mesh.face_tags`.  Faces with the
same tag are grouped into one patch.  Requires `mesh.face_tags !== nothing`.
"""
function extract_boundary_patches(mesh::UnstructuredFVMMesh)
    mesh.face_tags === nothing && error("mesh.face_tags is nothing; cannot extract patches")
    nf = size(mesh.face_cells, 2)
    groups = Dict{Symbol, Vector{Int}}()
    for f in 1:nf
        mesh.face_cells[2, f] == 0 || continue  # skip internal faces
        tag = mesh.face_tags[f]
        push!(get!(Vector{Int}, groups, tag), f)
    end
    return [BoundaryPatch(name, idxs) for (name, idxs) in sort!(collect(groups); by = first)]
end

# ── Collocated scalar field ──────────────────────────────────────────

"""
    CollocatedScalarField{T} <: AbstractCollocatedField

Cell-centered scalar field with explicit boundary face values.

# Fields
- `name::Symbol` — human-readable identifier (e.g. `:p`, `:T`, `:k`)
- `internal::Vector{T}` — values at cell centers, length `ncells`
- `boundary::Vector{T}` — values at boundary faces, length `n_boundary_faces`
- `boundary_face_indices::Vector{Int}` — mesh face index for each boundary entry
"""
struct CollocatedScalarField{T} <: AbstractCollocatedField
    name::Symbol
    internal::Vector{T}
    boundary::Vector{T}
    boundary_face_indices::Vector{Int}
end

"""
    CollocatedScalarField(name, mesh::UnstructuredFVMMesh{Dim, T}; value = zero(T))

Construct a zero-initialized scalar field on `mesh`.
"""
function CollocatedScalarField(
        name::Symbol, mesh::UnstructuredFVMMesh{Dim, T}; value = zero(T),
    ) where {Dim, T}
    ncells = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    bface_idxs = [f for f in 1:nf if mesh.face_cells[2, f] == 0]
    internal = fill(value, ncells)
    boundary = fill(value, length(bface_idxs))
    return CollocatedScalarField{T}(name, internal, boundary, bface_idxs)
end

"""Number of interior cells."""
ncells(field::CollocatedScalarField) = length(field.internal)

"""Number of boundary faces."""
n_boundary_faces(field::CollocatedScalarField) = length(field.boundary)

# ── Collocated vector field ──────────────────────────────────────────

"""
    CollocatedVectorField{Dim, T} <: AbstractCollocatedField

Cell-centered vector field stored as `Vector{SVector{Dim, T}}`.

# Fields
- `name::Symbol`
- `internal::Vector{SVector{Dim, T}}` — cell-center values, length `ncells`
- `boundary::Vector{SVector{Dim, T}}` — boundary face values
- `boundary_face_indices::Vector{Int}`
"""
struct CollocatedVectorField{Dim, T} <: AbstractCollocatedField
    name::Symbol
    internal::Vector{SVector{Dim, T}}
    boundary::Vector{SVector{Dim, T}}
    boundary_face_indices::Vector{Int}
end

"""
    CollocatedVectorField(name, mesh::UnstructuredFVMMesh{Dim, T}; value = zero(SVector{Dim, T}))

Construct a zero-initialized vector field on `mesh`.
"""
function CollocatedVectorField(
        name::Symbol, mesh::UnstructuredFVMMesh{Dim, T};
        value = zero(SVector{Dim, T}),
    ) where {Dim, T}
    ncells_val = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    bface_idxs = [f for f in 1:nf if mesh.face_cells[2, f] == 0]
    internal = fill(value, ncells_val)
    boundary = fill(value, length(bface_idxs))
    return CollocatedVectorField{Dim, T}(name, internal, boundary, bface_idxs)
end

ncells(field::CollocatedVectorField) = length(field.internal)
n_boundary_faces(field::CollocatedVectorField) = length(field.boundary)

# ── Face flux field ──────────────────────────────────────────────────

"""
    FaceFluxField{T}

Scalar face-normal flux field.  Stores one value per mesh face
(both internal and boundary).  Positive flux is in the direction
of `face_normals[:, f]`, i.e. from owner to neighbour.

Used for the volumetric flux `phi = U_f . S_f` in the incompressible
solver and for any advective transport operator.
"""
struct FaceFluxField{T}
    name::Symbol
    values::Vector{T}
end

"""
    FaceFluxField(name, mesh::UnstructuredFVMMesh{Dim, T}; value = zero(T))

Construct a zero-initialized face flux field.
"""
function FaceFluxField(
        name::Symbol, mesh::UnstructuredFVMMesh{Dim, T}; value = zero(T),
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    return FaceFluxField{T}(name, fill(value, nf))
end

nfaces(field::FaceFluxField) = length(field.values)

# ── Assembled equation ───────────────────────────────────────────────

"""
    CollocatedEquation{T}

Assembled linear system `A * x = b` for a single scalar equation
on `ncells` unknowns.  Operators (Laplacian, divergence, ddt) add
their contributions into `A` and `b`, then the equation is solved
via `LinearProblem(eq.A, eq.b)` from SciMLBase.

# Fields
- `A::SparseMatrixCSC{T, Int}` — system matrix
- `b::Vector{T}` — right-hand side
- `source::Vector{T}` — explicit source terms (added to `b` before solve)
"""
mutable struct CollocatedEquation{T}
    A::SparseMatrixCSC{T, Int}
    b::Vector{T}
    source::Vector{T}
end

"""
    CollocatedEquation(mesh::UnstructuredFVMMesh{Dim, T})

Construct an empty equation (zero matrix + zero RHS) sized for `mesh`.
"""
function CollocatedEquation(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}
    nc = length(mesh.cell_volumes)
    A = spzeros(T, nc, nc)
    b = zeros(T, nc)
    source = zeros(T, nc)
    return CollocatedEquation{T}(A, b, source)
end

"""
    reset!(eq::CollocatedEquation)

Zero all matrix and vector entries.  Re-uses allocated storage.
"""
function reset!(eq::CollocatedEquation{T}) where {T}
    fill!(eq.A.nzval, zero(T))
    fill!(eq.b, zero(T))
    fill!(eq.source, zero(T))
    return nothing
end

"""
    to_linear_problem(eq::CollocatedEquation)

Convert to `SciMLBase.LinearProblem(A, b + source)`.  The returned
problem can be solved with any `LinearSolve.jl` algorithm.
"""
function to_linear_problem(eq::CollocatedEquation)
    rhs = eq.b .+ eq.source
    return LinearProblem(eq.A, rhs)
end

# ── Mesh helpers ─────────────────────────────────────────────────────

"""
    is_internal_face(mesh::UnstructuredFVMMesh, f::Int) -> Bool

Return `true` if face `f` connects two cells (i.e. is not a boundary).
"""
is_internal_face(mesh::UnstructuredFVMMesh, f::Int) = mesh.face_cells[2, f] != 0

"""
    owner(mesh::UnstructuredFVMMesh, f::Int) -> Int

Cell index of the face owner.
"""
owner(mesh::UnstructuredFVMMesh, f::Int) = mesh.face_cells[1, f]

"""
    neighbour(mesh::UnstructuredFVMMesh, f::Int) -> Int

Cell index of the face neighbour (0 for boundary faces).
"""
neighbour(mesh::UnstructuredFVMMesh, f::Int) = mesh.face_cells[2, f]

"""
    face_normal_area(mesh::UnstructuredFVMMesh{Dim}, f::Int) -> SVector{Dim}

Outward-pointing face area vector `S_f = A_f * n_f` (area × unit normal).
"""
function face_normal_area(mesh::UnstructuredFVMMesh{Dim}, f::Int) where {Dim}
    A_f = mesh.face_areas[f]
    n = SVector{Dim}(ntuple(d -> mesh.face_normals[d, f], Val(Dim)))
    return A_f * n
end

"""
    cell_center(mesh::UnstructuredFVMMesh{Dim}, c::Int) -> SVector{Dim}

Position vector of cell center `c`.
"""
function cell_center(mesh::UnstructuredFVMMesh{Dim}, c::Int) where {Dim}
    return SVector{Dim}(ntuple(d -> mesh.cell_centers[d, c], Val(Dim)))
end

"""
    face_center(mesh::UnstructuredFVMMesh{Dim}, f::Int) -> SVector{Dim}

Position vector of face center `f`.
"""
function face_center(mesh::UnstructuredFVMMesh{Dim}, f::Int) where {Dim}
    return SVector{Dim}(ntuple(d -> mesh.face_centers[d, f], Val(Dim)))
end

"""
    owner_neighbour_distance(mesh::UnstructuredFVMMesh{Dim}, f::Int) -> Tuple{SVector{Dim}, T}

Return `(d_vec, |d_vec|)` where `d_vec = x_N - x_P` for internal face `f`.
"""
function owner_neighbour_distance(mesh::UnstructuredFVMMesh{Dim, T}, f::Int) where {Dim, T}
    c_P = cell_center(mesh, owner(mesh, f))
    c_N = cell_center(mesh, neighbour(mesh, f))
    d = c_N - c_P
    return d, norm(d)
end

"""
    face_weight(mesh::UnstructuredFVMMesh{Dim, T}, f::Int) -> T

Linear interpolation weight for the owner cell at face `f`:
`w = |x_f - x_N| / |x_P - x_N|` so that `φ_f ≈ w φ_P + (1-w) φ_N`.
"""
function face_weight(mesh::UnstructuredFVMMesh{Dim, T}, f::Int) where {Dim, T}
    c_P = cell_center(mesh, owner(mesh, f))
    c_N = cell_center(mesh, neighbour(mesh, f))
    x_f = face_center(mesh, f)
    d_fN = norm(x_f - c_N)
    d_PN = norm(c_P - c_N)
    return d_PN > zero(T) ? d_fN / d_PN : one(T) / 2
end

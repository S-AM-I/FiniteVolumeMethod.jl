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

# ── Sparsity pattern ─────────────────────────────────────────────────

"""
    SparsityPattern

Pre-computed `nzval` index tables for a `CollocatedEquation`'s matrix `A`.
Built once from the mesh's `face_cells` connectivity so every `assemble_*!`
operation can write into `A.nzval[idx]` in O(1) rather than going through
Julia's sparse `setindex!` (which does a binary search on the column and,
if the entry is new, grows the arrays).

# Fields
- `diag_idx::Vector{Int}` — `nzval` index of `A[c, c]` for each cell `c`.
- `offdiag_PN::Vector{Int}` — `nzval` index of `A[P, N]` for each internal
  face `f` (P = owner, N = neighbour). `0` for boundary faces.
- `offdiag_NP::Vector{Int}` — `nzval` index of `A[N, P]` for each internal
  face `f`. `0` for boundary faces.

Read-only after construction; `reset!` and subsequent assemblies only touch
`A.nzval` values, never structure.
"""
struct SparsityPattern
    diag_idx::Vector{Int}
    offdiag_PN::Vector{Int}
    offdiag_NP::Vector{Int}
end

"""
    build_collocated_sparsity(mesh::UnstructuredFVMMesh{Dim, T}) -> (A, pattern)

Build the empty matrix `A` (with all cell-neighbour structural entries
already present) and a `SparsityPattern` of `nzval` indices. After this,
an assembly kernel can write `A.nzval[pattern.diag_idx[c]] += …` etc.
without any structural changes to `A`.
"""
function build_collocated_sparsity(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Phase 1: compute how many nonzeros each column has.
    # Columns in CSC correspond to the "input" index j of A[i, j]. For our
    # stencil each cell has one diagonal entry plus one off-diagonal per
    # internal face touching it. So each column j (cell j) has 1 + degree(j)
    # structural nonzeros.
    nnz_per_col = Vector{Int}(undef, nc)
    fill!(nnz_per_col, 1)  # diagonal contribution
    for f in 1:nf
        if mesh.face_cells[2, f] != 0
            P = mesh.face_cells[1, f]
            N = mesh.face_cells[2, f]
            # A[P, N] lives in column N; A[N, P] lives in column P.
            nnz_per_col[N] += 1
            nnz_per_col[P] += 1
        end
    end

    total_nnz = sum(nnz_per_col)
    colptr = Vector{Int}(undef, nc + 1)
    colptr[1] = 1
    for j in 1:nc
        colptr[j + 1] = colptr[j] + nnz_per_col[j]
    end

    rowval = Vector{Int}(undef, total_nnz)
    nzval = zeros(T, total_nnz)

    # Phase 2: for each column j, collect the row indices that touch j.
    # We know: row j itself (diagonal) plus every owner P with neighbour j
    # plus every neighbour N of owner j.
    col_rows = [Int[] for _ in 1:nc]
    for j in 1:nc
        push!(col_rows[j], j)  # diagonal
    end
    for f in 1:nf
        if mesh.face_cells[2, f] != 0
            P = mesh.face_cells[1, f]
            N = mesh.face_cells[2, f]
            # A[P, N] → row P in column N
            push!(col_rows[N], P)
            # A[N, P] → row N in column P
            push!(col_rows[P], N)
        end
    end

    # Sort each column's rows and fill rowval
    for j in 1:nc
        sort!(col_rows[j])
        k = colptr[j]
        for r in col_rows[j]
            rowval[k] = r
            k += 1
        end
    end

    A = SparseMatrixCSC{T, Int}(nc, nc, colptr, rowval, nzval)

    # Phase 3: build the nzval-index lookup tables.
    function find_nzidx(col::Int, row::Int)
        for k in colptr[col]:(colptr[col + 1] - 1)
            rowval[k] == row && return k
        end
        return 0  # should not happen if structure was built correctly
    end

    diag_idx = Vector{Int}(undef, nc)
    for c in 1:nc
        diag_idx[c] = find_nzidx(c, c)
    end

    offdiag_PN = zeros(Int, nf)
    offdiag_NP = zeros(Int, nf)
    for f in 1:nf
        if mesh.face_cells[2, f] != 0
            P = mesh.face_cells[1, f]
            N = mesh.face_cells[2, f]
            offdiag_PN[f] = find_nzidx(N, P)  # A[P, N] — column N, row P
            offdiag_NP[f] = find_nzidx(P, N)  # A[N, P] — column P, row N
        end
    end

    return A, SparsityPattern(diag_idx, offdiag_PN, offdiag_NP)
end

# ── Assembled equation ───────────────────────────────────────────────

"""
    CollocatedEquation{T}

Assembled linear system `A * x = b` for a single scalar equation
on `ncells` unknowns.  Operators (Laplacian, divergence, ddt) add
their contributions into `A` and `b`, then the equation is solved
via `LinearProblem(eq.A, eq.b)` from SciMLBase.

# Fields
- `A::SparseMatrixCSC{T, Int}` — system matrix, structure built eagerly
  from the mesh's cell-neighbour connectivity.
- `b::Vector{T}` — right-hand side
- `source::Vector{T}` — explicit source terms (added to `b` before solve)
- `pattern::SparsityPattern` — nzval index lookup tables for fast
  assembly via `add_diag!` / `add_offdiag!`.
"""
mutable struct CollocatedEquation{T}
    A::SparseMatrixCSC{T, Int}
    b::Vector{T}
    source::Vector{T}
    pattern::SparsityPattern
end

"""
    CollocatedEquation(mesh::UnstructuredFVMMesh{Dim, T})

Construct an empty equation (zero matrix + zero RHS) sized for `mesh`.
The sparsity structure of `A` is built eagerly from the mesh's
cell-neighbour connectivity, so subsequent `assemble_*!` calls never
modify `A`'s structure — they only write into `A.nzval`.
"""
function CollocatedEquation(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}
    nc = length(mesh.cell_volumes)
    A, pattern = build_collocated_sparsity(mesh)
    b = zeros(T, nc)
    source = zeros(T, nc)
    return CollocatedEquation{T}(A, b, source, pattern)
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

# ── Fast-path assembly helpers (use these in new assembly code) ─────

"""
    add_diag!(eq::CollocatedEquation, c::Int, coeff)

Accumulate `coeff` into the diagonal entry `A[c, c]` via the pre-computed
`nzval` index. O(1) regardless of matrix size.
"""
@inline function add_diag!(eq::CollocatedEquation{T}, c::Int, coeff) where {T}
    eq.A.nzval[eq.pattern.diag_idx[c]] += T(coeff)
    return nothing
end

"""
    add_offdiag_PN!(eq, f, coeff)

Accumulate `coeff` into `A[P, N]` for internal face `f` (P = owner,
N = neighbour). O(1).
"""
@inline function add_offdiag_PN!(eq::CollocatedEquation{T}, f::Int, coeff) where {T}
    eq.A.nzval[eq.pattern.offdiag_PN[f]] += T(coeff)
    return nothing
end

"""
    add_offdiag_NP!(eq, f, coeff)

Accumulate `coeff` into `A[N, P]` for internal face `f`. O(1).
"""
@inline function add_offdiag_NP!(eq::CollocatedEquation{T}, f::Int, coeff) where {T}
    eq.A.nzval[eq.pattern.offdiag_NP[f]] += T(coeff)
    return nothing
end

"""
    add_face_coeffs!(eq, f, c_PP, c_PN, c_NP, c_NN)

Accumulate the four face-contributions for internal face `f` into
`A[P,P] += c_PP`, `A[P,N] += c_PN`, `A[N,P] += c_NP`, `A[N,N] += c_NN`.
Uses the pre-computed `nzval` indices; convenient for Laplacian and
divergence kernels that always touch the same four entries per face.
"""
@inline function add_face_coeffs!(
        eq::CollocatedEquation{T}, f::Int,
        c_PP, c_PN, c_NP, c_NN,
    ) where {T}
    nz = eq.A.nzval
    pat = eq.pattern
    nz[pat.diag_idx[_owner_of_face(eq, f)]] += T(c_PP)
    nz[pat.offdiag_PN[f]] += T(c_PN)
    nz[pat.offdiag_NP[f]] += T(c_NP)
    nz[pat.diag_idx[_neighbour_of_face(eq, f)]] += T(c_NN)
    return nothing
end

# Helpers — cached owner/neighbour lookups bound to the equation would
# cost per-call work; instead we ask assemblers to pass owner/neighbour
# directly. Keep these as placeholders for now (unused) until we decide
# to cache mesh refs in the equation itself.
_owner_of_face(::CollocatedEquation, ::Int) = error("internal: owner lookup not cached on equation; pass P, N to add_face_coeffs_PN!")
_neighbour_of_face(::CollocatedEquation, ::Int) = error("internal: neighbour lookup not cached on equation")

"""
    add_face_coeffs_PN!(eq, f, P, N, c_PP, c_PN, c_NP, c_NN)

Accumulate the four face-contributions when the caller already has `P`
and `N` in scope. Preferred for hot loops.
"""
@inline function add_face_coeffs_PN!(
        eq::CollocatedEquation{T}, f::Int, P::Int, N::Int,
        c_PP, c_PN, c_NP, c_NN,
    ) where {T}
    nz = eq.A.nzval
    pat = eq.pattern
    nz[pat.diag_idx[P]] += T(c_PP)
    nz[pat.offdiag_PN[f]] += T(c_PN)
    nz[pat.offdiag_NP[f]] += T(c_NP)
    nz[pat.diag_idx[N]] += T(c_NN)
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

# ── Block-coupled equation (Stage 1c) ────────────────────────────────

"""
    BlockCollocatedEquation{T, NBlocks}

Block-coupled linear system `A_block * x = b` for `NBlocks` scalar-per-cell
unknowns assembled together (e.g. Eulerian two-fluid α₁/α₂, coupled
momentum-energy, or RANS k/ε solved monolithically).

The block layout is cell-major with `NBlocks` unknowns per cell:
- `A_block::SparseMatrixCSC{T, Int}` has shape `(NBlocks * ncells, NBlocks * ncells)`.
- Row `(c - 1) * NBlocks + b` corresponds to block-unknown `b` at cell `c`.
- Sparsity structure copies the single-block `SparsityPattern` into each
  `(b_row, b_col)` block, giving `NBlocks²` structural entries per
  `(P, N)` face pair and `NBlocks²` per diagonal.

Reuses the same build-once-pre-compute-nzval-indices strategy as the
scalar `CollocatedEquation`. Intended to be wired into Eulerian two-fluid
(Stage 6e) and coupled momentum-energy for compressible flows (Stage 3).

# Fields
- `A::SparseMatrixCSC{T, Int}` — block system matrix.
- `b::Vector{T}` — right-hand side of length `NBlocks * ncells`.
- `source::Vector{T}` — explicit source terms (added to `b` before solve).
- `pattern::BlockSparsityPattern{NBlocks}` — nzval lookup tables.
"""
struct BlockSparsityPattern{NBlocks}
    # For each (block_row, block_col) in 1:NBlocks × 1:NBlocks, we store
    # the scalar-pattern-style tables shifted into the block position.
    diag_idx::Array{Int, 3}      # (NBlocks, NBlocks, ncells)
    offdiag_PN::Array{Int, 3}    # (NBlocks, NBlocks, nfaces)
    offdiag_NP::Array{Int, 3}    # (NBlocks, NBlocks, nfaces)
end

mutable struct BlockCollocatedEquation{T, NBlocks}
    A::SparseMatrixCSC{T, Int}
    b::Vector{T}
    source::Vector{T}
    pattern::BlockSparsityPattern{NBlocks}
end

"""
    build_block_collocated_sparsity(mesh, ::Val{NBlocks}) -> (A, pattern)

Build the block CSC skeleton and nzval lookup tables for a block-coupled
equation with `NBlocks` unknowns per cell. The block layout is cell-major:
cell `c`'s `NBlocks` unknowns occupy rows `(c-1)*NBlocks+1 : c*NBlocks`.
"""
function build_block_collocated_sparsity(
        mesh::UnstructuredFVMMesh{Dim, T}, ::Val{NBlocks},
    ) where {Dim, T, NBlocks}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    N = NBlocks * nc  # total system size

    # Column degree: each block column j (cell_col, block_col) has
    # NBlocks entries for the cell diagonal + NBlocks * degree(cell_col)
    # for the neighbour contributions.
    nnz_per_col = zeros(Int, N)
    for c in 1:nc
        for bc in 1:NBlocks
            col = (c - 1) * NBlocks + bc
            nnz_per_col[col] += NBlocks  # diagonal block contributes NBlocks rows
        end
    end
    for f in 1:nf
        if mesh.face_cells[2, f] != 0
            P = mesh.face_cells[1, f]
            Nc = mesh.face_cells[2, f]
            for bc in 1:NBlocks
                col_P = (P - 1) * NBlocks + bc
                col_N = (Nc - 1) * NBlocks + bc
                nnz_per_col[col_N] += NBlocks  # block A[P, N]
                nnz_per_col[col_P] += NBlocks  # block A[N, P]
            end
        end
    end

    total_nnz = sum(nnz_per_col)
    colptr = Vector{Int}(undef, N + 1)
    colptr[1] = 1
    for j in 1:N
        colptr[j + 1] = colptr[j] + nnz_per_col[j]
    end

    rowval = Vector{Int}(undef, total_nnz)
    nzval = zeros(T, total_nnz)

    # Collect rows per column, then sort
    col_rows = [Int[] for _ in 1:N]
    for c in 1:nc
        for bc in 1:NBlocks
            col = (c - 1) * NBlocks + bc
            for br in 1:NBlocks
                row = (c - 1) * NBlocks + br
                push!(col_rows[col], row)
            end
        end
    end
    for f in 1:nf
        if mesh.face_cells[2, f] != 0
            P = mesh.face_cells[1, f]
            Nc = mesh.face_cells[2, f]
            for bc in 1:NBlocks
                col_N = (Nc - 1) * NBlocks + bc
                col_P = (P - 1) * NBlocks + bc
                for br in 1:NBlocks
                    row_P = (P - 1) * NBlocks + br
                    row_N = (Nc - 1) * NBlocks + br
                    push!(col_rows[col_N], row_P)  # A[P, N]
                    push!(col_rows[col_P], row_N)  # A[N, P]
                end
            end
        end
    end
    for j in 1:N
        sort!(col_rows[j])
        k = colptr[j]
        for r in col_rows[j]
            rowval[k] = r
            k += 1
        end
    end

    A = SparseMatrixCSC{T, Int}(N, N, colptr, rowval, nzval)

    function find_nzidx(col::Int, row::Int)
        for k in colptr[col]:(colptr[col + 1] - 1)
            rowval[k] == row && return k
        end
        return 0
    end

    diag_idx = Array{Int}(undef, NBlocks, NBlocks, nc)
    for c in 1:nc, bc in 1:NBlocks, br in 1:NBlocks
        row = (c - 1) * NBlocks + br
        col = (c - 1) * NBlocks + bc
        diag_idx[br, bc, c] = find_nzidx(col, row)
    end

    offdiag_PN = zeros(Int, NBlocks, NBlocks, nf)
    offdiag_NP = zeros(Int, NBlocks, NBlocks, nf)
    for f in 1:nf
        if mesh.face_cells[2, f] != 0
            P = mesh.face_cells[1, f]
            Nc = mesh.face_cells[2, f]
            for bc in 1:NBlocks, br in 1:NBlocks
                row_P = (P - 1) * NBlocks + br
                col_N = (Nc - 1) * NBlocks + bc
                row_N = (Nc - 1) * NBlocks + br
                col_P = (P - 1) * NBlocks + bc
                offdiag_PN[br, bc, f] = find_nzidx(col_N, row_P)
                offdiag_NP[br, bc, f] = find_nzidx(col_P, row_N)
            end
        end
    end

    return A, BlockSparsityPattern{NBlocks}(diag_idx, offdiag_PN, offdiag_NP)
end

"""
    BlockCollocatedEquation(mesh, ::Val{NBlocks})

Construct an empty block-coupled equation for `NBlocks` unknowns per cell.
The sparsity structure is built eagerly so subsequent assembly writes
through `A.nzval[pattern.diag_idx[br, bc, c]]` without structural changes.
"""
function BlockCollocatedEquation(
        mesh::UnstructuredFVMMesh{Dim, T}, ::Val{NBlocks},
    ) where {Dim, T, NBlocks}
    nc = length(mesh.cell_volumes)
    A, pattern = build_block_collocated_sparsity(mesh, Val(NBlocks))
    b = zeros(T, NBlocks * nc)
    source = zeros(T, NBlocks * nc)
    return BlockCollocatedEquation{T, NBlocks}(A, b, source, pattern)
end

function reset!(eq::BlockCollocatedEquation{T}) where {T}
    fill!(eq.A.nzval, zero(T))
    fill!(eq.b, zero(T))
    fill!(eq.source, zero(T))
    return nothing
end

"""
    add_block_diag!(eq::BlockCollocatedEquation, c, br, bc, coeff)

Accumulate `coeff` into the `(br, bc)` entry of the diagonal block at
cell `c`. O(1).
"""
@inline function add_block_diag!(
        eq::BlockCollocatedEquation{T, NBlocks}, c::Int, br::Int, bc::Int, coeff,
    ) where {T, NBlocks}
    eq.A.nzval[eq.pattern.diag_idx[br, bc, c]] += T(coeff)
    return nothing
end

"""
    add_block_offdiag_PN!(eq, f, br, bc, coeff)

Accumulate `coeff` into the `(br, bc)` entry of the off-diagonal block at
A[P_block, N_block] for internal face `f`. O(1).
"""
@inline function add_block_offdiag_PN!(
        eq::BlockCollocatedEquation{T, NBlocks}, f::Int, br::Int, bc::Int, coeff,
    ) where {T, NBlocks}
    eq.A.nzval[eq.pattern.offdiag_PN[br, bc, f]] += T(coeff)
    return nothing
end

@inline function add_block_offdiag_NP!(
        eq::BlockCollocatedEquation{T, NBlocks}, f::Int, br::Int, bc::Int, coeff,
    ) where {T, NBlocks}
    eq.A.nzval[eq.pattern.offdiag_NP[br, bc, f]] += T(coeff)
    return nothing
end

function to_linear_problem(eq::BlockCollocatedEquation)
    rhs = eq.b .+ eq.source
    return LinearProblem(eq.A, rhs)
end

"""Number of scalar unknowns per cell."""
nblocks(::BlockCollocatedEquation{T, NB}) where {T, NB} = NB

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

# ── Nearest-cell lookup ─────────────────────────────────────────────

"""
    find_nearest_cell(mesh, point) -> Int

Find the cell whose center is nearest to `point` (brute-force search).
Returns `0` if the mesh has no cells.
"""
function find_nearest_cell(
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nc == 0 && return 0
    best_cell = 1
    best_dist = T(Inf)
    for c in 1:nc
        x_c = cell_center(mesh, c)
        d = norm(point - x_c)
        if d < best_dist
            best_dist = d
            best_cell = c
        end
    end
    return best_cell
end

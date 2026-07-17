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

# ── Function-valued Dirichlet primitive ─────────────────────────────

"""
    ParabolicDirichletFunc{F} <: AbstractBoundaryCondition

Spatially-varying Dirichlet primitive for the collocated operators.
`func(x_f::SVector) -> value` is evaluated at each boundary face center
during assembly, so spatially- and time-varying high-level BCs (e.g.
`SpatialVelocityBC`, `CodedFixedValueBC`) enter the matrix/RHS with their
true per-face values instead of a `Dirichlet(0)` placeholder.  Time
dependence is baked into the closure by the expansion step.
"""
struct ParabolicDirichletFunc{F} <: AbstractBoundaryCondition
    func::F
end

# ── Collocated scalar field ──────────────────────────────────────────

"""
    CollocatedScalarField{T, A <: AbstractVector{T}} <: AbstractCollocatedField

Cell-centered scalar field with explicit boundary face values.

Parameterized on the array container type `A` (defaults to `Vector{T}`) so
a future GPU backend can instantiate `CollocatedScalarField{Float32, CuVector{Float32}}`
without changing any downstream method signatures: callsites written as
`::CollocatedScalarField{T}` match any container type via Julia's UnionAll
dispatch (Stage 1g).

# Fields
- `name::Symbol` — human-readable identifier (e.g. `:p`, `:T`, `:k`)
- `internal::A` — values at cell centers, length `ncells`
- `boundary::A` — values at boundary faces, length `n_boundary_faces`
- `boundary_face_indices::Vector{Int}` — mesh face index for each boundary entry
"""
struct CollocatedScalarField{T, A <: AbstractVector{T}} <: AbstractCollocatedField
    name::Symbol
    internal::A
    boundary::A
    boundary_face_indices::Vector{Int}
end

# Preserve the old 1-parameter constructor form by inferring A from
# the supplied arrays. Existing code like
# `CollocatedScalarField{T}(name, internal, boundary, bface_idxs)` continues
# to work.
function CollocatedScalarField{T}(
        name::Symbol, internal::AbstractVector{T}, boundary::AbstractVector{T},
        boundary_face_indices::Vector{Int},
    ) where {T}
    A = typeof(internal)
    typeof(boundary) === A ||
        error("CollocatedScalarField: internal and boundary must have same container type")
    return CollocatedScalarField{T, A}(name, internal, boundary, boundary_face_indices)
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
    return CollocatedScalarField{T, typeof(internal)}(name, internal, boundary, bface_idxs)
end

"""Number of interior cells."""
ncells(field::CollocatedScalarField) = length(field.internal)

"""Number of boundary faces."""
n_boundary_faces(field::CollocatedScalarField) = length(field.boundary)

# ── Collocated vector field ──────────────────────────────────────────

"""
    CollocatedVectorField{Dim, T, A <: AbstractVector{SVector{Dim, T}}} <: AbstractCollocatedField

Cell-centered vector field stored as an abstract-container sequence of
`SVector{Dim, T}`. Parameterised on the container type `A` for future GPU
dispatch (Stage 1g). `CollocatedVectorField{Dim, T}` still matches any
container via Julia's UnionAll dispatch.

# Fields
- `name::Symbol`
- `internal::A` — cell-center values, length `ncells`
- `boundary::A` — boundary face values
- `boundary_face_indices::Vector{Int}`
"""
struct CollocatedVectorField{Dim, T, A <: AbstractVector{<:SVector{Dim, T}}} <: AbstractCollocatedField
    name::Symbol
    internal::A
    boundary::A
    boundary_face_indices::Vector{Int}
end

function CollocatedVectorField{Dim, T}(
        name::Symbol, internal::AbstractVector{<:SVector{Dim, T}},
        boundary::AbstractVector{<:SVector{Dim, T}},
        boundary_face_indices::Vector{Int},
    ) where {Dim, T}
    A = typeof(internal)
    typeof(boundary) === A ||
        error("CollocatedVectorField: internal and boundary must have same container type")
    return CollocatedVectorField{Dim, T, A}(name, internal, boundary, boundary_face_indices)
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
    return CollocatedVectorField{Dim, T, typeof(internal)}(name, internal, boundary, bface_idxs)
end

ncells(field::CollocatedVectorField) = length(field.internal)
n_boundary_faces(field::CollocatedVectorField) = length(field.boundary)

# ── Face flux field ──────────────────────────────────────────────────

"""
    FaceFluxField{T, A <: AbstractVector{T}}

Scalar face-normal flux field.  Stores one value per mesh face
(both internal and boundary).  Positive flux is in the direction
of `face_normals[:, f]`, i.e. from owner to neighbour.

Parameterised on array container `A` (Stage 1g); defaults to `Vector{T}`.
Existing `::FaceFluxField{T}` method signatures match any container via
UnionAll dispatch.

Used for the volumetric flux `phi = U_f . S_f` in the incompressible
solver and for any advective transport operator.
"""
struct FaceFluxField{T, A <: AbstractVector{T}}
    name::Symbol
    values::A
end

function FaceFluxField{T}(name::Symbol, values::AbstractVector{T}) where {T}
    return FaceFluxField{T, typeof(values)}(name, values)
end

"""
    FaceFluxField(name, mesh::UnstructuredFVMMesh{Dim, T}; value = zero(T))

Construct a zero-initialized face flux field.
"""
function FaceFluxField(
        name::Symbol, mesh::UnstructuredFVMMesh{Dim, T}; value = zero(T),
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    values = fill(value, nf)
    return FaceFluxField{T, typeof(values)}(name, values)
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
    build_collocated_sparsity(
        mesh::UnstructuredFVMMesh{Dim, T};
        extra_cell_pairs = Tuple{Int, Int}[],
    ) -> (A, pattern)

Build the empty matrix `A` (with all cell-neighbour structural entries
already present) and a `SparsityPattern` of `nzval` indices. After this,
an assembly kernel can write `A.nzval[pattern.diag_idx[c]] += …` etc.
without any structural changes to `A`.

`extra_cell_pairs` pre-allocates additional symmetric couplings
`A[c1, c2]` / `A[c2, c1]` beyond the face-neighbour stencil — used for
cyclic (periodic) boundary coupling so that `apply_cyclic_bc!` never
inserts new structural entries (which would invalidate the pre-computed
`nzval` index tables of a reused equation).
"""
function build_collocated_sparsity(
        mesh::UnstructuredFVMMesh{Dim, T};
        extra_cell_pairs::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[],
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Phase 2 (moved first): for each column j, collect the row indices
    # that touch j.  We know: row j itself (diagonal) plus every owner P
    # with neighbour j plus every neighbour N of owner j, plus any extra
    # cell-pair couplings.
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
    for (c1, c2) in extra_cell_pairs
        c1 == c2 && continue
        push!(col_rows[c2], c1)  # A[c1, c2]
        push!(col_rows[c1], c2)  # A[c2, c1]
    end

    # Sort each column's rows, deduplicate, and build colptr/rowval.
    for j in 1:nc
        sort!(col_rows[j])
        unique!(col_rows[j])
    end

    colptr = Vector{Int}(undef, nc + 1)
    colptr[1] = 1
    for j in 1:nc
        colptr[j + 1] = colptr[j] + length(col_rows[j])
    end
    total_nnz = colptr[nc + 1] - 1

    rowval = Vector{Int}(undef, total_nnz)
    nzval = zeros(T, total_nnz)
    for j in 1:nc
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
    CollocatedEquation(mesh::UnstructuredFVMMesh{Dim, T}; extra_cell_pairs = Tuple{Int, Int}[])

Construct an empty equation (zero matrix + zero RHS) sized for `mesh`.
The sparsity structure of `A` is built eagerly from the mesh's
cell-neighbour connectivity, so subsequent `assemble_*!` calls never
modify `A`'s structure — they only write into `A.nzval`.

Pass `extra_cell_pairs` (e.g. cyclic-partner owner-cell pairs) to
pre-allocate cross-boundary couplings so the equation can be safely
reused across iterations with `reset!` even when cyclic BCs are applied.
"""
function CollocatedEquation(
        mesh::UnstructuredFVMMesh{Dim, T};
        extra_cell_pairs::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[],
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    A, pattern = build_collocated_sparsity(mesh; extra_cell_pairs = extra_cell_pairs)
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

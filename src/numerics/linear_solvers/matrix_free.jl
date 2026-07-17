# linear_solvers/matrix_free.jl — Matrix-free AbstractLinearOperator (Stage 9e)
#
# Complements the Stage 1h SparseMatrixLinearOperator with a
# user-closure-based matvec. At industrial cell counts (10^7+) the
# assembled CSC doesn't fit memory, and a matrix-free Krylov solve is
# the only viable path. This operator plugs into the existing
# AbstractLinearOperator interface so every downstream solver that
# accepts `::AbstractLinearOperator` works unchanged.
#
# The user supplies a closure `matvec!(y, x)` implementing y := A·x.
# Optional companions:
#   - `diag_estimator(op)` — return an approximate diagonal for Jacobi
#     preconditioning.
#   - `transpose_matvec!(y, x)` — for left-preconditioned GMRES or
#     adjoint solvers.

import LinearAlgebra

"""
    MatrixFreeLinearOperator{T, F, Ft, D} <: AbstractLinearOperator{T}

Matrix-free linear operator backed by a user closure. `matvec!(y, x)`
fills `y` with `A·x`. Optional `transpose_matvec!` and `diagonal`
fields enable left-preconditioned and adjoint solves.

# Fields
- `n::Int`, `m::Int` — operator dimensions (A: n×m).
- `matvec!::F` — closure `(y, x) -> nothing` implementing y := A·x.
- `transpose_matvec!::Ft` — optional closure for y := A^T · x (or
  `nothing` if not available).
- `diagonal::D` — optional pre-computed or estimated diagonal of A
  (for Jacobi preconditioning), or `nothing`.
"""
struct MatrixFreeLinearOperator{T, F, Ft, D} <: AbstractLinearOperator{T}
    n::Int
    m::Int
    matvec!::F
    transpose_matvec!::Ft
    diagonal::D
end

function MatrixFreeLinearOperator{T}(
        n::Int, m::Int, matvec!;
        transpose_matvec! = nothing, diagonal = nothing,
    ) where {T}
    return MatrixFreeLinearOperator{T, typeof(matvec!), typeof(transpose_matvec!), typeof(diagonal)}(
        n, m, matvec!, transpose_matvec!, diagonal,
    )
end

# Square operator convenience
function MatrixFreeLinearOperator{T}(n::Int, matvec!; kwargs...) where {T}
    return MatrixFreeLinearOperator{T}(n, n, matvec!; kwargs...)
end

Base.size(op::MatrixFreeLinearOperator) = (op.n, op.m)
Base.size(op::MatrixFreeLinearOperator, d::Int) = d == 1 ? op.n : op.m

function LinearAlgebra.mul!(
        y::AbstractVector, op::MatrixFreeLinearOperator, x::AbstractVector,
    )
    length(x) == op.m ||
        error("MatrixFree mul!: x length $(length(x)) ≠ m = $(op.m)")
    length(y) == op.n ||
        error("MatrixFree mul!: y length $(length(y)) ≠ n = $(op.n)")
    op.matvec!(y, x)
    return y
end

# underlying_matrix inherits from AbstractLinearOperator and throws
# MatrixFreeError (defined in abstract_operator.jl) because no explicit
# matrix exists.

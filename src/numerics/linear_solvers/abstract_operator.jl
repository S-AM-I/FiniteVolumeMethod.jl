# linear_solvers/abstract_operator.jl — Stage 1h: AbstractLinearOperator
#
# Interface-only abstraction that wraps the linear-system matrix presented
# to the pressure-Poisson / momentum / turbulence-closure Krylov solvers.
# Today every assembled `CollocatedEquation` carries a concrete
# `SparseMatrixCSC{T, Int}`. Stage 9e (matrix-free operators — essential at
# 10^7+ cell counts where assembling the CSC doesn't fit memory) will plug
# in a `MatrixFreeLinearOperator` with the same abstract parent.
#
# This file adds *only* the abstract type and the `SparseMatrixLinearOperator`
# wrapper that reproduces today's behavior. No dispatch site is yet forced
# to go through the wrapper; doing that would be a wide churn without a
# concrete caller yet. When Stage 9e lands, `_dispatch_solve` and
# `to_linear_problem(eq::CollocatedEquation)` can be taught to walk through
# `AbstractLinearOperator` before constructing the SciMLBase `LinearProblem`.

import LinearAlgebra
using SparseArrays: SparseMatrixCSC

"""
    AbstractLinearOperator{T}

Abstract supertype for any linear operator `Ax = b` used by the collocated
solver family's Krylov / direct solvers. `T` is the scalar element type.

Concrete subtypes must implement at least:

- `Base.size(op)` — `(M, N)` returning the operator's dimensions.
- `LinearAlgebra.mul!(y, op, x)` — compute `y = op * x` in place.
- `underlying_matrix(op)` — return a matrix representation if one exists,
  or throw `MatrixFreeError` if the operator is matrix-free.

Optional (for Krylov compatibility):
- `Base.eltype(op)` — defaults to `T`.
- `LinearAlgebra.issymmetric(op)` / `isposdef(op)` — trait overrides.
"""
abstract type AbstractLinearOperator{T} end

Base.eltype(::AbstractLinearOperator{T}) where {T} = T

"""
    MatrixFreeError(op)

Thrown by `underlying_matrix(op)` when `op` is a matrix-free operator
and no explicit matrix exists. Stage 9e matrix-free Krylov paths will
catch and handle this; the current sparse-backed code never sees it.
"""
struct MatrixFreeError <: Exception
    op::Any
end
function Base.showerror(io::IO, e::MatrixFreeError)
    return print(io, "MatrixFreeError: $(typeof(e.op)) is matrix-free; no explicit matrix available.")
end

"""
    SparseMatrixLinearOperator{T, M <: SparseMatrixCSC{T}} <: AbstractLinearOperator{T}

Thin wrapper around a `SparseMatrixCSC` that satisfies the
`AbstractLinearOperator` interface. Delegates `size` / `mul!` /
`underlying_matrix` to the wrapped sparse matrix.

Intended as the identity-preserving path that reproduces today's
`SparseMatrixCSC`-only assembly pipeline once Stage 9e introduces
matrix-free alternatives.
"""
struct SparseMatrixLinearOperator{T, M <: SparseMatrixCSC{T}} <: AbstractLinearOperator{T}
    A::M
end

Base.size(op::SparseMatrixLinearOperator) = size(op.A)
Base.size(op::SparseMatrixLinearOperator, d) = size(op.A, d)

LinearAlgebra.mul!(y::AbstractVector, op::SparseMatrixLinearOperator, x::AbstractVector) =
    LinearAlgebra.mul!(y, op.A, x)

"""
    underlying_matrix(op::AbstractLinearOperator)

Return the matrix representation of `op` if one exists, otherwise throw
`MatrixFreeError(op)`. The sparse-backed path returns the wrapped
`SparseMatrixCSC`; future matrix-free implementations override with their
own behavior.
"""
underlying_matrix(op::SparseMatrixLinearOperator) = op.A
function underlying_matrix(op::AbstractLinearOperator)
    return throw(MatrixFreeError(op))
end

"""
    as_linear_operator(A)

Convenience: wrap `A` in a `SparseMatrixLinearOperator` if it is a
`SparseMatrixCSC`; return it unchanged if it is already an
`AbstractLinearOperator`. Used by downstream solver glue (Stage 9e) to
normalise heterogeneous inputs into the common abstract interface.
"""
as_linear_operator(op::AbstractLinearOperator) = op
as_linear_operator(A::SparseMatrixCSC) = SparseMatrixLinearOperator(A)

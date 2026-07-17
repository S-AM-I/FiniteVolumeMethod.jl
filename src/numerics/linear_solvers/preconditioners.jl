# linear_solvers/preconditioners.jl — Preconditioner construction dispatch
#
# Builds preconditioners from a symbol tag and a sparse matrix.
# Built-in: :none, :diagonal. Extensions provide :amg and :ilu.

using LinearAlgebra: Diagonal, diag
using SparseArrays: SparseMatrixCSC

"""
    build_preconditioner(tag::Symbol, A::SparseMatrixCSC) -> preconditioner or nothing

Construct a preconditioner for sparse matrix `A` based on the tag.

Built-in tags:
- `:none` — no preconditioner (returns `nothing`)
- `:diagonal` — Jacobi preconditioner (`Diagonal(diag(A))`)

Extension-provided tags (require loading the package):
- `:amg` — Algebraic Multigrid (requires `using AlgebraicMultigrid`)
- `:ilu` — Incomplete LU (requires `using IncompleteLU`)
"""
function build_preconditioner(tag::Symbol, A::SparseMatrixCSC)
    tag == :none && return nothing
    tag == :diagonal && return Diagonal(diag(A))
    return _extension_preconditioner(Val(tag), A)
end

"""
    _extension_preconditioner(::Val{tag}, A)

Fallback for extension-provided preconditioners. Warns and returns `nothing`
if the required package is not loaded.

Package extensions override this method for specific tags:
- `FVMAMGExt` overrides `Val{:amg}`
- `FVMILUExt` overrides `Val{:ilu}`
"""
function _extension_preconditioner(::Val{T}, A) where {T}
    @warn "Preconditioner :$T not available. Load the required package:\n" *
        "  :amg → `using AlgebraicMultigrid`\n" *
        "  :ilu → `using IncompleteLU`\n" *
        "Falling back to no preconditioner."
    return nothing
end

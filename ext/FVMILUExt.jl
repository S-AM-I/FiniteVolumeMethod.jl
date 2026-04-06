module FVMILUExt

using FiniteVolumeMethod
using IncompleteLU
using SparseArrays: SparseMatrixCSC

"""
Override the ILU preconditioner dispatch to use incomplete LU
factorization from IncompleteLU.jl.
"""
function FiniteVolumeMethod._extension_preconditioner(
        ::Val{:ilu}, A::SparseMatrixCSC,
    )
    return IncompleteLU.ilu(A; τ = 0.1)
end

end # module

module FVMAMGExt

using FiniteVolumeMethod
using AlgebraicMultigrid
using SparseArrays: SparseMatrixCSC

"""
Override the AMG preconditioner dispatch to use Ruge-Stuben AMG
from AlgebraicMultigrid.jl.
"""
function FiniteVolumeMethod._extension_preconditioner(
        ::Val{:amg}, A::SparseMatrixCSC,
    )
    ml = AlgebraicMultigrid.ruge_stuben(A)
    return AlgebraicMultigrid.aspreconditioner(ml)
end

end # module

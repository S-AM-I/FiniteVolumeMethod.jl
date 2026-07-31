module FVMMPIExt

using FiniteVolumeMethod
using MPI: MPI
using PartitionedArrays: PartitionedArrays, distribute_with_mpi, LocalIndices
using SparseArrays: SparseArrays, SparseMatrixCSC, sparse, nnz, nzrange
using LinearAlgebra: norm
using StaticArrays: SVector

include("distributed_mesh.jl")
include("halo_exchange.jl")
include("partitioning.jl")
include("distributed_fields.jl")
include("distributed_solve.jl")

end # module FVMMPIExt

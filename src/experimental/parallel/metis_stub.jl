# parallel/metis_stub.jl — Interface stub for the Metis.jl weak-dep extension.
#
# The real implementation lives in `ext/FVMMetisExt.jl` and takes effect
# once the user does `using Metis` alongside `using FiniteVolumeMethod`.
# Until then, `partition_mesh_metis` errors with a helpful message.
#
# The sibling `partition_rcb` (`src/parallel/rcb_partitioner.jl`) is
# dependency-free and always available as a fallback.

"""
    partition_mesh_metis(mesh::UnstructuredFVMMesh{Dim, T}, nranks::Int) -> Vector{Int}

Partition `mesh`'s cells into `nranks` balanced buckets using Metis's
multilevel k-way graph partitioner, returning a length-`ncells` vector
`cell_to_rank` with entries in `0:nranks-1` (MPI's 0-based rank convention).

Metis partitions the cell-adjacency graph built from `mesh.face_cells`
(two cells are graph-neighbours iff they share an internal face), which
gives better communication balance than the geometric `partition_rcb`
fallback on meshes with poor spatial locality or high aspect ratio.

Metis.jl is a weak dependency — this function is a stub until the user
loads `using Metis`. The real implementation is provided by
`ext/FVMMetisExt.jl` and overrides this definition at ext-load time.

Fall back to [`partition_rcb`](@ref) if Metis.jl is not available.
"""
function partition_mesh_metis(::AbstractFVMMesh, ::Integer)
    error(
        "partition_mesh_metis requires Metis.jl. " *
            "Run `using Metis` (and have Metis.jl installed) to activate the " *
            "FVMMetisExt extension, or fall back to the dependency-free " *
            "`partition_rcb` geometric partitioner.",
    )
end

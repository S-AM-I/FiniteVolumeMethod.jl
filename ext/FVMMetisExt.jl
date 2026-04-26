module FVMMetisExt

# FVMMetisExt — Metis.jl-backed graph partitioner for `UnstructuredFVMMesh`.
#
# Builds a symmetric cell-adjacency graph from `mesh.face_cells` (cells
# sharing an internal face get an edge) and hands it to
# `Metis.partition(graph, nparts)`. Returns a length-`ncells` partition
# vector in 0:nranks-1, matching the convention used by the dep-free
# `partition_rcb` and consumed by `distribute_mesh` / `extract_local_mesh`.
#
# Loaded automatically when the user runs `using Metis` alongside
# `using FiniteVolumeMethod`. Overrides `partition_mesh_metis` from
# `src/parallel/metis_stub.jl`.

using FiniteVolumeMethod
using Metis: Metis
using SparseArrays: sparse, SparseMatrixCSC

"""
    FiniteVolumeMethod.partition_mesh_metis(
        mesh::FiniteVolumeMethod.UnstructuredFVMMesh, nranks::Integer,
    ) -> Vector{Int}

Metis.jl-backed partitioner. Builds the cell-adjacency graph from
`mesh.face_cells` and calls `Metis.partition`. Returns 0-based rank
IDs so the result slots directly into `extract_local_mesh` and
`distribute_mesh`.

Trivial cases (`nranks == 1`, or `nranks >= ncells`) short-circuit
Metis to avoid its minimum-part-size requirements on tiny meshes.
"""
function FiniteVolumeMethod.partition_mesh_metis(
        mesh::FiniteVolumeMethod.UnstructuredFVMMesh, nranks::Integer,
    )
    nranks >= 1 || error("nranks must be >= 1, got $nranks")
    nc = length(mesh.cell_volumes)
    nc > 0 || error("mesh has zero cells")

    # Trivial / degenerate cases — skip Metis entirely.
    if nranks == 1
        return zeros(Int, nc)
    end
    if nranks >= nc
        # One cell per rank (first nc ranks), remainder ranks get nothing.
        return collect(0:(nc - 1))
    end

    graph = _cell_adjacency_graph(mesh)
    parts = Metis.partition(graph, Int(nranks))

    # Metis returns 1-based part IDs; convert to the 0-based convention
    # shared with `partition_rcb` and MPI.
    return Int[Int(p) - 1 for p in parts]
end

"""
    _cell_adjacency_graph(mesh) -> SparseMatrixCSC{Int, Int}

Build the symmetric cell-cell adjacency as a sparse `nc × nc` matrix.
Two cells are adjacent iff they share an internal face
(`face_cells[2, f] != 0`). Self-edges are omitted — Metis expects the
graph without them.
"""
function _cell_adjacency_graph(mesh::FiniteVolumeMethod.UnstructuredFVMMesh)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Collect unique (P, N) pairs across internal faces.
    rows = Int[]
    cols = Int[]
    @inbounds for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        N == 0 && continue
        P == N && continue
        push!(rows, P); push!(cols, N)
        push!(rows, N); push!(cols, P)
    end

    vals = ones(Int, length(rows))
    # `sparse` will coalesce duplicates by summing weights; that's fine —
    # Metis only cares about connectivity (non-zero pattern).
    return sparse(rows, cols, vals, nc, nc)
end

end # module FVMMetisExt

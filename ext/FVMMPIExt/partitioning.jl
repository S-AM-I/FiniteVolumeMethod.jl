# partitioning.jl — Mesh distribution across MPI ranks

"""
    FiniteVolumeMethod.distribute_mesh(
        mesh::FiniteVolumeMethod.UnstructuredFVMMesh{Dim, T},
        comm::MPI.Comm;
        method::Symbol = :rcb,
    ) -> DistributedFVMMesh{Dim, T}

Partition a global `UnstructuredFVMMesh` across MPI ranks.

Each rank receives the full mesh but only "owns" a contiguous slice of cells.
Ghost cells are the non-owned cells that share a face with owned cells.
A [`HaloPattern`](@ref) is built from face connectivity so that
[`halo_exchange!`](@ref) can synchronize field values across ranks.

This initial implementation replicates the full mesh on every rank for
correctness validation.  A memory-efficient version that stores only
local + ghost cells is planned for a future iteration.

# Arguments
- `mesh` — global `UnstructuredFVMMesh{Dim, T}`
- `comm` — MPI communicator
- `method` — partitioning strategy (`:rcb` only, currently)
"""
function FiniteVolumeMethod.distribute_mesh(
        mesh::FiniteVolumeMethod.UnstructuredFVMMesh{Dim, T},
        comm::MPI.Comm;
        method::Symbol = :rcb,
    ) where {Dim, T}
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Divide cells evenly across ranks (contiguous ranges)
    nc = length(mesh.cell_volumes)
    cells_per_rank = div(nc, nranks)
    my_start = rank * cells_per_rank + 1
    my_end = rank == nranks - 1 ? nc : (rank + 1) * cells_per_rank
    n_owned = my_end - my_start + 1
    n_ghost = nc - n_owned

    global_to_local = Dict(i => i for i in 1:nc)
    local_to_global = collect(1:nc)

    # Build halo pattern from face connectivity
    halo = _build_halo_pattern(mesh, my_start, my_end, rank, nranks)

    return DistributedFVMMesh{Dim, T}(
        mesh, n_owned, n_ghost, halo, comm, rank, nranks,
        global_to_local, local_to_global,
    )
end

"""
    _build_halo_pattern(mesh, my_start, my_end, rank, nranks) -> HaloPattern

Scan face connectivity to identify which cells must be sent to / received
from each neighbor rank.  Cells are assigned to ranks by contiguous ranges.
"""
function _build_halo_pattern(mesh, my_start, my_end, rank, nranks)
    send_indices = Dict{Int, Vector{Int}}()
    recv_indices = Dict{Int, Vector{Int}}()
    nc = length(mesh.cell_volumes)
    cells_per_rank = div(nc, nranks)

    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        N == 0 && continue  # boundary face

        P_owned = my_start <= P <= my_end
        N_owned = my_start <= N <= my_end

        if P_owned && !N_owned
            other_rank = _cell_to_rank(N, cells_per_rank, nranks)
            push!(get!(Vector{Int}, send_indices, other_rank), P)
            push!(get!(Vector{Int}, recv_indices, other_rank), N)
        elseif !P_owned && N_owned
            other_rank = _cell_to_rank(P, cells_per_rank, nranks)
            push!(get!(Vector{Int}, send_indices, other_rank), N)
            push!(get!(Vector{Int}, recv_indices, other_rank), P)
        end
    end

    # Deduplicate indices
    for (k, v) in send_indices
        send_indices[k] = unique(v)
    end
    for (k, v) in recv_indices
        recv_indices[k] = unique(v)
    end

    neighbor_ranks = sort(collect(keys(send_indices)))
    return HaloPattern(send_indices, recv_indices, neighbor_ranks)
end

"""Map a 1-based cell index to a 0-based MPI rank given contiguous partitioning."""
function _cell_to_rank(cell_idx::Int, cells_per_rank::Int, nranks::Int)
    r = div(cell_idx - 1, cells_per_rank)
    return min(r, nranks - 1)
end

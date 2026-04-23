# partitioning.jl — Mesh distribution across MPI ranks (Stage 2 rewrite)
#
# Replaces the Stage-0/1 full-mesh-per-rank implementation with a true
# per-rank submesh plus halo layer. Uses the dep-free RCB partitioner
# in src/parallel/rcb_partitioner.jl to assign each global cell to a
# rank, then calls src/parallel/local_mesh.jl's `extract_local_mesh`
# on each process.

"""
    FiniteVolumeMethod.distribute_mesh(
        mesh::FiniteVolumeMethod.UnstructuredFVMMesh{Dim, T},
        comm::MPI.Comm;
        method::Symbol = :rcb,
    ) -> DistributedFVMMesh{Dim, T}

Partition a global `UnstructuredFVMMesh` across MPI ranks and build the
local submesh plus halo pattern for the calling rank.

Each rank:
1. Computes the global partition via `partition_rcb` (same result on every
   rank — cheap and deterministic compared to shipping the partition).
2. Extracts its own `LocalMeshData` (owned + halo cells).
3. Scans global faces to build a rank-by-rank `HaloPattern` in local
   indices.

Supported methods:
- `:rcb` — recursive coordinate bisection (default, deps-free).

Metis (`:metis`) is planned as a follow-up for meshes with poor
geometric locality.
"""
function FiniteVolumeMethod.distribute_mesh(
        mesh::FiniteVolumeMethod.UnstructuredFVMMesh{Dim, T},
        comm::MPI.Comm;
        method::Symbol = :rcb,
    ) where {Dim, T}
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    cell_to_rank = if method === :rcb
        FiniteVolumeMethod.partition_rcb(mesh, nranks)
    else
        error("unknown partitioning method :$method — only :rcb is currently supported")
    end

    local_data = FiniteVolumeMethod.extract_local_mesh(mesh, cell_to_rank, rank)
    halo = _build_halo_pattern(mesh, cell_to_rank, local_data, rank)

    return DistributedFVMMesh{Dim, T}(
        local_data.mesh,
        local_data.n_owned,
        local_data.n_local,
        halo,
        comm, rank, nranks,
        local_data.global_to_local,
        local_data.local_to_global,
        local_data.halo_owner_rank,
    )
end

"""
    _build_halo_pattern(global_mesh, cell_to_rank, local_data, my_rank) -> HaloPattern

Construct a `HaloPattern` expressed in LOCAL indices. For each internal
global face crossing a rank boundary into `my_rank`:

- the owned cell's local index goes into `send_indices[other_rank]`,
- the halo cell's local index goes into `recv_indices[other_rank]`.

The resulting lists are deduplicated so each cell is communicated at
most once per neighbour rank per exchange.
"""
function _build_halo_pattern(global_mesh, cell_to_rank, local_data, my_rank::Int)
    send_indices = Dict{Int, Vector{Int}}()
    recv_indices = Dict{Int, Vector{Int}}()
    nf = size(global_mesh.face_cells, 2)

    @inbounds for f in 1:nf
        P = global_mesh.face_cells[1, f]
        N = global_mesh.face_cells[2, f]
        N == 0 && continue  # boundary face

        P_owner = cell_to_rank[P]
        N_owner = cell_to_rank[N]
        P_owner == N_owner && continue  # both on same rank

        if P_owner == my_rank
            # I own P; neighbour N is halo — send P to N_owner, expect N back.
            local_P = local_data.global_to_local[P]
            local_N = local_data.global_to_local[N]
            push!(get!(Vector{Int}, send_indices, N_owner), local_P)
            push!(get!(Vector{Int}, recv_indices, N_owner), local_N)
        elseif N_owner == my_rank
            # I own N; neighbour P is halo — send N to P_owner, expect P back.
            local_P = local_data.global_to_local[P]
            local_N = local_data.global_to_local[N]
            push!(get!(Vector{Int}, send_indices, P_owner), local_N)
            push!(get!(Vector{Int}, recv_indices, P_owner), local_P)
        end
    end

    for (k, v) in send_indices
        send_indices[k] = unique(v)
    end
    for (k, v) in recv_indices
        recv_indices[k] = unique(v)
    end

    neighbor_ranks = sort(union(keys(send_indices), keys(recv_indices)))
    return HaloPattern(send_indices, recv_indices, neighbor_ranks)
end

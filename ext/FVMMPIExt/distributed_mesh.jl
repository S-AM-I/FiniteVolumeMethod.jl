# distributed_mesh.jl — Types for distributed FVM meshes with ghost cells

"""
    HaloPattern

Communication pattern for ghost cell halo exchange.

Stores per-neighbor send/receive index maps so that `halo_exchange!` can
synchronize ghost cell values via non-blocking MPI calls.

# Fields
- `send_indices` — `Dict{Int, Vector{Int}}`: neighbor rank to local cell indices to send
- `recv_indices` — `Dict{Int, Vector{Int}}`: neighbor rank to ghost cell indices to fill
- `neighbor_ranks` — sorted list of neighbor MPI ranks
"""
struct HaloPattern
    send_indices::Dict{Int, Vector{Int}}
    recv_indices::Dict{Int, Vector{Int}}
    neighbor_ranks::Vector{Int}
end

"""
    DistributedFVMMesh{Dim, T}

Distributed FVM mesh: local submesh plus ghost cells plus MPI bookkeeping.

Each rank holds the full mesh data but only "owns" a contiguous range of cells.
Ghost cells are the cells adjacent to the owned region that belong to other ranks.
This initial implementation prioritises correctness over memory efficiency; a
future version will store only local + ghost cells.

# Fields
- `local_mesh` — the underlying `UnstructuredFVMMesh{Dim, T}`
- `n_owned` — number of cells owned by this rank
- `n_ghost` — number of ghost cells from neighbors
- `halo` — [`HaloPattern`](@ref) for MPI communication
- `comm` — MPI communicator
- `rank` — this process's MPI rank (0-based)
- `nranks` — total number of MPI ranks
- `global_to_local` — mapping from global cell index to local index
- `local_to_global` — mapping from local index to global cell index
"""
struct DistributedFVMMesh{Dim, T}
    local_mesh::FiniteVolumeMethod.UnstructuredFVMMesh{Dim, T}
    n_owned::Int
    n_ghost::Int
    halo::HaloPattern
    comm::MPI.Comm
    rank::Int
    nranks::Int
    global_to_local::Dict{Int, Int}
    local_to_global::Vector{Int}
end

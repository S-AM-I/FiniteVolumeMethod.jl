# distributed_mesh.jl — Types for distributed FVM meshes with ghost cells
#
# Stage 2: real submesh-per-rank layout. Replaces the Stage 0/1
# "every rank holds the full mesh" workaround.
#
# Each rank constructs a `LocalMeshData` (owned cells + halo layer) using
# the dep-free RCB partitioner in src/parallel/rcb_partitioner.jl. The
# `HaloPattern` is then built from the partition so halo_exchange! can
# post MPI Irecv/Isend pairs against only the cells that actually need
# to travel between ranks.

"""
    HaloPattern

Communication pattern for ghost cell halo exchange, expressed in terms
of LOCAL cell indices on each rank.

# Fields
- `send_indices::Dict{Int, Vector{Int}}` — neighbour rank → local indices
  of owned cells that rank needs a copy of.
- `recv_indices::Dict{Int, Vector{Int}}` — neighbour rank → local indices
  (in the halo region, i.e. `n_owned + k` for some `k`) where values from
  that neighbour arrive.
- `neighbor_ranks::Vector{Int}` — sorted list of peer MPI ranks.
"""
struct HaloPattern
    send_indices::Dict{Int, Vector{Int}}
    recv_indices::Dict{Int, Vector{Int}}
    neighbor_ranks::Vector{Int}
end

"""
    DistributedFVMMesh{Dim, T}

Distributed FVM mesh: a real per-rank submesh plus MPI bookkeeping.

Stage 2 replaces the prior "each rank holds the full mesh" layout with a
true local-plus-halo representation:

- `local_mesh` contains only this rank's owned cells (`1..n_owned`)
  followed by its halo cells (`n_owned+1..n_local`). Halo cells carry
  the geometric data needed for 2nd-order stencil evaluation but are not
  updated by this rank's solvers — their values arrive via `halo_exchange!`.
- `halo` carries the send/recv index lists against the `local_mesh`
  indexing above.
- Global↔local maps let the solver lift local solutions back to the
  global frame for gather-based reporting.
"""
struct DistributedFVMMesh{Dim, T}
    local_mesh::FiniteVolumeMethod.UnstructuredFVMMesh{Dim, T}
    n_owned::Int
    n_local::Int
    halo::HaloPattern
    comm::MPI.Comm
    rank::Int
    nranks::Int
    global_to_local::Dict{Int, Int}
    local_to_global::Vector{Int}
    halo_owner_rank::Vector{Int}
end

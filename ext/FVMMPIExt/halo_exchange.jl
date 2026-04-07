# halo_exchange.jl — Non-blocking MPI halo exchange for ghost cell synchronization

"""
    FiniteVolumeMethod.halo_exchange!(values::Vector{T}, dmesh::DistributedFVMMesh) where {T}

Synchronize ghost cell values via non-blocking MPI send/recv.

Posts `Irecv!` for each neighbor rank, packs and sends owned-cell data
via `Isend`, waits for receives, unpacks into ghost positions, then
waits for sends to complete.
"""
function FiniteVolumeMethod.halo_exchange!(
        values::Vector{T}, dmesh::DistributedFVMMesh,
    ) where {T}
    halo = dmesh.halo

    # Post non-blocking receives
    recv_reqs = MPI.Request[]
    recv_buffers = Dict{Int, Vector{T}}()
    for rank in halo.neighbor_ranks
        indices = halo.recv_indices[rank]
        buf = Vector{T}(undef, length(indices))
        recv_buffers[rank] = buf
        push!(recv_reqs, MPI.Irecv!(buf, dmesh.comm; source = rank, tag = 0))
    end

    # Pack and send
    send_reqs = MPI.Request[]
    for rank in halo.neighbor_ranks
        indices = halo.send_indices[rank]
        buf = values[indices]
        push!(send_reqs, MPI.Isend(buf, dmesh.comm; dest = rank, tag = 0))
    end

    # Wait for receives and unpack
    MPI.Waitall(recv_reqs)
    for rank in halo.neighbor_ranks
        indices = halo.recv_indices[rank]
        for (i, idx) in enumerate(indices)
            values[idx] = recv_buffers[rank][i]
        end
    end

    # Wait for sends to complete before returning
    MPI.Waitall(send_reqs)

    return nothing
end

"""
    FiniteVolumeMethod.halo_exchange!(
        values::Vector{SVector{Dim, T}}, dmesh::DistributedFVMMesh{Dim},
    ) where {Dim, T}

Vector field version: exchange each spatial component independently.
"""
function FiniteVolumeMethod.halo_exchange!(
        values::Vector{SVector{Dim, T}}, dmesh::DistributedFVMMesh{Dim},
    ) where {Dim, T}
    n = length(values)
    for d in 1:Dim
        component = [values[i][d] for i in 1:n]
        FiniteVolumeMethod.halo_exchange!(component, dmesh)
        for i in 1:n
            values[i] = Base.setindex(values[i], component[i], d)
        end
    end
    return nothing
end

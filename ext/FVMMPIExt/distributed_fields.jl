# distributed_fields.jl — Distributed field wrappers with halo synchronization

"""
    DistributedScalarField{T}

Wrapper for a `CollocatedScalarField` on a distributed mesh.
Provides `sync!` to trigger halo exchange before gradient operations.
"""
struct DistributedScalarField{T}
    field::FiniteVolumeMethod.CollocatedScalarField{T}
    dmesh::DistributedFVMMesh
end

"""
    sync!(df::DistributedScalarField)

Synchronize ghost cell values of the wrapped scalar field across MPI ranks.
"""
function sync!(df::DistributedScalarField)
    FiniteVolumeMethod.halo_exchange!(df.field.internal, df.dmesh)
    return nothing
end

"""
    DistributedVectorField{Dim, T}

Wrapper for a `CollocatedVectorField` on a distributed mesh.
Provides `sync!` to trigger halo exchange before gradient operations.
"""
struct DistributedVectorField{Dim, T}
    field::FiniteVolumeMethod.CollocatedVectorField{Dim, T}
    dmesh::DistributedFVMMesh{Dim, T}
end

"""
    sync!(df::DistributedVectorField)

Synchronize ghost cell values of the wrapped vector field across MPI ranks.
"""
function sync!(df::DistributedVectorField)
    FiniteVolumeMethod.halo_exchange!(df.field.internal, df.dmesh)
    return nothing
end

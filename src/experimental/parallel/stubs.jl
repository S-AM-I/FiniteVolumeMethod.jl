# parallel/stubs.jl — Interface stubs for MPI extension
#
# These functions are overridden by FVMMPIExt when MPI.jl is loaded.

"""
    distribute_mesh(mesh, comm; method = :rcb)

Partition and distribute a mesh across MPI ranks.
Requires `using MPI, PartitionedArrays` to activate.
"""
function distribute_mesh end

"""
    halo_exchange!(values, dmesh)

Synchronize ghost cell values via MPI.
Requires `using MPI, PartitionedArrays` to activate.
"""
function halo_exchange! end

"""
    solve_simple_distributed(prob, dmesh; kwargs...)

Parallel SIMPLE solver with MPI communication.
Requires `using MPI, PartitionedArrays` to activate.
"""
function solve_simple_distributed end

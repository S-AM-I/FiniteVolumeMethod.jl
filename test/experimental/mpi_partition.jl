# test/mpi_partition.jl — Stage 2 serial contract test for the RCB
# partitioner and LocalFVMMesh extractor.
#
# Runs without MPI / mpiexec. Exercises the dep-free logic that the
# FVMMPIExt extension wires into `distribute_mesh` at runtime:
#
#   partition_rcb(mesh, nranks) → cell_to_rank::Vector{Int}
#   extract_local_mesh(global_mesh, cell_to_rank, my_rank)
#           → LocalMeshData{Dim, T}
#
# The real mpiexec parity test is test/mpi_parity.jl, which must be
# launched via `mpiexec -n N julia --project=test test/mpi_parity.jl`
# and is not part of the default `runtests.jl` loop.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: LocalFVMMesh, LocalMeshData, distribute_mesh, extract_local_mesh, partition_rcb
using FiniteVolumeMethod: CollocatedEquation, assemble_laplacian!
using FiniteVolumeMethod.Parabolic: DirichletBC
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "Stage 2b: partition_rcb load balance + determinism" begin
    mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)  # 32 cells
    nc = length(mesh.cell_volumes)

    # Single rank: all cells in bucket 0.
    c2r_1 = partition_rcb(mesh, 1)
    @test all(==(0), c2r_1)

    # Four ranks: perfectly balanced 8-cell buckets.
    c2r_4 = partition_rcb(mesh, 4)
    bucket_sizes = [count(==(r), c2r_4) for r in 0:3]
    @test bucket_sizes == [8, 8, 8, 8]
    @test all(r -> 0 <= r < 4, c2r_4)

    # Determinism: re-running gives the same partition.
    c2r_4_again = partition_rcb(mesh, 4)
    @test c2r_4 == c2r_4_again

    # Uneven counts (3 ranks over 32 cells): sum = 32, each bucket size
    # in {10, 11}.
    c2r_3 = partition_rcb(mesh, 3)
    bucket_sizes_3 = [count(==(r), c2r_3) for r in 0:2]
    @test sum(bucket_sizes_3) == nc
    @test all(s -> 10 <= s <= 11, bucket_sizes_3)
end

@testset "Stage 2c: extract_local_mesh — sizes, maps, halo correctness" begin
    mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    c2r = partition_rcb(mesh, 4)

    # Re-union the local meshes: every global cell appears in exactly one
    # rank's "owned" set, and every halo cell on rank r corresponds to an
    # owned cell on another rank.
    total_owned = 0
    global_cells_seen = Set{Int}()
    for my_rank in 0:3
        local_data = extract_local_mesh(mesh, c2r, my_rank)

        # Sizes agree with the partition.
        @test local_data.n_owned == count(==(my_rank), c2r)
        @test local_data.n_local >= local_data.n_owned
        @test length(local_data.local_to_global) == local_data.n_local
        @test length(local_data.halo_owner_rank) == local_data.n_local

        # First n_owned local cells point at this rank's globals.
        for i in 1:local_data.n_owned
            g = local_data.local_to_global[i]
            @test c2r[g] == my_rank
            @test local_data.halo_owner_rank[i] == my_rank
            push!(global_cells_seen, g)
        end

        # Halo cells point at other-rank globals.
        for i in (local_data.n_owned + 1):local_data.n_local
            g = local_data.local_to_global[i]
            @test c2r[g] != my_rank
            @test local_data.halo_owner_rank[i] == c2r[g]
        end

        # global_to_local is the inverse of local_to_global over the
        # local index range.
        for i in 1:local_data.n_local
            g = local_data.local_to_global[i]
            @test local_data.global_to_local[g] == i
        end

        # Local mesh geometry matches the global mesh for every local cell.
        for i in 1:local_data.n_local
            g = local_data.local_to_global[i]
            @test local_data.mesh.cell_volumes[i] == mesh.cell_volumes[g]
            for d in 1:2
                @test local_data.mesh.cell_centers[d, i] == mesh.cell_centers[d, g]
            end
        end

        total_owned += local_data.n_owned
    end

    # Every global cell is owned by exactly one rank.
    @test total_owned == nc
    @test length(global_cells_seen) == nc
end

@testset "Stage 2c: local face connectivity is well-formed" begin
    mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)
    c2r = partition_rcb(mesh, 4)

    for my_rank in 0:3
        local_data = extract_local_mesh(mesh, c2r, my_rank)
        lm = local_data.mesh

        # Every face-cells entry is in range.
        n_local_faces = size(lm.face_cells, 2)
        for f in 1:n_local_faces
            P = lm.face_cells[1, f]
            N = lm.face_cells[2, f]
            @test 1 <= P <= local_data.n_local
            @test N == 0 || 1 <= N <= local_data.n_local
        end

        # cell_faces is consistent with face_cells.
        for c in 1:local_data.n_local
            for f in lm.cell_faces[c]
                @test 1 <= f <= n_local_faces
                @test lm.face_cells[1, f] == c || lm.face_cells[2, f] == c
            end
        end

        # Boundary faces attached to an owned cell carry their original tag.
        if mesh.face_tags !== nothing
            @test lm.face_tags !== nothing
            for f in 1:n_local_faces
                gf = local_data.local_to_global_face[f]
                @test lm.face_tags[f] === mesh.face_tags[gf]
            end
        end
    end
end

@testset "Stage 2: CollocatedEquation on a local submesh assembles correctly" begin
    # Assemble Laplacian on the global mesh, then on each rank's local
    # submesh. For each owned cell, the local matrix row should match the
    # global matrix row restricted to owned+halo columns. This is the
    # invariant that makes Additive Schwarz converge correctly.

    mesh = build_cartesian_unstructured_mesh(6, 4, 1.5, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(1.0),
    )

    # Global reference assembly.
    global_eq = CollocatedEquation(mesh)
    assemble_laplacian!(global_eq, 1.0, mesh, bcs)

    c2r = partition_rcb(mesh, 2)
    for my_rank in 0:1
        local_data = extract_local_mesh(mesh, c2r, my_rank)
        local_eq = CollocatedEquation(local_data.mesh)
        assemble_laplacian!(local_eq, 1.0, local_data.mesh, bcs)

        # For every owned local cell, the diagonal should equal the global
        # diagonal (matrix rows are self-contained up to the halo layer).
        for i in 1:local_data.n_owned
            g = local_data.local_to_global[i]
            @test local_eq.A[i, i] ≈ global_eq.A[g, g]
            @test local_eq.b[i] ≈ global_eq.b[g]
        end
    end
end

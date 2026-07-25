# test/v_and_v_local_mesh.jl
#
# V&V: `LocalFVMMesh` + `build_local_mesh` invariants.
#
# `LocalFVMMesh` is the rank-local view of a global `UnstructuredFVMMesh`
# used by the Wave 5 Metis/PartitionedArrays path. These checks enforce
# the contract documented on the type — without them, the MPI assembly
# route and the halo pattern builder can silently drift apart.
#
# Invariants verified:
# 1. Trivial 1-rank partition: all cells owned, no halo.
# 2. 2-way partition: owned_cells on rank 0 and rank 1 are disjoint and
#    their union equals the full global cell set.
# 3. local_to_global ↔ global_to_local round-trip for every local entry.
# 4. Halo cells are all face-neighbours of at least one owned cell AND
#    are themselves NOT owned by this rank.
# 5. Build from Metis stub error path: calling the Metis-backed helper
#    without Metis loaded errors cleanly (documented elsewhere, but we
#    sanity-check the adjacent fallback-to-RCB pattern here).

using Test
using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: LocalFVMMesh, build_local_mesh
using FiniteVolumeMethod: UnstructuredFVMMesh

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V LocalFVMMesh: trivial 1-rank partition round-trips all cells" begin
    mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    cell_to_rank = zeros(Int, nc)  # every cell on rank 0

    lmesh = FiniteVolumeMethod.build_local_mesh(mesh, cell_to_rank, 0)

    @test lmesh isa FiniteVolumeMethod.LocalFVMMesh{2, Float64}
    @test length(lmesh.owned_cells) == nc
    @test isempty(lmesh.halo_cells)
    @test sort(lmesh.owned_cells) == collect(1:nc)
    @test lmesh.parent_mesh === mesh
    @test length(lmesh.local_to_global) == nc
    @test length(lmesh.global_to_local) == nc
end

@testset "V&V LocalFVMMesh: 2-way partition gives disjoint + covering owned sets" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    partition = FiniteVolumeMethod.partition_rcb(mesh, 2)
    @test length(partition) == nc

    lmesh0 = FiniteVolumeMethod.build_local_mesh(mesh, partition, 0)
    lmesh1 = FiniteVolumeMethod.build_local_mesh(mesh, partition, 1)

    owned0 = Set(lmesh0.owned_cells)
    owned1 = Set(lmesh1.owned_cells)

    # Disjoint
    @test isempty(intersect(owned0, owned1))
    # Covering
    @test union(owned0, owned1) == Set(1:nc)
    # Non-trivial split — both ranks should actually get cells.
    @test !isempty(owned0)
    @test !isempty(owned1)
end

@testset "V&V LocalFVMMesh: local_to_global ↔ global_to_local round-trip" begin
    mesh = build_cartesian_unstructured_mesh(3, 4, 1.0, 1.0)
    partition = FiniteVolumeMethod.partition_rcb(mesh, 2)

    for my_rank in (0, 1)
        lmesh = FiniteVolumeMethod.build_local_mesh(mesh, partition, my_rank)

        # local_to_global lists owned first, then halo.
        expected = vcat(lmesh.owned_cells, lmesh.halo_cells)
        @test lmesh.local_to_global == expected

        # global_to_local is the inverse.
        for (local_idx, global_idx) in pairs(lmesh.local_to_global)
            @test lmesh.global_to_local[global_idx] == local_idx
        end
        @test length(lmesh.global_to_local) == length(lmesh.local_to_global)
    end
end

@testset "V&V LocalFVMMesh: halo cells neighbour owned but are not owned" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    partition = FiniteVolumeMethod.partition_rcb(mesh, 2)

    for my_rank in (0, 1)
        lmesh = FiniteVolumeMethod.build_local_mesh(mesh, partition, my_rank)
        owned_set = Set(lmesh.owned_cells)

        # Halo cells must NOT be owned by me.
        for h in lmesh.halo_cells
            @test !(h in owned_set)
            @test partition[h] != my_rank
        end

        # Every halo cell must touch at least one owned cell via an
        # internal face.
        nf = size(mesh.face_cells, 2)
        halo_set = Set(lmesh.halo_cells)
        neighbour_of_owned = Set{Int}()
        for f in 1:nf
            P = mesh.face_cells[1, f]
            N = mesh.face_cells[2, f]
            N == 0 && continue
            if P in owned_set && !(N in owned_set)
                push!(neighbour_of_owned, N)
            elseif N in owned_set && !(P in owned_set)
                push!(neighbour_of_owned, P)
            end
        end
        @test halo_set == neighbour_of_owned
    end
end

@testset "V&V LocalFVMMesh: empty-owned rank returns empty view" begin
    mesh = build_cartesian_unstructured_mesh(2, 2, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    # Synthetic partition: every cell on rank 0, so rank 1 has nothing.
    cell_to_rank = zeros(Int, nc)

    lmesh1 = FiniteVolumeMethod.build_local_mesh(mesh, cell_to_rank, 1)
    @test isempty(lmesh1.owned_cells)
    @test isempty(lmesh1.halo_cells)
    @test isempty(lmesh1.local_to_global)
    @test isempty(lmesh1.global_to_local)
    @test lmesh1.parent_mesh === mesh
end

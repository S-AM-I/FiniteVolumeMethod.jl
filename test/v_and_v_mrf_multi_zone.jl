# test/v_and_v_mrf_multi_zone.jl — Multi-zone MRF V&V
#
# Gates the `MultiMRF` / `add_multi_mrf_source!` /
# `build_multi_mrf_from_zones` primitives. Verifies:
#
#   1. Disjoint zones: multi-zone source equals the sum of per-zone
#      sources.
#   2. Non-disjoint zones: `build_multi_mrf_from_zones` raises
#      `ArgumentError`.
#   3. Cells outside every zone get zero source.
#   4. Two-zone hand-computed reference, rtol 1e-14.

using FiniteVolumeMethod
using FiniteVolumeMethod: MRFZone, MultiMRF, build_multi_mrf_from_zones,
    add_mrf_source!, add_multi_mrf_source!, mrf_cell_source
using LinearAlgebra: norm
using StaticArrays: SVector
using Test

include("TestHelpers.jl")

@testset "V&V MRF multi-zone: disjoint union equals per-zone sum" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    n = length(mesh.cell_volumes)

    zone_a = MRFZone{Float64}(
        SVector(0.0, 0.0, 3.0),
        SVector(0.25, 0.25, 0.0),
        [1, 2, 5, 6],
    )
    zone_b = MRFZone{Float64}(
        SVector(0.0, 0.0, -2.0),
        SVector(0.75, 0.75, 0.0),
        [11, 12, 15, 16],
    )

    multi = build_multi_mrf_from_zones([zone_a, zone_b])
    rho = 1.1
    U = [SVector(0.3, -0.7, 0.0) for _ in 1:n]

    # Compute multi-zone source.
    src_multi = fill(SVector(0.0, 0.0, 0.0), n)
    add_multi_mrf_source!(src_multi, U, mesh, multi, rho)

    # Compute per-zone sources.
    src_a = fill(SVector(0.0, 0.0, 0.0), n)
    add_mrf_source!(src_a, U, mesh, zone_a, rho)
    src_b = fill(SVector(0.0, 0.0, 0.0), n)
    add_mrf_source!(src_b, U, mesh, zone_b, rho)

    for c in 1:n
        @test src_multi[c] ≈ src_a[c] + src_b[c] rtol = 1.0e-14 atol = 1.0e-14
    end
end

@testset "V&V MRF multi-zone: overlapping zones rejected" begin
    zone_a = MRFZone{Float64}(
        SVector(0.0, 0.0, 1.0),
        SVector(0.0, 0.0, 0.0),
        [1, 2, 3, 4],
    )
    zone_b = MRFZone{Float64}(
        SVector(0.0, 0.0, -1.0),
        SVector(1.0, 0.0, 0.0),
        [3, 4, 5, 6],  # 3 and 4 overlap with zone_a
    )

    @test_throws ArgumentError build_multi_mrf_from_zones([zone_a, zone_b])

    # Same-zone duplicates are also caught (a cell appearing twice in the
    # combined set must raise — here by listing zone_a twice).
    @test_throws ArgumentError build_multi_mrf_from_zones([zone_a, zone_a])
end

@testset "V&V MRF multi-zone: cells outside every zone get zero source" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    n = length(mesh.cell_volumes)

    zone = MRFZone{Float64}(
        SVector(0.0, 0.0, 2.5),
        SVector(0.0, 0.0, 0.0),
        [1, 4, 9],
    )
    multi = build_multi_mrf_from_zones([zone])
    rho = 1.0
    U = [SVector(1.0, 1.0, 0.0) for _ in 1:n]

    src = fill(SVector(0.0, 0.0, 0.0), n)
    add_multi_mrf_source!(src, U, mesh, multi, rho)

    for c in 1:n
        c in zone.cells && continue
        @test src[c] ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-14
    end
end

@testset "V&V MRF multi-zone: two-zone hand-computed reference" begin
    mesh = build_cartesian_unstructured_mesh(2, 2, 1.0, 1.0)
    n = length(mesh.cell_volumes)  # 4 cells, centres at (0.25, 0.25), etc.

    # Zone 1: covers cell 1, ω_z = 2, axis through (0, 0, 0).
    # Zone 2: covers cell 4, ω_z = -3, axis through (1, 1, 0).
    zone_a = MRFZone{Float64}(
        SVector(0.0, 0.0, 2.0),
        SVector(0.0, 0.0, 0.0),
        [1],
    )
    zone_b = MRFZone{Float64}(
        SVector(0.0, 0.0, -3.0),
        SVector(1.0, 1.0, 0.0),
        [4],
    )
    multi = build_multi_mrf_from_zones([zone_a, zone_b])

    rho = 2.0
    U = [SVector(1.0, 0.0, 0.0) for _ in 1:n]
    U[1] = SVector(1.0, 2.0, 0.0)
    U[4] = SVector(-0.5, 1.5, 0.0)

    src = fill(SVector(0.0, 0.0, 0.0), n)
    add_multi_mrf_source!(src, U, mesh, multi, rho)

    # Cell 1 centre: (0.25, 0.25, 0), r_1 = (0.25, 0.25, 0), ω = (0,0,2).
    # Coriolis: -2 ρ (ω × U_1).  ω × U_1 = (0,0,2)×(1,2,0) = (-4, 2, 0).
    #   F_cor_1 = -2·2·(-4, 2, 0) = (16, -8, 0).
    # Centrifugal: -ρ (ω × (ω × r_1)).
    #   ω × r_1 = (0,0,2)×(0.25,0.25,0) = (-0.5, 0.5, 0).
    #   ω × (ω × r_1) = (0,0,2)×(-0.5,0.5,0) = (-1, -1, 0).
    #   F_cent_1 = -2·(-1, -1, 0) = (2, 2, 0).
    expected_1 = SVector(16.0, -8.0, 0.0) + SVector(2.0, 2.0, 0.0)
    @test src[1] ≈ expected_1 rtol = 1.0e-14

    # Cell 4 centre: (0.75, 0.75, 0), r_4 = (-0.25, -0.25, 0), ω = (0,0,-3).
    # Coriolis: -2 ρ (ω × U_4).
    #   ω × U_4 = (0,0,-3)×(-0.5,1.5,0) = (0·0 − -3·1.5, -3·-0.5 − 0·0, 0·1.5 − 0·-0.5)
    #           = (4.5, 1.5, 0).
    #   F_cor_4 = -2·2·(4.5, 1.5, 0) = (-18, -6, 0).
    # Centrifugal: -ρ (ω × (ω × r_4)).
    #   ω × r_4 = (0,0,-3)×(-0.25,-0.25,0) = (-0.75, 0.75, 0).
    #   ω × (ω × r_4) = (0,0,-3)×(-0.75, 0.75, 0) = (2.25, 2.25, 0).
    #   F_cent_4 = -2·(2.25, 2.25, 0) = (-4.5, -4.5, 0).
    expected_4 = SVector(-18.0, -6.0, 0.0) + SVector(-4.5, -4.5, 0.0)
    @test src[4] ≈ expected_4 rtol = 1.0e-14

    # Cells 2 and 3 are outside both zones.
    @test src[2] ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-14
    @test src[3] ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-14
end

@testset "V&V MRF multi-zone: per-zone density vector" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    n = length(mesh.cell_volumes)

    zone_a = MRFZone{Float64}(
        SVector(0.0, 0.0, 1.0),
        SVector(0.0, 0.0, 0.0),
        [1, 2],
    )
    zone_b = MRFZone{Float64}(
        SVector(0.0, 0.0, -1.0),
        SVector(1.0, 1.0, 0.0),
        [15, 16],
    )
    multi = build_multi_mrf_from_zones([zone_a, zone_b])

    U = [SVector(1.0, 0.0, 0.0) for _ in 1:n]
    rho_vec = [0.5, 2.0]

    src = fill(SVector(0.0, 0.0, 0.0), n)
    add_multi_mrf_source!(src, U, mesh, multi, rho_vec)

    # Reconstruct from per-zone sources.
    src_a = fill(SVector(0.0, 0.0, 0.0), n)
    add_mrf_source!(src_a, U, mesh, zone_a, 0.5)
    src_b = fill(SVector(0.0, 0.0, 0.0), n)
    add_mrf_source!(src_b, U, mesh, zone_b, 2.0)

    for c in 1:n
        @test src[c] ≈ src_a[c] + src_b[c] rtol = 1.0e-14 atol = 1.0e-14
    end

    # Mismatched length raises.
    bad_rho = [1.0]
    @test_throws ArgumentError add_multi_mrf_source!(
        fill(SVector(0.0, 0.0, 0.0), n), U, mesh, multi, bad_rho,
    )
end

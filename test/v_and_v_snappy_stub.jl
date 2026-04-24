# test/v_and_v_snappy_stub.jl — verify snappy stub round-trip.

using FiniteVolumeMethod
using StaticArrays
using Test

@testset "V&V: SnappyMesher — constructor round-trip" begin
    mesher = SnappyMesher(
        ; stl_path = "/tmp/unused.stl",
        bbox_min = SVector(0.0, 0.0, 0.0),
        bbox_max = SVector(1.0, 1.0, 1.0),
        base_level = 1, surface_level = 3,
    )
    @test mesher.stl_path == "/tmp/unused.stl"
    @test mesher.base_level == 1
    @test mesher.surface_level == 3
end

@testset "V&V: SnappyMesher — defaults" begin
    mesher = SnappyMesher()
    @test mesher.base_level == 1
    @test mesher.n_layers == 0
    @test mesher.expansion_ratio == 1.2
end

# test/v_and_v_octree.jl — octree refinement primitive V&V.

using FiniteVolumeMethod
using StaticArrays
using Test

@testset "V&V: octree — single root has one leaf" begin
    tree = FiniteVolumeMethod.Octree{3, Float64}(
        SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 0,
    )
    @test FiniteVolumeMethod.count_leaves(tree) == 1
    @test length(FiniteVolumeMethod.leaves(tree)) == 1
end

@testset "V&V: octree — one refinement of a 3D node gives 8 children" begin
    tree = FiniteVolumeMethod.Octree{3, Float64}(
        SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 0,
    )
    FiniteVolumeMethod.subdivide!(tree)
    @test FiniteVolumeMethod.count_leaves(tree) == 8
end

@testset "V&V: octree — uniform refinement to depth 2 gives 8^2 = 64 leaves" begin
    tree = FiniteVolumeMethod.Octree{3, Float64}(
        SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 0,
    )
    FiniteVolumeMethod._uniform_refine!(tree, 2)
    @test FiniteVolumeMethod.count_leaves(tree) == 64
end

@testset "V&V: octree — callback-driven refinement with false criterion keeps root" begin
    tree = FiniteVolumeMethod.build_octree(
        SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 3, node -> false,
    )
    @test FiniteVolumeMethod.count_leaves(tree) == 1
end

@testset "V&V: octree — callback-driven refinement with always-true up to depth 2" begin
    tree = FiniteVolumeMethod.build_octree(
        SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 2, node -> true,
    )
    @test FiniteVolumeMethod.count_leaves(tree) == 64
end

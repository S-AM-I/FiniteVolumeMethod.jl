# test/experimental/octree_meshing.jl — octree castellated refinement primitives.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: Octree, build_octree, count_leaves, refine_near_sphere!, subdivide!
using FiniteVolumeMethod: is_leaf
using Test
using StaticArrays: SVector

@testset "Octree uniform refinement produces 2^(Dim·level) leaves" begin
    # 3D level-0: single leaf.
    tree0 = build_octree(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 0)
    @test count_leaves(tree0) == 1
    @test is_leaf(tree0)

    # 3D level-2: 8^2 = 64 leaves.
    tree2 = build_octree(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 2)
    @test count_leaves(tree2) == 64

    # 3D level-3: 8^3 = 512 leaves.
    tree3 = build_octree(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 3)
    @test count_leaves(tree3) == 512

    # 2D level-2: 4^2 = 16 leaves.
    tree2d = build_octree(SVector(0.0, 0.0), SVector(1.0, 1.0), 2)
    @test count_leaves(tree2d) == 16
end

@testset "Subdivide is a no-op on non-leaves" begin
    tree = build_octree(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 1)
    # already has children
    @test !is_leaf(tree)
    leaves_before = count_leaves(tree)
    subdivide!(tree)   # no effect
    @test count_leaves(tree) == leaves_before
end

@testset "Sphere intersection drives surface refinement" begin
    # Empty box (far from sphere): no refinement.
    tree_far = Octree{3, Float64}(
        SVector(10.0, 10.0, 10.0), SVector(11.0, 11.0, 11.0), 0,
    )
    refine_near_sphere!(tree_far, SVector(0.0, 0.0, 0.0), 0.1, 3)
    @test count_leaves(tree_far) == 1   # unchanged

    # Box containing the sphere: refined to target level.
    tree_near = Octree{3, Float64}(
        SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 0,
    )
    refine_near_sphere!(tree_near, SVector(0.5, 0.5, 0.5), 0.3, 3)
    leaves = count_leaves(tree_near)
    @test leaves > 1
    @test leaves <= 512   # bounded by uniform level-3
end

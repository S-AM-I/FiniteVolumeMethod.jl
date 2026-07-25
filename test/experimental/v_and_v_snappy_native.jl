# test/v_and_v_snappy_native.jl — castellated + snap invariants (v3.1).

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: SnappyMesher
using FiniteVolumeMethod:
    read_stl_ascii, write_stl_ascii,
    build_castellated_mesh, build_snappy_mesh,
    snap_to_surface!, cell_count, leaves, triangle_intersects_aabb
using StaticArrays
using LinearAlgebra: norm
using Test

"""
    _unit_cube_stl(path; lo = 0.27, hi = 0.73) -> path

Write a 12-triangle unit-cube STL at `[lo, hi]^3` into `path`.  Default
`lo`/`hi` are deliberately off the octree cell boundaries so that the
surface-snap step has something to project onto.
"""
function _unit_cube_stl(path::AbstractString; lo::Float64 = 0.27, hi::Float64 = 0.73)
    corners = [
        SVector(lo, lo, lo), SVector(hi, lo, lo),
        SVector(lo, hi, lo), SVector(hi, hi, lo),
        SVector(lo, lo, hi), SVector(hi, lo, hi),
        SVector(lo, hi, hi), SVector(hi, hi, hi),
    ]
    tri_table = [
        (1, 3, 2, SVector(0.0, 0.0, -1.0)),
        (2, 3, 4, SVector(0.0, 0.0, -1.0)),
        (5, 6, 7, SVector(0.0, 0.0, 1.0)),
        (6, 8, 7, SVector(0.0, 0.0, 1.0)),
        (1, 2, 5, SVector(0.0, -1.0, 0.0)),
        (2, 6, 5, SVector(0.0, -1.0, 0.0)),
        (3, 7, 4, SVector(0.0, 1.0, 0.0)),
        (4, 7, 8, SVector(0.0, 1.0, 0.0)),
        (1, 5, 3, SVector(-1.0, 0.0, 0.0)),
        (3, 5, 7, SVector(-1.0, 0.0, 0.0)),
        (2, 4, 6, SVector(1.0, 0.0, 0.0)),
        (4, 8, 6, SVector(1.0, 0.0, 0.0)),
    ]
    faces = [(t[1], t[2], t[3]) for t in tri_table]
    normals = [t[4] for t in tri_table]
    return write_stl_ascii(path, corners, faces, normals)
end

@testset "V&V: snappy native — castellated invariants" begin
    tmp = mktempdir()
    stl = joinpath(tmp, "cube.stl")
    _unit_cube_stl(stl)

    mesher = SnappyMesher(;
        stl_path = stl,
        bbox_min = SVector(0.0, 0.0, 0.0),
        bbox_max = SVector(1.0, 1.0, 1.0),
        base_level = 2,
        surface_level = 4,
        snap_iterations = 0,  # castellated only
    )

    uniform_octree = FiniteVolumeMethod.build_octree(
        mesher.bbox_min, mesher.bbox_max, mesher.base_level,
    )
    uniform_count = cell_count(uniform_octree)

    octree = build_castellated_mesh(mesher)
    @test cell_count(octree) >= uniform_count
    @test cell_count(octree) > uniform_count  # STL must trigger refinement

    # Deepest leaves lie near the STL bounding box.
    deepest = maximum(leaf.level for leaf in leaves(octree))
    @test deepest == mesher.surface_level

    cube_min = SVector(0.27, 0.27, 0.27)
    cube_max = SVector(0.73, 0.73, 0.73)

    # Every deepest leaf sits within one cell-size of the STL bounding
    # box.  Siblings of a surface-intersecting cell can remain at the
    # deepest level after a parent subdivide (they are created but not
    # re-subdivided), so the tight invariant is a one-cell band, not a
    # strict AABB overlap.
    deepest_cell_size = (mesher.bbox_max - mesher.bbox_min) / 2^deepest
    near_band = deepest_cell_size
    for leaf in leaves(octree)
        leaf.level == deepest || continue
        near = all(
            leaf.bbox_min[d] <= cube_max[d] + near_band[d] + 1.0e-12 &&
                leaf.bbox_max[d] >= cube_min[d] - near_band[d] - 1.0e-12
                for d in 1:3
        )
        @test near
    end

    # At least one deepest leaf strictly overlaps the cube bbox — so the
    # STL-driven refinement really did target the surface.
    strict_overlap = any(
        all(
                leaf.bbox_min[d] <= cube_max[d] + 1.0e-12 &&
                leaf.bbox_max[d] >= cube_min[d] - 1.0e-12
                for d in 1:3
            ) && leaf.level == deepest
            for leaf in leaves(octree)
    )
    @test strict_overlap
end

@testset "V&V: snappy native — surface snap moves vertices" begin
    tmp = mktempdir()
    stl = joinpath(tmp, "cube.stl")
    _unit_cube_stl(stl)

    mesher = SnappyMesher(;
        stl_path = stl,
        bbox_min = SVector(0.0, 0.0, 0.0),
        bbox_max = SVector(1.0, 1.0, 1.0),
        base_level = 2,
        surface_level = 3,
        snap_iterations = 10,
    )

    snapshot = build_snappy_mesh(mesher)
    @test snapshot.snap_applied
    @test snapshot.snap_moved > 0.0
    @test snapshot.snap_iters >= 1
    @test snapshot.layers_added == 0  # deferred to v3.2
    @test snapshot.n_cells == cell_count(snapshot.octree)
end

@testset "V&V: snappy native — snap idempotency (fixed-point)" begin
    tmp = mktempdir()
    stl = joinpath(tmp, "cube.stl")
    _unit_cube_stl(stl)

    mesher = SnappyMesher(;
        stl_path = stl,
        bbox_min = SVector(0.0, 0.0, 0.0),
        bbox_max = SVector(1.0, 1.0, 1.0),
        base_level = 2,
        surface_level = 3,
        snap_iterations = 30,
    )

    octree = build_castellated_mesh(mesher)
    stl_vertices, stl_faces, _ = read_stl_ascii(stl)

    # First snap to near convergence.
    snap_to_surface!(
        octree, stl_vertices, stl_faces;
        max_iters = 30, tol = 1.0e-8, damping = 0.5,
    )

    # Record the leaf bboxes after snap 1.
    leaf_list = leaves(octree)
    mins_1 = [leaf.bbox_min for leaf in leaf_list]
    maxs_1 = [leaf.bbox_max for leaf in leaf_list]

    # Re-run snap on the already-snapped octree.
    moved_2, _ = snap_to_surface!(
        octree, stl_vertices, stl_faces;
        max_iters = 30, tol = 1.0e-8, damping = 0.5,
    )

    leaf_list_2 = leaves(octree)
    mins_2 = [leaf.bbox_min for leaf in leaf_list_2]
    maxs_2 = [leaf.bbox_max for leaf in leaf_list_2]

    @test length(leaf_list) == length(leaf_list_2)
    max_drift = 0.0
    for k in eachindex(mins_1)
        max_drift = max(max_drift, norm(mins_2[k] - mins_1[k]))
        max_drift = max(max_drift, norm(maxs_2[k] - maxs_1[k]))
    end
    # Fixed-point property: a second snap pass barely moves anything.
    @test max_drift < 1.0e-6
    # The residual movement is small in absolute terms too.
    @test moved_2 < 1.0e-4
end

@testset "V&V: snappy native — build_snappy_mesh no-STL graceful" begin
    mesher = SnappyMesher(;
        stl_path = "",
        bbox_min = SVector(0.0, 0.0, 0.0),
        bbox_max = SVector(1.0, 1.0, 1.0),
        base_level = 2, surface_level = 3,
        snap_iterations = 5,
    )
    snapshot = build_snappy_mesh(mesher)
    @test snapshot.snap_applied == false
    @test snapshot.layers_added == 0
    @test snapshot.n_cells == cell_count(snapshot.octree)
end

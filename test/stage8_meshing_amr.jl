# test/stage8_meshing_amr.jl — Stage 8 octree meshing + collocated AMR gates

using FiniteVolumeMethod
using Test
using StaticArrays: SVector

include("TestHelpers.jl")

@testset "Stage 8a: Octree uniform refinement produces 2^(Dim·level) leaves" begin
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

@testset "Stage 8a: Subdivide is a no-op on non-leaves" begin
    tree = build_octree(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), 1)
    # already has children
    @test !is_leaf(tree)
    leaves_before = count_leaves(tree)
    subdivide!(tree)   # no effect
    @test count_leaves(tree) == leaves_before
end

@testset "Stage 8a: Sphere intersection drives surface refinement" begin
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

@testset "Stage 8c: mark_cells_by_gradient produces expected markers" begin
    mesh = build_cartesian_unstructured_mesh(5, 5, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    # Synthetic gradient: large in cell 1, small elsewhere.
    grad = fill(SVector(0.0, 0.0), nc)
    grad[1] = SVector(100.0, 0.0)

    markers = mark_cells_by_gradient(
        grad, mesh;
        refine_threshold = 1.0, coarsen_threshold = 0.01,
    )
    @test length(markers) == nc
    @test markers[1] === :refine  # big gradient → refine
    @test markers[2] === :coarsen  # zero gradient < coarsen threshold
end

@testset "Stage 8c: flux_correction_factor reports conservation ratio" begin
    # Four children tile parent exactly (Cartesian 2:1 refinement in 2D).
    parent_area = 1.0
    child_areas = [0.25, 0.25, 0.25, 0.25]
    @test flux_correction_factor(parent_area, child_areas) ≈ 1.0

    # Mismatched areas expose the discrepancy.
    child_mismatched = [0.25, 0.25, 0.25, 0.2]
    factor = flux_correction_factor(parent_area, child_mismatched)
    @test factor > 1.0   # missing area → scale correction factor > 1
end

@testset "Stage 8d: ZZ error indicator small in the interior for linear field" begin
    # A linear scalar field has a constant gradient in the interior. The
    # boundary-adjacent cells pick up discretization error from the
    # Green-Gauss boundary stencil (no explicit BC on this constructed
    # field), so we only assert smallness on the deep-interior cells.
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi = CollocatedScalarField(:phi, mesh)
    for c in 1:nc
        phi.internal[c] = mesh.cell_centers[1, c]  # linear in x
    end

    indicator = zz_error_indicator(phi, mesh)
    @test length(indicator) == nc

    # Interior cells: strictly inside the 12x12 grid (rows 4-9, cols 4-9).
    # They have neighbours that are themselves interior, so both local
    # and recovered gradients equal the analytical 1·x̂ to machine
    # precision.
    interior_ids = Int[]
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.3 < x < 0.7 && 0.3 < y < 0.7
            push!(interior_ids, c)
        end
    end
    @test !isempty(interior_ids)
    @test maximum(indicator[interior_ids]) < 1.0e-8
end

@testset "Stage 8d: ZZ error indicator catches large-gradient contrast" begin
    # A piecewise-constant step function: sharp transition → large
    # discrepancy between local and recovered gradients.
    #
    # On a Cartesian mesh the Green-Gauss and least-squares gradients
    # agree to machine precision for any scalar field (both reduce to
    # central differences), so the GG-vs-LSQ form of `zz_error_indicator`
    # is not the right diagnostic here. Use the smoothed face-neighbour
    # variant that compares each cell's local gradient with the
    # volume-weighted average of its neighbours' gradients — that
    # discrepancy peaks at interface cells.
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi = CollocatedScalarField(:phi, mesh)
    for c in 1:nc
        phi.internal[c] = mesh.cell_centers[1, c] < 0.5 ? 0.0 : 1.0
    end

    indicator = FiniteVolumeMethod._zz_indicator_smoothed(phi, mesh)
    # At least some cells near the interface should have non-zero indicator.
    @test maximum(indicator) > 0.1
end

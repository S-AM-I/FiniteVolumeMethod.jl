# test/v_and_v_overset.jl — Overset/chimera donor-receiver transfer V&V
#
# Invariants checked:
# 1. Constant donor field ⇒ constant receiver field (rtol 1e-14)
# 2. Linear field x + 2y reproduced exactly at overset cells (rtol 1e-12)
# 3. Barycentric weights sum to 1 per receiver
# 4. Receivers are correctly masked via `is_receiver`

using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "src", "collocated", "dynamic_mesh", "overset.jl"))

@testset "V&V overset: constant donor field ⇒ constant receiver field" begin
    # Donor mesh: 3×3 background Cartesian centers
    donor_centers = zeros(Float64, 2, 9)
    for j in 1:3, i in 1:3
        c = (j - 1) * 3 + i
        donor_centers[1, c] = (i - 0.5) * (1.0 / 3)
        donor_centers[2, c] = (j - 0.5) * (1.0 / 3)
    end
    # Overset mesh: 2×2 cells slightly offset
    overset_centers = zeros(Float64, 2, 4)
    for j in 1:2, i in 1:2
        c = (j - 1) * 2 + i
        overset_centers[1, c] = 0.25 + (i - 0.5) * 0.25
        overset_centers[2, c] = 0.25 + (j - 0.5) * 0.25
    end

    receiver_idxs = [1, 2, 3, 4]
    assembly = build_nearest_neighbour_assembly(
        overset_centers, donor_centers, receiver_idxs,
    )

    phi_donor = fill(7.25, 9)
    phi_receiver = zeros(Float64, 4)
    interpolate_overset!(phi_receiver, phi_donor, assembly)

    for r in receiver_idxs
        @test phi_receiver[r] ≈ 7.25 rtol = 1.0e-14
    end
end

@testset "V&V overset: linear field reproduced exactly with barycentric weights" begin
    # Donor triangle with 3 explicit donor centers
    donor_centers = [
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]  # 2 × 3

    # Single receiver at the centroid of the triangle
    overset_centers = reshape([1 / 3, 1 / 3], 2, 1)

    donor_triplets = [(1, 2, 3)]
    receiver_idxs = [1]
    assembly = build_linear_donor_assembly(
        overset_centers, donor_centers, receiver_idxs, donor_triplets,
    )

    # Weights sum to 1
    @test sum(assembly.stencils[1].weights) ≈ 1.0 rtol = 1.0e-14
    # Centroid ⇒ all weights = 1/3
    for w in assembly.stencils[1].weights
        @test w ≈ 1 / 3 rtol = 1.0e-14
    end

    # Linear field φ(x, y) = x + 2y — exact at donor and receiver points
    phi_donor = Float64[
        donor_centers[1, d] + 2.0 * donor_centers[2, d]
            for d in 1:size(donor_centers, 2)
    ]
    phi_expected = overset_centers[1, 1] + 2.0 * overset_centers[2, 1]

    phi_receiver = zeros(Float64, 1)
    interpolate_overset!(phi_receiver, phi_donor, assembly)

    @test phi_receiver[1] ≈ phi_expected rtol = 1.0e-12
end

@testset "V&V overset: linear interpolation exact for interior receivers" begin
    # Donor mesh: triangle vertices at (0,0), (2,0), (0,2)
    donor_centers = [
        0.0 2.0 0.0
        0.0 0.0 2.0
    ]
    # Three receivers inside the triangle
    overset_centers = [
        0.25 1.0  0.5
        0.25 0.25 1.0
    ]
    donor_triplets = [(1, 2, 3), (1, 2, 3), (1, 2, 3)]
    receiver_idxs = [1, 2, 3]
    assembly = build_linear_donor_assembly(
        overset_centers, donor_centers, receiver_idxs, donor_triplets,
    )

    # Linear field φ(x, y) = 3x − 0.5y + 2
    linear(x, y) = 3 * x - 0.5 * y + 2
    phi_donor = Float64[linear(donor_centers[1, d], donor_centers[2, d]) for d in 1:3]
    phi_expected = Float64[linear(overset_centers[1, r], overset_centers[2, r]) for r in 1:3]

    phi_receiver = zeros(Float64, 3)
    interpolate_overset!(phi_receiver, phi_donor, assembly)

    for r in 1:3
        @test phi_receiver[r] ≈ phi_expected[r] rtol = 1.0e-12
    end
    # Weights sum to 1
    for s in assembly.stencils
        @test sum(s.weights) ≈ 1.0 rtol = 1.0e-14
    end
end

@testset "V&V overset: receiver mask correctly flags fringe cells" begin
    donor_centers = zeros(Float64, 2, 4)
    donor_centers[1, :] = [0.0, 1.0, 0.0, 1.0]
    donor_centers[2, :] = [0.0, 0.0, 1.0, 1.0]

    overset_centers = zeros(Float64, 2, 5)
    receiver_idxs = [1, 3, 5]  # cells 1, 3, 5 are fringe; 2, 4 are interior
    assembly = build_nearest_neighbour_assembly(
        overset_centers, donor_centers, receiver_idxs,
    )

    @test is_receiver(assembly, 1)
    @test !is_receiver(assembly, 2)
    @test is_receiver(assembly, 3)
    @test !is_receiver(assembly, 4)
    @test is_receiver(assembly, 5)
    @test_throws ErrorException is_receiver(assembly, 0)
    @test_throws ErrorException is_receiver(assembly, 6)
end

@testset "V&V overset: interpolate leaves non-receivers untouched" begin
    donor_centers = [
        0.0 1.0
        0.0 1.0
    ]
    overset_centers = [
        0.25 0.75 0.5
        0.25 0.75 0.5
    ]
    receiver_idxs = [1, 3]  # skip cell 2
    assembly = build_nearest_neighbour_assembly(
        overset_centers, donor_centers, receiver_idxs,
    )

    phi_donor = [0.0, 1.0]
    phi_receiver = fill(42.0, 3)
    interpolate_overset!(phi_receiver, phi_donor, assembly)

    # Cell 2 should be untouched.
    @test phi_receiver[2] == 42.0
    # Cells 1 and 3 should have been overwritten.
    @test phi_receiver[1] != 42.0
    @test phi_receiver[3] != 42.0
end

@testset "V&V overset: validation on assembly construction" begin
    # Weights not summing to 1 should error in assembly construction
    bad_stencil = DonorStencil{Float64}([1, 2], [0.3, 0.3])
    @test_throws ErrorException OversetAssembly{Float64}(
        [1], [bad_stencil], [true],
    )
    # Mismatched receivers / stencils lengths
    good_stencil = DonorStencil{Float64}([1], [1.0])
    @test_throws ErrorException OversetAssembly{Float64}(
        [1, 2], [good_stencil], [true, true],
    )
end

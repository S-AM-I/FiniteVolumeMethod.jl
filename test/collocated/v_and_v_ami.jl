# test/v_and_v_ami.jl — Arbitrary Mesh Interface (AMI) flux conservation V&V
#
# Invariants checked:
# 1. Uniform donor flux ⇒ uniform receiver flux (rtol 1e-12)
# 2. Σ(φ_donor · A_donor_overlap) == Σ(φ_receiver · A_receiver_overlap)  (rtol 1e-10)
# 3. Non-overlapping pair ⇒ zero flux transferred
# 4. Full overlap pair (1:1 matching) ⇒ donor value passes through unchanged

using Test

include(joinpath(@__DIR__, "..", "..", "src", "collocated", "dynamic_mesh", "ami.jl"))

@testset "V&V AMI: uniform donor ⇒ uniform receiver (non-conformal pairing)" begin
    # 3 donor faces of area 1, 1, 1. Two receiver faces of area 1.5, 1.5.
    # Each receiver face overlaps 1.5 worth of donor area split as 1.0 + 0.5.
    donor_areas = [1.0, 1.0, 1.0]
    receiver_areas = [1.5, 1.5]
    pairs = AMIFacePair{Float64}[
        AMIFacePair{Float64}(1, 1, 1.0),
        AMIFacePair{Float64}(2, 1, 0.5),
        AMIFacePair{Float64}(2, 2, 0.5),
        AMIFacePair{Float64}(3, 2, 1.0),
    ]
    ami = AMIInterface{Float64}(pairs, donor_areas, receiver_areas)

    phi_donor = fill(3.7, 3)
    phi_receiver = zeros(Float64, 2)
    project_ami_flux!(phi_receiver, phi_donor, ami)

    @test phi_receiver[1] ≈ 3.7 rtol = 1.0e-12
    @test phi_receiver[2] ≈ 3.7 rtol = 1.0e-12
end

@testset "V&V AMI: integrated flux conserved across interface" begin
    # Varied donor flux, verify Σ φ_d · overlap == Σ φ_r · overlap after projection.
    donor_areas = [1.0, 1.0, 1.0]
    receiver_areas = [1.5, 1.5]
    pairs = AMIFacePair{Float64}[
        AMIFacePair{Float64}(1, 1, 1.0),
        AMIFacePair{Float64}(2, 1, 0.5),
        AMIFacePair{Float64}(2, 2, 0.5),
        AMIFacePair{Float64}(3, 2, 1.0),
    ]
    ami = AMIInterface{Float64}(pairs, donor_areas, receiver_areas)

    phi_donor = [1.0, 2.0, 3.0]
    phi_receiver = zeros(Float64, 2)
    project_ami_flux!(phi_receiver, phi_donor, ami)

    integ_donor = ami_flux_integral_over_overlaps(phi_donor, ami; side = :donor)
    integ_recv = ami_flux_integral_over_overlaps(phi_receiver, ami; side = :receiver)

    @test integ_recv ≈ integ_donor rtol = 1.0e-10
end

@testset "V&V AMI: non-overlapping face pair ⇒ zero flux transferred" begin
    # Two receiver faces: the second has NO overlap with any donor.
    donor_areas = [1.0, 1.0]
    receiver_areas = [1.0, 1.0]
    pairs = AMIFacePair{Float64}[
        AMIFacePair{Float64}(1, 1, 1.0),
        AMIFacePair{Float64}(2, 1, 0.0),
    ]
    ami = AMIInterface{Float64}(pairs, donor_areas, receiver_areas)

    phi_donor = [5.0, 5.0]
    phi_receiver = fill(99.0, 2)  # untouched receivers keep initial value
    project_ami_flux!(phi_receiver, phi_donor, ami)

    @test phi_receiver[1] ≈ 5.0 rtol = 1.0e-14
    # Receiver 2 has no overlap with any donor ⇒ left at initial value (no pair).
    @test phi_receiver[2] == 99.0
end

@testset "V&V AMI: single pair with zero overlap collapses flux to 0" begin
    # A receiver that is paired ONLY with a zero-overlap donor must receive 0.
    donor_areas = [1.0]
    receiver_areas = [1.0]
    pairs = AMIFacePair{Float64}[AMIFacePair{Float64}(1, 1, 0.0)]
    ami = AMIInterface{Float64}(pairs, donor_areas, receiver_areas)

    phi_donor = [5.0]
    phi_receiver = [42.0]
    project_ami_flux!(phi_receiver, phi_donor, ami)

    @test phi_receiver[1] == 0.0
end

@testset "V&V AMI: full 1:1 overlap passes donor value through exactly" begin
    # Identity matching — each donor paired 1:1 with a receiver via full overlap.
    ami = build_matching_ami(5)
    phi_donor = [1.0, 2.0, 3.0, 4.0, 5.0]
    phi_receiver = zeros(Float64, 5)
    project_ami_flux!(phi_receiver, phi_donor, ami)

    for f in 1:5
        @test phi_receiver[f] ≈ phi_donor[f] rtol = 1.0e-14
    end

    # Integrated flux on both sides equals dot(phi_donor, ones) for unit areas
    integ_donor = ami_flux_integral_over_overlaps(phi_donor, ami; side = :donor)
    integ_recv = ami_flux_integral_over_overlaps(phi_receiver, ami; side = :receiver)
    @test integ_donor ≈ integ_recv rtol = 1.0e-14
    @test integ_donor ≈ sum(phi_donor) rtol = 1.0e-14
end

@testset "V&V AMI: ami_flux_integral helper" begin
    phi = [1.0, 2.0, 3.0]
    areas = [0.5, 1.0, 2.0]
    @test ami_flux_integral(phi, areas) ≈ 0.5 + 2.0 + 6.0 rtol = 1.0e-14
    @test_throws ErrorException ami_flux_integral([1.0], areas)
end

@testset "V&V AMI: validation rejects bad overlaps / indices" begin
    # Negative overlap
    @test_throws ErrorException AMIFacePair{Float64}(1, 1, -0.1)
    # Out-of-range donor index
    bad_pair = AMIFacePair{Float64}(5, 1, 1.0)
    @test_throws ErrorException AMIInterface{Float64}([bad_pair], [1.0], [1.0])
    # Negative donor area
    good_pair = AMIFacePair{Float64}(1, 1, 1.0)
    @test_throws ErrorException AMIInterface{Float64}([good_pair], [-1.0], [1.0])
end

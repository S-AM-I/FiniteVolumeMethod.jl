# test/v_and_v_mrf_single_zone.jl — Single-zone MRF Coriolis+centrifugal V&V
#
# Algebraic gates for the `MRFZone`-based force primitives
# (`coriolis_force`, `centrifugal_force`, `mrf_cell_source`,
# `add_mrf_source!`). Verifies:
#
#   1. ω = 0 ⇒ both forces are zero (no rotation, no pseudo-forces).
#   2. U = 0 ⇒ Coriolis vanishes (but centrifugal need not).
#   3. r = 0 ⇒ centrifugal vanishes (on the rotation axis).
#   4. Coriolis ⊥ U and Coriolis ⊥ ω (cross-product geometry).
#   5. Centrifugal points radially outward from the rotation axis.
#   6. Closed-form check at a reference (ω, U, r, ρ) sample, rtol 1e-14.
#   7. ρ-linear scaling.
#   8. ω-linear scaling of Coriolis, ω-quadratic scaling of centrifugal.
#   9. `add_mrf_source!` leaves out-of-zone cells untouched.

using FiniteVolumeMethod
using FiniteVolumeMethod: MRFZone
using FiniteVolumeMethod: MRFZone, coriolis_force, centrifugal_force,
    mrf_cell_source, add_mrf_source!
using LinearAlgebra: dot, cross, norm
using StaticArrays: SVector
using Test

include("TestHelpers.jl")

@testset "V&V MRF single zone: omega = 0 gives zero force" begin
    omega = SVector(0.0, 0.0, 0.0)
    U = SVector(3.0, -1.5, 2.7)
    r = SVector(1.0, 2.0, -0.5)
    rho = 1.2

    @test coriolis_force(omega, U, rho) ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-14
    @test centrifugal_force(omega, r, rho) ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-14
end

@testset "V&V MRF single zone: U = 0 kills Coriolis only" begin
    omega = SVector(0.0, 0.0, 5.0)
    U = SVector(0.0, 0.0, 0.0)
    r = SVector(1.0, 0.0, 0.0)
    rho = 1.0

    @test coriolis_force(omega, U, rho) ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-14
    # Centrifugal at r = (1,0,0), ω = (0,0,5): -ρ (ω × (ω × r))
    # ω × r = (0, 5, 0), ω × (ω × r) = (-25, 0, 0), so source = (25, 0, 0).
    cent = centrifugal_force(omega, r, rho)
    @test cent ≈ SVector(25.0, 0.0, 0.0) rtol = 1.0e-14
    @test norm(cent) > 0
end

@testset "V&V MRF single zone: r = 0 kills centrifugal only" begin
    omega = SVector(0.0, 0.0, 3.0)
    U = SVector(1.0, 2.0, 0.0)
    r = SVector(0.0, 0.0, 0.0)
    rho = 1.0

    @test centrifugal_force(omega, r, rho) ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-14
    @test norm(coriolis_force(omega, U, rho)) > 0
end

@testset "V&V MRF single zone: Coriolis perpendicular to U and omega" begin
    # Generic triad.
    omega = SVector(1.0, 2.0, -0.5)
    U = SVector(0.7, -1.3, 2.1)
    rho = 1.4

    F_cor = coriolis_force(omega, U, rho)
    # F_cor = -2 ρ (ω × U) must be perpendicular to both ω and U.
    @test abs(dot(F_cor, omega)) < 1.0e-12
    @test abs(dot(F_cor, U)) < 1.0e-12
end

@testset "V&V MRF single zone: centrifugal points radially outward" begin
    # Planar rotation: ω along z, r in xy-plane. Centrifugal on such a
    # point equals +ρ ω² r (outward from axis).
    omega = SVector(0.0, 0.0, 4.0)
    r = SVector(0.5, 0.8, 0.0)
    rho = 1.0

    F_cent = centrifugal_force(omega, r, rho)
    # Projection onto r̂ must be positive (radially outward).
    r_hat = r / norm(r)
    @test dot(F_cent, r_hat) > 0
    # Closed form: F_cent = ρ ω² r.
    @test F_cent ≈ rho * 16.0 * r rtol = 1.0e-14
end

@testset "V&V MRF single zone: closed-form reference sample" begin
    # Fully 3D reference. Pick values with non-zero components on every axis
    # so every index of the cross product matters.
    omega = SVector(1.0, -2.0, 3.0)
    U = SVector(4.0, 5.0, -6.0)
    r = SVector(-1.0, 2.0, 0.5)
    rho = 0.8

    # ω × U = (1,-2,3) × (4,5,-6) = (-2·-6 − 3·5, 3·4 − 1·-6, 1·5 − -2·4)
    #        = (12−15, 12+6, 5+8) = (-3, 18, 13)
    expected_cor = -2 * rho * SVector(-3.0, 18.0, 13.0)
    @test coriolis_force(omega, U, rho) ≈ expected_cor rtol = 1.0e-14

    # ω × r = (1,-2,3) × (-1,2,0.5) = (-2·0.5 − 3·2, 3·-1 − 1·0.5, 1·2 − -2·-1)
    #        = (-1 − 6, -3 − 0.5, 2 − 2) = (-7, -3.5, 0)
    # ω × (ω × r) = (1,-2,3) × (-7,-3.5,0)
    #             = (-2·0 − 3·-3.5, 3·-7 − 1·0, 1·-3.5 − -2·-7)
    #             = (10.5, -21, -3.5 − 14) = (10.5, -21, -17.5)
    expected_cent = -rho * SVector(10.5, -21.0, -17.5)
    @test centrifugal_force(omega, r, rho) ≈ expected_cent rtol = 1.0e-14

    # Combined per-cell source through `mrf_cell_source`.
    zone = MRFZone{Float64}(omega, SVector(0.0, 0.0, 0.0), [1])
    total = mrf_cell_source(zone, r, U, rho)
    @test total ≈ expected_cor + expected_cent rtol = 1.0e-14
end

@testset "V&V MRF single zone: rho-linear scaling" begin
    omega = SVector(0.3, 0.7, -1.1)
    U = SVector(1.0, 0.0, 0.5)
    r = SVector(0.2, -0.4, 0.6)

    F_cor_1 = coriolis_force(omega, U, 1.0)
    F_cor_3 = coriolis_force(omega, U, 3.0)
    @test F_cor_3 ≈ 3.0 * F_cor_1 rtol = 1.0e-14

    F_cent_1 = centrifugal_force(omega, r, 1.0)
    F_cent_5 = centrifugal_force(omega, r, 5.0)
    @test F_cent_5 ≈ 5.0 * F_cent_1 rtol = 1.0e-14
end

@testset "V&V MRF single zone: omega scaling laws" begin
    # Coriolis is linear in ω (at fixed U): F(aω) = a F(ω).
    U = SVector(1.0, 2.0, 3.0)
    rho = 1.2
    omega = SVector(0.5, -0.3, 1.1)
    F1 = coriolis_force(omega, U, rho)
    F2 = coriolis_force(2.0 * omega, U, rho)
    @test F2 ≈ 2.0 * F1 rtol = 1.0e-14

    # Centrifugal is quadratic in ω (at fixed r): F(aω) = a² F(ω).
    r = SVector(0.7, -0.2, 1.4)
    G1 = centrifugal_force(omega, r, rho)
    G2 = centrifugal_force(2.0 * omega, r, rho)
    @test G2 ≈ 4.0 * G1 rtol = 1.0e-14
end

@testset "V&V MRF single zone: add_mrf_source! respects zone membership" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    n = length(mesh.cell_volumes)

    # Rotate about the cell (2, 2) centre, ω = 5 rad/s out of plane.
    cell_in = [1, 2, 3]
    zone = MRFZone{Float64}(
        SVector(0.0, 0.0, 5.0),
        SVector(0.5, 0.5, 0.0),
        cell_in,
    )

    source = fill(SVector(0.0, 0.0, 0.0), n)
    U = [SVector(1.0, 0.0, 0.0) for _ in 1:n]
    rho = 1.3

    add_mrf_source!(source, U, mesh, zone, rho)

    # Cells outside the zone untouched.
    for c in 1:n
        c in cell_in && continue
        @test source[c] ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-14
    end

    # Cells in the zone match `mrf_cell_source` at their x_cell.
    for c in cell_in
        xc = SVector(mesh.cell_centers[1, c], mesh.cell_centers[2, c], 0.0)
        expected = mrf_cell_source(zone, xc, U[c], rho)
        @test source[c] ≈ expected rtol = 1.0e-14
    end
end

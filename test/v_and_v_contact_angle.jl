# test/v_and_v_contact_angle.jl — static + Cox-Voinov contact angles (v3.92)
#
# Covers the wall-adhesion helpers added to `surface_tension.jl` in
# Wave-1 Agent D:
#
#   * `StaticContactAngle` — rotates the interface normal n' to
#       n' = cos(θ_s) · n_wall + sin(θ_s) · t_wall
#     with special limits at θ_s = 0° (n' ≡ t_wall) and 180° (n' ≡ -t_wall).
#
#   * `DynamicContactAngle` — Cox-Voinov θ³ = θ_s³ + 9·Ca·ln(L/L_s).
#
# Invariants:
#
#   Static:
#     1. θ_s = π/2 ⇒ n' = n_wall (no correction in the tangential plane).
#     2. θ_s = 0   ⇒ n' = t_wall (complete wetting).
#     3. θ_s = π   ⇒ n' = -t_wall (complete dewetting).
#
#   Dynamic:
#     4. Ca = 0 recovers θ_s exactly (rtol 1e-12).
#     5. Monotone in Ca at fixed θ_s.
#     6. Closed-form agreement at three sample Ca values (rtol 1e-12).

using FiniteVolumeMethod
using LinearAlgebra
using StaticArrays
using Test

include("TestHelpers.jl")

const _apply_contact_angle = FiniteVolumeMethod.apply_contact_angle
const _cox_voinov_angle = FiniteVolumeMethod.cox_voinov_angle
const StaticContactAngle = FiniteVolumeMethod.StaticContactAngle
const DynamicContactAngle = FiniteVolumeMethod.DynamicContactAngle

@testset "V&V: static contact angle — θ_s = 90° leaves n_wall unchanged" begin
    n_wall = SVector{2, Float64}(0.0, 1.0)
    n_interface = SVector{2, Float64}(1.0, 1.0) / sqrt(2.0)
    model = StaticContactAngle(π / 2)

    n_corr = _apply_contact_angle(n_interface, n_wall, model)
    # With θ_s = 90°, cos(θ) ≈ 0 and sin(θ) = 1 → n_corr aligns with
    # the tangent component of n_interface (perpendicular to n_wall).
    # The tangent component of (1,1)/√2 relative to (0,1) is (1,0).
    @test isapprox(n_corr[1], 1.0; atol = 1.0e-12)
    @test isapprox(n_corr[2], 0.0; atol = 1.0e-12)
end

@testset "V&V: static contact angle — θ_s = 0° ⇒ n' = n_wall (collinear with wall)" begin
    # With θ_s = 0 the CSF normal is prescribed to be aligned with
    # the wall normal (the interface plane is parallel to the wall
    # tangent, so the surface normal = wall normal).
    n_wall = SVector{2, Float64}(0.0, 1.0)
    n_interface = SVector{2, Float64}(1.0, 1.0) / sqrt(2.0)
    model = StaticContactAngle(0.0)

    n_corr = _apply_contact_angle(n_interface, n_wall, model)
    @test isapprox(n_corr[1], 0.0; atol = 1.0e-12)
    @test isapprox(n_corr[2], 1.0; atol = 1.0e-12)
end

@testset "V&V: static contact angle — θ_s = 180° ⇒ n' = -n_wall (complete dewetting)" begin
    n_wall = SVector{2, Float64}(0.0, 1.0)
    n_interface = SVector{2, Float64}(1.0, 1.0) / sqrt(2.0)
    model = StaticContactAngle(Float64(π))

    n_corr = _apply_contact_angle(n_interface, n_wall, model)
    @test isapprox(n_corr[1], 0.0; atol = 1.0e-12)
    @test isapprox(n_corr[2], -1.0; atol = 1.0e-12)
end

@testset "V&V: static contact angle — result is a unit vector" begin
    n_wall = SVector{2, Float64}(0.0, 1.0)
    n_interface = SVector{2, Float64}(0.6, 0.8)
    for theta_s in (0.0, π / 6, π / 4, π / 3, π / 2, 2π / 3, 5π / 6, Float64(π))
        model = StaticContactAngle(Float64(theta_s))
        n_corr = _apply_contact_angle(n_interface, n_wall, model)
        @test isapprox(norm(n_corr), 1.0; rtol = 1.0e-12)
    end
end

@testset "V&V: Cox-Voinov — Ca = 0 ⇒ θ = θ_s (rtol 1e-12)" begin
    theta_s = π / 4
    model = DynamicContactAngle(theta_s, 1.0e-3, 0.072, 1.0e-3, 1.0e-9)
    theta = _cox_voinov_angle(model, 0.0)
    @test isapprox(theta, theta_s; rtol = 1.0e-12, atol = 1.0e-14)
end

@testset "V&V: Cox-Voinov — monotone in Ca at fixed θ_s" begin
    # ln(L/L_s) > 0 (L > L_s) → θ(Ca) is monotonically increasing.
    theta_s = π / 4
    mu = 1.0e-3
    sigma = 0.072
    L = 1.0e-3
    L_s = 1.0e-9
    model = DynamicContactAngle(theta_s, mu, sigma, L, L_s)

    Us = 0.0:0.01:0.1
    prev = _cox_voinov_angle(model, first(Us))
    for u in Us[2:end]
        theta = _cox_voinov_angle(model, u)
        @test theta >= prev - 1.0e-14
        prev = theta
    end
end

@testset "V&V: Cox-Voinov — closed-form match at three sample Ca values" begin
    theta_s = π / 4
    mu = 1.0e-3
    sigma = 0.05
    L = 1.0e-3
    L_s = 1.0e-9
    model = DynamicContactAngle(theta_s, mu, sigma, L, L_s)

    for U_cl in (0.01, 0.05, 0.2)
        Ca = mu * U_cl / sigma
        expected = cbrt(theta_s^3 + 9 * Ca * log(L / L_s))
        theta = _cox_voinov_angle(model, U_cl)
        @test isapprox(theta, expected; rtol = 1.0e-12, atol = 1.0e-14)
    end
end

@testset "V&V: dynamic contact angle → apply_contact_angle uses Ca-corrected θ" begin
    # At Ca = 0 the dynamic apply_contact_angle must match the static
    # application with θ = θ_s. At Ca > 0 it should differ.
    n_wall = SVector{2, Float64}(0.0, 1.0)
    n_interface = SVector{2, Float64}(0.8, 0.6)
    theta_s = π / 3
    model_d = DynamicContactAngle(theta_s, 1.0e-3, 0.072, 1.0e-3, 1.0e-9)
    model_s = StaticContactAngle(theta_s)

    n_d0 = _apply_contact_angle(n_interface, n_wall, model_d; U_cl = 0.0)
    n_s = _apply_contact_angle(n_interface, n_wall, model_s)
    @test isapprox(n_d0[1], n_s[1]; rtol = 1.0e-12)
    @test isapprox(n_d0[2], n_s[2]; rtol = 1.0e-12)

    n_d1 = _apply_contact_angle(n_interface, n_wall, model_d; U_cl = 0.1)
    # Different Ca ⇒ different rotation angle ⇒ different result.
    @test !isapprox(n_d1[1], n_s[1]; atol = 1.0e-8)
end

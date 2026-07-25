# test/v_and_v_primary_breakup.jl — Primary-atomisation V&V
#
# Algebraic verification of Reitz's KH-ACT correlation and the LISA
# sheet-break-up correlation. Invariants checked:
#
# KH-ACT:
#   - Zero relative velocity ⇒ breakup time is infinite (no break-up)
#   - Child diameter monotone decreasing in `|U_rel|` at fixed fluid properties
#   - Closed-form check at three (We_g, Z) sample points (rtol 1e-12)
#
# LISA:
#   - Λ_LISA is linear in σ
#   - Λ_LISA is inversely proportional to U²

using LinearAlgebra: norm
using FiniteVolumeMethod: LagrangianParticle
using StaticArrays
using Test

# primary_breakup.jl has no dependencies on LagrangianParticle; it ships
# only algebraic helpers, so we include it directly.
include(joinpath(@__DIR__, "..", "..", "src", "collocated", "lagrangian", "primary_breakup.jl"))

@testset "V&V: KH-ACT — zero slip ⇒ no break-up" begin
    d_child, τ = kh_act_breakup(1.0e-4, 0.0, 1.2, 1000.0, 1.0e-3, 0.072)
    @test d_child == 1.0e-4
    @test τ == Inf
end

@testset "V&V: KH-ACT — monotone in We_g" begin
    # Raise the slip velocity and verify monotone decrease in child d.
    d_parent = 1.0e-4
    ρ_g = 1.2
    ρ_l = 1000.0
    μ_l = 1.0e-3
    σ = 0.072
    d_prev = d_parent
    τ_prev = Inf
    for U in (50.0, 75.0, 100.0, 150.0, 200.0, 300.0)
        d_child, τ = kh_act_breakup(d_parent, U, ρ_g, ρ_l, μ_l, σ)
        @test d_child <= d_prev
        @test τ <= τ_prev
        d_prev = d_child
        τ_prev = τ
    end
end

@testset "V&V: KH-ACT — closed-form algebra at sample points" begin
    # Recompute the correlation from scratch and compare.
    function ref_kh_act(d, U, ρ_g, ρ_l, μ_l, σ)
        a = d / 2
        We_g = ρ_g * U^2 * a / σ
        We_l = ρ_l * U^2 * a / σ
        Re_l = ρ_l * U * a / μ_l
        Z = sqrt(We_l) / Re_l
        Ta = Z * sqrt(We_g)
        Λ = a * 9.02 * (1 + 0.45 * sqrt(Z)) * (1 + 0.4 * Ta^0.7) /
            (1 + 0.87 * We_g^1.67)^0.6
        Ω_dim = (0.34 + 0.38 * We_g^1.5) / ((1 + Z) * (1 + 1.4 * Ta^0.6))
        Ω = Ω_dim / sqrt(ρ_l * a^3 / σ)
        r_child_raw = 0.61 * Λ
        r_child = min(r_child_raw, a)
        τ = 3.726 * 1.73 * a / (Λ * Ω)
        return 2 * r_child, τ, We_g, Z
    end

    samples = (
        (d = 1.0e-4, U = 100.0, ρ_g = 1.2, ρ_l = 1000.0, μ_l = 1.0e-3, σ = 0.072),
        (d = 1.0e-4, U = 150.0, ρ_g = 1.2, ρ_l = 1000.0, μ_l = 1.0e-3, σ = 0.072),
        (d = 2.0e-4, U = 200.0, ρ_g = 1.2, ρ_l = 800.0, μ_l = 2.0e-3, σ = 0.03),
    )
    for s in samples
        d_ref, τ_ref, We_ref, Z_ref = ref_kh_act(s.d, s.U, s.ρ_g, s.ρ_l, s.μ_l, s.σ)
        d_got, τ_got = kh_act_breakup(s.d, s.U, s.ρ_g, s.ρ_l, s.μ_l, s.σ)
        @test isapprox(d_got, d_ref; rtol = 1.0e-12)
        @test isapprox(τ_got, τ_ref; rtol = 1.0e-12)
    end
end

@testset "V&V: KH-ACT — SVector dispatch matches scalar" begin
    U_mag = 120.0
    U_vec = SVector(120.0, 0.0, 0.0)
    d_scalar, τ_scalar = kh_act_breakup(1.0e-4, U_mag, 1.2, 1000.0, 1.0e-3, 0.072)
    d_vec, τ_vec = kh_act_breakup(1.0e-4, U_vec, 1.2, 1000.0, 1.0e-3, 0.072)
    @test isapprox(d_scalar, d_vec; rtol = 1.0e-14)
    @test isapprox(τ_scalar, τ_vec; rtol = 1.0e-14)
end

@testset "V&V: KH-ACT — dimensionless group sanity" begin
    Λ, Ω, We_g, Z, Ta = kh_act_wavelength_growth(
        1.0e-4, 100.0, 1.2, 1000.0, 1.0e-3, 0.072,
    )
    # Expected We_g = ρ_g·U²·a/σ = 1.2·10⁴·5e-5/0.072 ≈ 8.333...
    @test isapprox(We_g, 8.333333333333334; rtol = 1.0e-12)
    @test Ta > 0 && Z > 0
    @test isfinite(Λ) && Ω > 0
end

@testset "V&V: LISA — Λ ∝ σ (linear in surface tension)" begin
    h = 5.0e-5
    U = 50.0
    ρ_g = 1.2
    Λ1 = lisa_wavelength(h, U, ρ_g, 0.03)
    Λ2 = lisa_wavelength(h, U, ρ_g, 0.06)
    # With σ doubled, both the leading σ and the We_g change. We use the
    # limiting ratio by subtracting the σ-free part. Simpler invariant:
    # Λ_LISA = 2π σ (1 + We_g)/(ρ_g U²) = 2π σ /(ρ_g U²) + 2π h / ρ_g U² * (ρ_g/(ρ_g))·... wait.
    # We_g = ρ_g U² h / σ, so (1 + We_g)·σ = σ + ρ_g U² h; therefore
    # Λ_LISA = 2π (σ + ρ_g U² h)/(ρ_g U²) — LINEAR in σ with slope 2π/(ρ_g U²).
    slope = (Λ2 - Λ1) / (0.06 - 0.03)
    slope_expected = 2π / (ρ_g * U^2)
    @test isapprox(slope, slope_expected; rtol = 1.0e-12)
end

@testset "V&V: LISA — Λ ∝ 1/U² scaling" begin
    # Λ = 2π σ (1 + We_g) / (ρ_g U²) with We_g = ρ_g U² h / σ, so
    # Λ = 2π σ / (ρ_g U²) + 2π h.  The h-piece is U-independent; the
    # σ-piece scales as 1/U². Verify the σ-piece ratio.
    h = 5.0e-5
    ρ_g = 1.2
    σ = 0.03
    Λ50 = lisa_wavelength(h, 50.0, ρ_g, σ)
    Λ100 = lisa_wavelength(h, 100.0, ρ_g, σ)
    const_part = 2π * h
    piece_50 = Λ50 - const_part
    piece_100 = Λ100 - const_part
    @test isapprox(piece_50 / piece_100, 4.0; rtol = 1.0e-12)
end

@testset "V&V: LISA — zero U returns Inf" begin
    @test lisa_wavelength(1.0e-4, 0.0, 1.2, 0.03) == Inf
    d_child, τ = lisa_breakup(1.0e-4, 0.0, 1.2, 1000.0, 0.03)
    @test d_child == Inf
    @test τ == Inf
end

@testset "V&V: LISA — monotone in |U_rel|" begin
    # The σ-piece decreases as 1/U²; the h-piece is constant. So Λ is
    # monotone decreasing in U.
    Λ_prev = Inf
    for U in (10.0, 50.0, 100.0, 200.0)
        Λ = lisa_wavelength(5.0e-5, U, 1.2, 0.03)
        @test Λ < Λ_prev
        Λ_prev = Λ
    end
end

# test/v_and_v_drag_closures.jl — Ishii-Zuber / Gibilaro drag V&V (v3.0 Wave 5)
#
# Algebraic / closed-form verification of the Eulerian two-fluid
# interphase drag closures. All invariants here are primitive (no
# mesh, no solver): the tests confirm the correlations match the
# textbook form at both asymptotic limits and at hand-computed
# sample points.
#
# Closures covered:
#   * `IshiiZuberDrag` — bubbly, `Re_b ∈ [1, 1000]`,
#       C_D = (24/Re_b)·(1 + 0.1·Re_b^0.75)·(1 − α_g)^(−1.5)
#   * `GibilaroDrag`  — denser clusters, (1 − α_g)^(−2.65) exponent.
#
# Evidence toward v3.1 promotion of the two-fluid module from
# `experimental` to `smoke_tested`.

using FiniteVolumeMethod
using StaticArrays
using Test

# The Wave-5 two-fluid files are owned by this agent and wired into
# the package by the main thread in a later pass. Include them
# directly so this V&V file is runnable standalone.
const _WAVE5_SRC = joinpath(@__DIR__, "..", "..", "src", "collocated", "multiphase")
isdefined(Main, :AbstractDragClosure) || include(joinpath(_WAVE5_SRC, "drag_closures.jl"))

@testset "V&V: drag closures — zero slip gives zero force" begin
    rho_l = 1000.0
    mu_l = 1.0e-3
    d_b = 1.0e-3
    alpha_g = 0.1

    # Scalar U_rel = 0
    F_iz = drag_force_density(IshiiZuberDrag(), rho_l, 0.0, alpha_g, d_b, mu_l)
    F_gi = drag_force_density(GibilaroDrag(), rho_l, 0.0, alpha_g, d_b, mu_l)
    @test F_iz == 0.0
    @test F_gi == 0.0

    # Vector U_rel = (0, 0)
    F_iz_vec = drag_force_density(
        IshiiZuberDrag(), rho_l, SVector(0.0, 0.0), alpha_g, d_b, mu_l,
    )
    F_gi_vec = drag_force_density(
        GibilaroDrag(), rho_l, SVector(0.0, 0.0), alpha_g, d_b, mu_l,
    )
    @test F_iz_vec == SVector(0.0, 0.0)
    @test F_gi_vec == SVector(0.0, 0.0)
end

@testset "V&V: drag closures — alpha_g = 0 gives zero force" begin
    # No dispersed phase means no interphase momentum transfer.
    rho_l = 1000.0
    mu_l = 1.0e-3
    d_b = 1.0e-3
    U_rel = SVector(0.5, 0.2)

    F_iz = drag_force_density(IshiiZuberDrag(), rho_l, U_rel, 0.0, d_b, mu_l)
    F_gi = drag_force_density(GibilaroDrag(), rho_l, U_rel, 0.0, d_b, mu_l)
    @test F_iz == SVector(0.0, 0.0)
    @test F_gi == SVector(0.0, 0.0)
end

@testset "V&V: drag closures — Re_b → 0 Stokes limit (IshiiZuber)" begin
    # In the Re_b → 0 limit and α_g → 0, Ishii-Zuber reduces to
    # F = 18 μ α U_rel / d². Verify the computed drag approaches the
    # analytical Stokes form as slip → 0.
    rho_l = 1000.0
    mu_l = 1.0e-3
    d_b = 1.0e-3
    alpha_g = 1.0e-6  # near-zero to kill the (1-α)^(-1.5) correction

    for slip in (1.0e-7, 1.0e-6, 1.0e-5)
        U_rel = SVector(slip, 0.0)

        Re_b = rho_l * slip * d_b / mu_l
        @test Re_b <= 1.0e-2
        F_iz = drag_force_density(
            IshiiZuberDrag(), rho_l, U_rel, alpha_g, d_b, mu_l,
        )
        F_stokes = stokes_limit_drag(rho_l, U_rel, alpha_g, d_b, mu_l)
        # Inertial correction contributes ≤ 0.1·Re^0.75, ≤ 2 % at
        # Re = 1e-1. Cluster correction adds another O(alpha_g).
        @test isapprox(F_iz[1], F_stokes[1]; rtol = 0.05)
        @test isapprox(F_iz[2], F_stokes[2]; atol = 1.0e-18)
    end
end

@testset "V&V: drag closures — Re_b → ∞ asymptotic plateau" begin
    # At very large Re_b, C_D from the Ishii-Zuber form is dominated
    # by the inertial term but still decreases as 24·(0.1·Re^0.75)/Re
    # = 2.4·Re^(-0.25). It doesn't literally approach a constant, but
    # the *inertial coefficient* 0.1·Re^0.75 grows faster than the
    # 24/Re decay, so C_D eventually grows slowly. Our textbook
    # contract is that the correlation is valid for Re_b ∈ [1, 1000],
    # so verify the Re = 1000 value is in the expected band ≈ 0.44
    # ± cluster correction.
    alpha_g = 0.0   # no cluster correction
    C_D_at_1000 = drag_coefficient(IshiiZuberDrag(), 1000.0, alpha_g)
    # 24/1000 · (1 + 0.1·1000^0.75) = 0.024 · (1 + 17.78) ≈ 0.451
    @test isapprox(C_D_at_1000, 0.024 * (1 + 0.1 * 1000.0^0.75); rtol = 1.0e-12)
    @test 0.4 < C_D_at_1000 < 0.5
end

@testset "V&V: drag closures — closed-form match at sample points" begin
    # Hand-compute C_D and F_D at three (U_rel, α_g, Re_b) samples
    # and compare against the implementation at rtol = 1e-12.
    rho_l = 1000.0
    mu_l = 1.0e-3
    d_b = 1.0e-3

    samples = [
        (slip = 1.0e-3, alpha_g = 0.05),   # Re = 1.0, α = 5 %
        (slip = 1.0e-2, alpha_g = 0.2),   # Re = 10,  α = 20 %
        (slip = 1.0e-1, alpha_g = 0.4),   # Re = 100, α = 40 %
    ]
    for s in samples
        slip = s.slip
        alpha_g = s.alpha_g
        U_rel = SVector(slip, 0.0)

        Re_b = rho_l * slip * d_b / mu_l
        C_D_ref = (24.0 / Re_b) * (1.0 + 0.1 * Re_b^0.75) * (1.0 - alpha_g)^(-1.5)
        prefactor = 0.75 * C_D_ref * rho_l * alpha_g * slip / d_b
        F_ref = prefactor * U_rel

        C_D_impl = drag_coefficient(IshiiZuberDrag(), Re_b, alpha_g)
        F_impl = drag_force_density(
            IshiiZuberDrag(), rho_l, U_rel, alpha_g, d_b, mu_l,
        )

        @test isapprox(C_D_impl, C_D_ref; rtol = 1.0e-12)
        @test isapprox(F_impl[1], F_ref[1]; rtol = 1.0e-12)
        @test isapprox(F_impl[2], F_ref[2]; atol = 1.0e-18)

        # Gibilaro closure: same structure, (1 - α)^(-2.65) exponent.
        C_D_gi_ref = (24.0 / Re_b) * (1.0 + 0.1 * Re_b^0.75) * (1.0 - alpha_g)^(-2.65)
        C_D_gi_impl = drag_coefficient(GibilaroDrag(), Re_b, alpha_g)
        @test isapprox(C_D_gi_impl, C_D_gi_ref; rtol = 1.0e-12)
        @test C_D_gi_impl > C_D_impl   # denser cluster correction
    end
end

@testset "V&V: drag closures — bubble_reynolds sanity" begin
    rho_l = 1000.0
    mu_l = 1.0e-3
    d_b = 2.0e-3

    # Scalar slip
    Re1 = bubble_reynolds(rho_l, d_b, 0.5, mu_l)
    @test isapprox(Re1, 1000.0 * 2.0e-3 * 0.5 / 1.0e-3; rtol = 1.0e-12)

    # SVector slip |slip| = sqrt(0.3^2 + 0.4^2) = 0.5
    Re2 = bubble_reynolds(rho_l, d_b, SVector(0.3, 0.4), mu_l)
    @test isapprox(Re2, Re1; rtol = 1.0e-12)

    @test_throws ArgumentError bubble_reynolds(rho_l, 0.0, 0.5, mu_l)
    @test_throws ArgumentError bubble_reynolds(rho_l, d_b, 0.5, 0.0)
end

@testset "V&V: drag closures — invalid alpha_g guards" begin
    # α_g = 1 would zero-divide in the cluster correction; the
    # constructor must refuse anything ≥ 1.
    @test_throws ArgumentError drag_coefficient(IshiiZuberDrag(), 10.0, 1.0)
    @test_throws ArgumentError drag_coefficient(GibilaroDrag(), 10.0, 1.0)
    @test_throws ArgumentError drag_coefficient(IshiiZuberDrag(), 10.0, -0.01)
end

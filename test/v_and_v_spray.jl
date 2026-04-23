# test/v_and_v_spray.jl — Spray breakup algebra V&V (v3.45)
#
# Fourth convergence-verified benchmark for `lagrangian_dpm`,
# joining Stokes terminal velocity (v3.13), Schiller-Naumann
# drag (v3.26), and Ranz-Marshall heat transfer (v3.33).
# Covers the TAB droplet-breakup closure primitives:
#
#   We = ρ_f · |U_rel|² · d / σ            (Weber number)
#   should_breakup(We) = (We > We_crit)    (breakup criterion)
#   d_child = d_parent · (We_crit / We)^(1/3)   (child diameter)
#
# Seven algebraic invariants:
#
#   1. We = 0 when U_rel = 0.
#   2. We ∝ |U_rel|² at fixed d, ρ_f, σ.
#   3. We ∝ d (linear).
#   4. should_breakup returns true iff We > We_crit.
#   5. d_child = d_parent at We = We_crit.
#   6. d_child < d_parent for We > We_crit (breakup shrinks droplet).
#   7. d_child ∝ (1/We)^(1/3) at fixed d_parent.

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: Spray — Weber number ≡ 0 at zero slip" begin
    d = 100.0e-6
    rho_f = 1.2
    sigma = 0.072

    @test weber_number(SVector(0.0, 0.0), d, rho_f, sigma) == 0.0
    @test weber_number(0.0, d, rho_f, sigma) == 0.0
end

@testset "V&V: Spray — Weber number ∝ |U_rel|² scaling" begin
    d = 100.0e-6
    rho_f = 1.2
    sigma = 0.072

    We_1 = weber_number(SVector(1.0, 0.0), d, rho_f, sigma)
    We_2 = weber_number(SVector(2.0, 0.0), d, rho_f, sigma)
    We_3 = weber_number(SVector(3.0, 0.0), d, rho_f, sigma)

    # |U|² scales as 1, 4, 9.
    @test isapprox(We_2 / We_1, 4.0; rtol = 1.0e-12)
    @test isapprox(We_3 / We_1, 9.0; rtol = 1.0e-12)
end

@testset "V&V: Spray — Weber number ∝ d linear scaling" begin
    U_rel = SVector(5.0, 0.0)
    rho_f = 1.2
    sigma = 0.072

    We_1 = weber_number(U_rel, 100.0e-6, rho_f, sigma)
    We_2 = weber_number(U_rel, 200.0e-6, rho_f, sigma)
    We_3 = weber_number(U_rel, 400.0e-6, rho_f, sigma)

    @test isapprox(We_2 / We_1, 2.0; rtol = 1.0e-12)
    @test isapprox(We_3 / We_1, 4.0; rtol = 1.0e-12)
end

@testset "V&V: Spray — Weber closed-form against analytical" begin
    d = 50.0e-6
    rho_f = 1.0
    sigma = 0.05
    U_mag = 10.0

    We_expected = rho_f * U_mag^2 * d / sigma
    We_computed = weber_number(SVector(U_mag, 0.0), d, rho_f, sigma)

    @test isapprox(We_computed, We_expected; rtol = 1.0e-14)
end

@testset "V&V: Spray — TAB should_breakup threshold" begin
    tab = TABBreakup(; We_crit = 12.0)

    @test should_breakup(tab, 5.0) == false
    @test should_breakup(tab, 11.99) == false
    @test should_breakup(tab, 12.0) == false       # strict >
    @test should_breakup(tab, 12.01) == true
    @test should_breakup(tab, 100.0) == true
end

@testset "V&V: Spray — TAB child diameter at We = We_crit ⇒ d_child = d_parent" begin
    tab = TABBreakup(; We_crit = 12.0)
    d_parent = 100.0e-6

    d_child = breakup_diameter(tab, d_parent, 12.0)
    @test isapprox(d_child, d_parent; rtol = 1.0e-14)
end

@testset "V&V: Spray — TAB child diameter shrinks at We > We_crit" begin
    tab = TABBreakup(; We_crit = 12.0)
    d_parent = 100.0e-6

    # d_child = d_parent · (We_crit / We)^(1/3)
    for We in (15.0, 30.0, 60.0, 120.0)
        d_child = breakup_diameter(tab, d_parent, We)
        expected = d_parent * (12.0 / We)^(1 / 3)
        @test isapprox(d_child, expected; rtol = 1.0e-14)
        @test d_child < d_parent   # breakup shrinks
    end
end

@testset "V&V: Spray — TAB d_child^3 · We = d_parent^3 · We_crit invariant" begin
    # Algebraic rearrangement: d_child = d_parent · (We_crit/We)^(1/3)
    # ⟹ d_child^3 · We = d_parent^3 · We_crit (independent of We).
    tab = TABBreakup(; We_crit = 12.0)
    d_parent = 80.0e-6

    invariant = d_parent^3 * tab.We_crit
    for We in (12.0, 25.0, 50.0, 100.0, 500.0)
        d_child = breakup_diameter(tab, d_parent, We)
        @test isapprox(d_child^3 * We, invariant; rtol = 1.0e-12)
    end
end

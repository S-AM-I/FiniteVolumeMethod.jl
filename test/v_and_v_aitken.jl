# test/v_and_v_aitken.jl — Aitken-Δ² under-relaxation V&V (Wave 3)
#
# Primitive/algebraic verification of the Aitken-Δ² update rule used by
# the partitioned FSI loop in `src/fsi/`:
#
#   ω_new = −ω_old · (δ_prev · (δ_new − δ_prev)) / ‖δ_new − δ_prev‖²
#
# Five gates:
#   1. Closed-form ω_new for a concrete residual pair (rtol 1e-12).
#   2. Zero-difference guard: δ_new == δ_prev ⇒ ω stays finite and
#      remains in [ω_min, ω_max] (no NaN from 0/0).
#   3. Clamp: ω must always lie in the configured interval.
#   4. First iteration returns the seed ω₀.
#   5. Three consecutive updates: ω scales by the analytical
#      factor 1/(1−α) for a geometric residual sequence δ_{k+1} = α δ_k.

using FiniteVolumeMethod
using Test
using LinearAlgebra: dot, norm

const update_aitken_omega! = FiniteVolumeMethod.update_aitken_omega!

@testset "V&V: Aitken — closed-form ω_new" begin
    relax = AitkenRelaxation(; omega0 = 0.5, omega_min = 1.0e-4, omega_max = 10.0)

    delta_prev = [0.3, 0.2, 0.1]
    delta_new = [0.15, 0.1, 0.05]

    # Seed prev_residual via first call.
    ω0 = update_aitken_omega!(relax, delta_prev)
    @test ω0 == 0.5

    ω1 = update_aitken_omega!(relax, delta_new)

    # Closed-form expectation:
    diff = delta_new .- delta_prev
    expected = -0.5 * dot(delta_prev, diff) / dot(diff, diff)
    expected = clamp(expected, 1.0e-4, 10.0)

    @test isapprox(ω1, expected; rtol = 1.0e-12)
end

@testset "V&V: Aitken — zero residual difference keeps ω finite and in bounds" begin
    relax = AitkenRelaxation(; omega0 = 0.7, omega_min = 0.01, omega_max = 0.9)
    δ = [0.1, 0.2]

    ω_seed = update_aitken_omega!(relax, δ)
    @test ω_seed == 0.7

    # Re-submit the same residual ⇒ denominator ‖diff‖² = 0 ⇒ guard
    # must leave ω untouched (finite, inside clamp).
    ω_again = update_aitken_omega!(relax, copy(δ))

    @test isfinite(ω_again)
    @test ω_again == 0.7
    @test 0.01 ≤ ω_again ≤ 0.9
end

@testset "V&V: Aitken — clamp enforces [ω_min, ω_max]" begin
    # Lower-bound clamp: engineer a residual pair whose analytical ω
    # is 0 (prev ⟂ diff) so clamp must pull ω up to ω_min.
    relax_lo = AitkenRelaxation(; omega0 = 0.5, omega_min = 0.3, omega_max = 0.9)
    update_aitken_omega!(relax_lo, [1.0, 0.0])
    # δ_new = [1.0, 1.0] ⇒ diff = [0,1] ⊥ prev = [1,0] ⇒ raw ω = 0,
    # clamp forces 0.3.
    ω_lo = update_aitken_omega!(relax_lo, [1.0, 1.0])
    @test ω_lo == 0.3

    # Upper-bound clamp: construct residuals whose raw ω exceeds
    # ω_max = 0.5.
    relax_hi = AitkenRelaxation(; omega0 = 0.9, omega_min = 0.01, omega_max = 0.5)
    update_aitken_omega!(relax_hi, [1.0, 0.0])
    # δ_new = [0.1, 0.0] ⇒ diff = [-0.9, 0] ⇒ raw ω = -0.9 · (-0.9)/0.81
    # = 0.9·(0.9/0.81) = 1.0, then clamp ⇒ 0.5.
    ω_hi = update_aitken_omega!(relax_hi, [0.1, 0.0])
    @test ω_hi == 0.5
end

@testset "V&V: Aitken — first iteration returns seed ω₀" begin
    for ω0 in (0.1, 0.25, 0.5, 0.99)
        relax = AitkenRelaxation(; omega0 = ω0, omega_min = 0.001, omega_max = 2.0)
        ω = update_aitken_omega!(relax, [0.3, -0.1, 0.2])
        @test ω == ω0
    end
end

@testset "V&V: Aitken — ω scales by 1/(1−α) for geometric residuals" begin
    # δ_{k+1} = α · δ_k ⇒ Aitken rule collapses to
    #   ω_{k+1} = ω_k / (1 − α).
    # Three successive updates must match this analytic factor to
    # 1e-10 relative accuracy.
    α = 0.4
    c = 1 / (1 - α)    # ≈ 1.6667

    relax = AitkenRelaxation(; omega0 = 0.5, omega_min = 1.0e-3, omega_max = 5.0)

    δ0 = [0.5, -0.25]
    δ1 = α .* δ0
    δ2 = α .* δ1
    δ3 = α .* δ2

    ω0 = update_aitken_omega!(relax, δ0)
    ω1 = update_aitken_omega!(relax, δ1)
    ω2 = update_aitken_omega!(relax, δ2)
    ω3 = update_aitken_omega!(relax, δ3)

    # First call returns seed.
    @test ω0 == 0.5

    # Each subsequent update multiplies ω by 1/(1−α), modulo clamp.
    @test isapprox(ω1, ω0 * c; rtol = 1.0e-10)
    @test isapprox(ω2, ω1 * c; rtol = 1.0e-10)
    @test isapprox(ω3, ω2 * c; rtol = 1.0e-10)

    # All finite and inside clamp.
    for ω in (ω0, ω1, ω2, ω3)
        @test isfinite(ω)
        @test 1.0e-3 ≤ ω ≤ 5.0
    end
end

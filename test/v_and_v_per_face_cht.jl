# test/v_and_v_per_face_cht.jl — Per-face Patankar CHT coupling V&V
#
# Algebraic invariants for `patankar_interface_coupling(k_f, k_s,
# delta_f, delta_s, T_f, T_s)`, the per-face coupling primitive used by
# the conjugate heat-transfer solver. Covers the six gates listed in the
# Wave 1 fast-path plan:
#
#   1. k_f == k_s && delta_f == delta_s  ⇒  k_eff == k_f
#   2. k_s → ∞                           ⇒  k_eff → k_f · (delta_f + delta_s) / delta_f
#   3. T_f == T_s                        ⇒  q_f ≡ 0
#   4. Sign convention: T_s > T_f        ⇒  q_f > 0 (into fluid)
#   5. Symmetry: swap (f, s) swaps sign of q_f
#   6. T_interface matches the closed form at four random samples
#
# All invariants are primitive/algebraic — no mesh, no linear solve.

using FiniteVolumeMethod
using Test

const patankar = FiniteVolumeMethod.patankar_interface_coupling

@testset "V&V: per-face CHT — symmetric case" begin
    k_f = 2.0
    k_s = 2.0
    delta_f = 0.01
    delta_s = 0.01
    T_f = 300.0
    T_s = 400.0

    k_eff, q_f, T_int = patankar(k_f, k_s, delta_f, delta_s, T_f, T_s)

    @test isapprox(k_eff, k_f; rtol = 1.0e-14)
    @test isapprox(T_int, 0.5 * (T_f + T_s); rtol = 1.0e-14)
    # q_f = k_eff · ΔT / (2δ)
    @test isapprox(q_f, k_f * (T_s - T_f) / (delta_f + delta_s); rtol = 1.0e-14)
end

@testset "V&V: per-face CHT — Robin / perfect-conductor solid limit" begin
    # As k_s → ∞ the solid is a perfect conductor: all resistance is on
    # the fluid side, so k_eff → k_f · (δ_f + δ_s) / δ_f and the
    # interface temperature collapses to T_s.
    k_f = 0.5
    delta_f = 0.01
    delta_s = 0.02
    T_f = 350.0
    T_s = 500.0

    # Approach the limit with k_s = 1e12.
    k_s_huge = 1.0e12
    k_eff, q_f, T_int = patankar(k_f, k_s_huge, delta_f, delta_s, T_f, T_s)

    k_eff_limit = k_f * (delta_f + delta_s) / delta_f
    @test isapprox(k_eff, k_eff_limit; rtol = 1.0e-8)
    @test isapprox(T_int, T_s; rtol = 1.0e-9)
    # Flux into the fluid approaches k_f · (T_s - T_f) / δ_f.
    @test isapprox(q_f, k_f * (T_s - T_f) / delta_f; rtol = 1.0e-8)
end

@testset "V&V: per-face CHT — zero ΔT ⇒ zero flux" begin
    # For any conductivities and distances, equal side temperatures
    # must produce exactly zero heat flux.
    for k_f in (0.5, 2.0, 40.0), k_s in (0.1, 5.0, 100.0),
            delta_f in (0.005, 0.02), delta_s in (0.005, 0.03)
        T = 317.0
        k_eff, q_f, T_int = patankar(k_f, k_s, delta_f, delta_s, T, T)
        @test q_f == 0.0
        @test isapprox(T_int, T; rtol = 1.0e-14)
        @test k_eff > 0.0
    end
end

@testset "V&V: per-face CHT — sign convention" begin
    # T_s > T_f ⇒ q_f > 0 (heat flows from solid into fluid).
    k_f, k_s, d_f, d_s = 0.7, 15.0, 0.01, 0.02
    _, q_hot_solid, _ = patankar(k_f, k_s, d_f, d_s, 300.0, 400.0)
    @test q_hot_solid > 0.0

    # T_f > T_s ⇒ q_f < 0.
    _, q_hot_fluid, _ = patankar(k_f, k_s, d_f, d_s, 400.0, 300.0)
    @test q_hot_fluid < 0.0

    @test isapprox(q_hot_solid, -q_hot_fluid; rtol = 1.0e-14)
end

@testset "V&V: per-face CHT — swap symmetry" begin
    # Swapping (f ↔ s) must negate the flux (the flux direction is
    # relative to "into fluid"), preserve k_eff, and keep the interface
    # temperature invariant.
    k_f, k_s, d_f, d_s = 1.2, 30.0, 0.008, 0.025
    T_f, T_s = 310.0, 520.0

    k_fs, q_fs, T_fs = patankar(k_f, k_s, d_f, d_s, T_f, T_s)
    k_sf, q_sf, T_sf = patankar(k_s, k_f, d_s, d_f, T_s, T_f)

    @test isapprox(k_fs, k_sf; rtol = 1.0e-14)
    @test isapprox(q_fs, -q_sf; rtol = 1.0e-14)
    @test isapprox(T_fs, T_sf; rtol = 1.0e-14)
end

@testset "V&V: per-face CHT — closed-form interface temperature" begin
    # Four random samples. The closed form is
    #   T_int = (k_f·T_f/δ_f + k_s·T_s/δ_s) / (k_f/δ_f + k_s/δ_s)
    samples = [
        (0.5, 15.0, 0.01, 0.02, 300.0, 400.0),
        (2.0, 50.0, 0.005, 0.015, 350.0, 500.0),
        (0.1, 200.0, 0.02, 0.002, 280.0, 600.0),
        (5.0, 5.0, 0.01, 0.01, 400.0, 400.0),
    ]
    for (k_f, k_s, d_f, d_s, T_f, T_s) in samples
        _, _, T_int = patankar(k_f, k_s, d_f, d_s, T_f, T_s)
        w_f = k_f / d_f
        w_s = k_s / d_s
        expected = (w_f * T_f + w_s * T_s) / (w_f + w_s)
        @test isapprox(T_int, expected; rtol = 1.0e-14)
    end
end

@testset "V&V: per-face CHT — k_eff harmonic mean at δ_f = δ_s" begin
    # When δ_f = δ_s = δ, the effective conductivity reduces to the
    # classical harmonic mean 2·k_f·k_s / (k_f + k_s).
    δ = 0.01
    for (k_f, k_s) in ((0.5, 2.0), (0.1, 100.0), (40.0, 400.0))
        k_eff, _, _ = patankar(k_f, k_s, δ, δ, 300.0, 400.0)
        expected = 2 * k_f * k_s / (k_f + k_s)
        @test isapprox(k_eff, expected; rtol = 1.0e-14)
    end
end

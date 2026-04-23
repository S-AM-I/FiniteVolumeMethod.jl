# test/v_and_v_durbin.jl — Durbin realizability cap algebra V&V (v3.0 / Wave 1)
#
# Verifies the k-ε eddy-viscosity realizability cap
#
#     ν_t ← min(C_μ · k²/ε,   C_T · k / |S|)
#
# as implemented by `_apply_durbin_cap!` in `k_epsilon_rans.jl`. We
# exercise the cap as a pure algebraic operation on a small vector
# state — no mesh assembly, no linear solve. That keeps the test
# reproducible to machine epsilon and independent of every other
# Wave 1 change.

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

const _apply_cap = FiniteVolumeMethod._apply_durbin_cap!
const _durbin_C_T = FiniteVolumeMethod._durbin_C_T

"""Minimal stand-in for CollocatedScalarField exposing only the
`internal` vector that `_apply_durbin_cap!` actually reads. Avoids
needing to build a full mesh for an algebra-only test."""
struct _DurbinKField{T}
    internal::Vector{T}
end

@testset "V&V: Durbin — cap inactive at equilibrium (ν_t << C_T k/|S|)" begin
    # At equilibrium we have ν_t = C_μ k²/ε with k ~ O(1), ε ~ O(1),
    # |S| ~ O(1). The unbounded ν_t is O(0.09); the Durbin bound is
    # O(0.6). The cap must be inactive — we read back exactly what we
    # wrote.
    k = _DurbinKField([1.0, 0.5, 2.0, 0.1])
    S_mag = [1.0, 2.0, 0.5, 3.0]
    # ν_t at equilibrium (below the cap for each entry)
    nu_t = [0.09, 0.0225, 0.36, 0.003]
    nu_t_in = copy(nu_t)
    _apply_cap(nu_t, k, S_mag, 0.6)
    # Each cap: 0.6·k/|S| = {0.6, 0.15, 2.4, 0.02} — all > nu_t
    for i in 1:length(nu_t)
        @test nu_t[i] == nu_t_in[i]
    end
end

@testset "V&V: Durbin — cap active at high |S| (ν_t = C_T·k/|S|)" begin
    # Construct an unbounded ν_t that exceeds the Durbin cap. The
    # in-place function must write back the cap value exactly.
    k_vec = [1.0, 2.0, 0.5, 10.0]
    k = _DurbinKField(k_vec)
    S_mag = [100.0, 50.0, 200.0, 1.0]
    C_T = 0.6

    # Start with a deliberately-huge ν_t
    nu_t = [10.0, 20.0, 5.0, 100.0]
    _apply_cap(nu_t, k, S_mag, C_T)
    for i in 1:length(nu_t)
        expected = C_T * k_vec[i] / S_mag[i]
        @test isapprox(nu_t[i], expected; rtol = 1.0e-14)
    end
end

@testset "V&V: Durbin — cap matches closed form to machine precision" begin
    # Randomised-but-deterministic closed-form audit.
    k_vec = [0.1, 1.0, 0.3, 2.5, 0.05]
    S_mag = [2.0, 0.1, 5.0, 1.0, 10.0]
    C_T = 0.6
    k = _DurbinKField(k_vec)
    # Make ν_t always > cap
    nu_t = [1.0e3 for _ in 1:length(k_vec)]
    _apply_cap(nu_t, k, S_mag, C_T)
    for i in 1:length(nu_t)
        expected = C_T * k_vec[i] / S_mag[i]
        @test isapprox(nu_t[i], expected; rtol = 1.0e-12)
    end
end

@testset "V&V: Durbin — cap coefficient defaults to 0.6 when α=0" begin
    # When the user leaves `realizability_alpha = 0`, `_durbin_C_T`
    # must fall back to the Durbin 1996 constant 0.6. When α > 0 it
    # must honour the override.
    m_default = StandardKEpsilon()
    @test _durbin_C_T(m_default) == 0.6

    m_override = StandardKEpsilon(; realizability_alpha = 2 / 3)
    @test isapprox(_durbin_C_T(m_override), 2 / 3; rtol = 1.0e-14)
end

@testset "V&V: Durbin — vanishing |S| leaves ν_t untouched" begin
    # When |S| → 0 the bound k/|S| → ∞; the cap must not fire,
    # preserving the finite ν_t that the equilibrium formula already
    # produced.
    k = _DurbinKField([1.0, 2.0])
    S_mag = [0.0, 1.0e-16]
    nu_t = [0.5, 0.25]
    nu_t_in = copy(nu_t)
    _apply_cap(nu_t, k, S_mag, 0.6)
    for i in 1:length(nu_t)
        @test nu_t[i] == nu_t_in[i]
    end
end

@testset "V&V: Durbin — k floor protects against negative/zero k" begin
    # `_apply_durbin_cap!` clamps k with a 1e-10 floor; so even with
    # degenerate k values the cap does not blow up or go negative.
    k = _DurbinKField([0.0, -1.0e-12, 1.0])
    S_mag = [1.0, 1.0, 1.0]
    nu_t = [1.0e3, 1.0e3, 1.0e3]
    _apply_cap(nu_t, k, S_mag, 0.6)
    @test all(isfinite, nu_t)
    @test all(>=(0.0), nu_t)
    # The third entry should match the closed form exactly.
    @test isapprox(nu_t[3], 0.6; rtol = 1.0e-14)
end

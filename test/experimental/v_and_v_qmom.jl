# test/v_and_v_qmom.jl — V&V for QMoM Wheeler moment inversion
#
# Verifies algebraic invariants of the Wheeler / product-difference
# algorithm (`qmom_recover_abscissae_weights` + `wheeler_inversion`):
#
# 1. Monodisperse: m_k = m_0 · L_0^k recovers (w, L) = (m_0, L_0) at
#    one quadrature node and zero weight at the others.
# 2. Bidisperse: m_k = w_1 L_1^k + w_2 L_2^k (N=2, 4 moments) recovers
#    exactly both pairs.
# 3. Moment-sum invariants: Σ w_i = m_0 and Σ w_i L_i = m_1 to
#    rtol 1e-12.
# 4. Weights non-negative for any realisable moment sequence.
# 5. N=2 closed-form: the Gauss–Chebyshev quadrature from the first 4
#    moments of a uniform n(L) on [a, b] reproduces m_0, m_1, m_2, m_3
#    up to rtol 1e-12.

using LinearAlgebra
using Test

_experimental_warn(::Symbol) = nothing # no-op shim: source included standalone, outside module Experimental
include(joinpath(@__DIR__, "..", "..", "src", "experimental", "population_balance", "qmom.jl"))

function _moments_monodisperse(m_0::T, L_0::T, n_moments::Int) where {T}
    return T[m_0 * L_0^k for k in 0:(n_moments - 1)]
end

function _moments_bidisperse(w::Vector{T}, L::Vector{T}, n_moments::Int) where {T}
    return T[sum(w[i] * L[i]^k for i in eachindex(L)) for k in 0:(n_moments - 1)]
end

function _moments_uniform(a::T, b::T, n_moments::Int) where {T}
    # Moments of the uniform density n(L) = 1 on [a, b]:
    # m_k = (b^(k+1) - a^(k+1)) / (k + 1)
    return T[(b^(k + 1) - a^(k + 1)) / T(k + 1) for k in 0:(n_moments - 1)]
end

@testset "V&V: QMoM Wheeler — monodisperse delta" begin
    m_0 = 1.5
    L_0 = 0.42
    moments = _moments_monodisperse(m_0, L_0, 4)
    weights, abscissae = wheeler_inversion(moments)

    @test length(weights) == 2
    @test length(abscissae) == 2

    # One abscissa must equal L_0 and carry weight ≈ m_0.
    idx = argmax(weights)
    @test isapprox(abscissae[idx], L_0; rtol = 1.0e-10)
    @test isapprox(weights[idx], m_0; rtol = 1.0e-10)

    # The other weight must be effectively zero.
    other = idx == 1 ? 2 : 1
    @test weights[other] < 1.0e-10 * m_0

    # Moment reconstruction holds regardless of node duplication.
    for k in 0:3
        recon = sum(weights[i] * abscissae[i]^k for i in eachindex(weights))
        @test isapprox(recon, moments[k + 1]; rtol = 1.0e-10, atol = 1.0e-12)
    end
end

@testset "V&V: QMoM Wheeler — bidisperse recovery" begin
    w_true = [2.0, 3.0]
    L_true = [1.0, 4.0]
    moments = _moments_bidisperse(w_true, L_true, 4)
    weights, abscissae = wheeler_inversion(moments)

    # Wheeler returns sorted abscissae; check against sorted truth.
    perm = sortperm(L_true)
    L_sorted = L_true[perm]
    w_sorted = w_true[perm]
    @test isapprox(abscissae, L_sorted; rtol = 1.0e-10)
    @test isapprox(weights, w_sorted; rtol = 1.0e-10)
end

@testset "V&V: QMoM Wheeler — moment-sum invariants" begin
    # Realisable moments (mix of two lognormal-like pieces).
    moments = _moments_bidisperse([1.25, 0.75], [0.3, 1.1], 6)
    weights, abscissae = wheeler_inversion(moments)
    @test length(weights) == 3
    @test isapprox(sum(weights), moments[1]; rtol = 1.0e-12)
    @test isapprox(
        sum(weights[i] * abscissae[i] for i in eachindex(abscissae)),
        moments[2]; rtol = 1.0e-12
    )
    @test all(weights .>= -1.0e-12)
end

@testset "V&V: QMoM Wheeler — uniform density (N=2, 4 moments)" begin
    a = 0.5
    b = 1.5
    moments = _moments_uniform(a, b, 4)
    weights, abscissae = wheeler_inversion(moments)
    @test length(weights) == 2

    # The 2-point Gauss quadrature for a uniform density on [a, b]
    # must integrate polynomials of degree ≤ 3 exactly ⇒ reconstructs
    # m_0..m_3 exactly.
    for k in 0:3
        recon = sum(weights[i] * abscissae[i]^k for i in eachindex(weights))
        @test isapprox(recon, moments[k + 1]; rtol = 1.0e-12)
    end

    # Weights non-negative.
    @test all(weights .>= 0.0)

    # Abscissae inside support [a, b].
    @test all(a .<= abscissae .<= b)
end

@testset "V&V: QMoM Wheeler — realisability guard" begin
    # Fabricate an unrealisable moment sequence with negative β
    # coefficient (m_2 too small vs m_1 for any distribution).
    bad = [1.0, 1.0, 0.5, 1.0]  # violates Cauchy-Schwarz: m_0 m_2 < m_1^2
    @test_throws ErrorException wheeler_inversion(bad)
end

@testset "V&V: QMoM aggregation — constant kernel total-count decay" begin
    # Smoluchowski with β = const preserves total volume (m_3 · π/6)
    # and decays total count (m_0) per analytical Smoluchowski.
    w_true = [1.0, 1.0]
    L_true = [1.0, 2.0]
    moments = _moments_bidisperse(w_true, L_true, 4)
    beta_const(_, _) = 1.0
    src = moment_source_aggregation(moments, beta_const)

    # Total-count source (k=0) must be strictly negative (net loss).
    @test src[1] < 0

    # Volume (third moment, up to π/6) is preserved:
    # d(m_3)/dt_agg = 0 exactly because merging conserves volume.
    @test isapprox(src[4], 0.0; atol = 1.0e-12)
end

@testset "V&V: QMoM breakage — no-op when rate is zero" begin
    moments = _moments_bidisperse([1.0, 2.0], [0.5, 1.5], 4)
    Kb_zero(_) = 0.0
    daughter(_, _) = 0.0
    src = moment_source_breakage(moments, Kb_zero, daughter)
    @test all(isapprox.(src, 0.0; atol = 1.0e-14))
end

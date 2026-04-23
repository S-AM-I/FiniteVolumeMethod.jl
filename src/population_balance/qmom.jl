# population_balance/qmom.jl — Moment-based population balance (Stage 6g)
#
# Evolves the zeroth through 2N-th moments of a particle / bubble /
# droplet size distribution n(L, t) in each CFD cell:
#
#   m_k(x, t) = ∫₀^∞  L^k · n(L, x, t) dL
#
# Growth, aggregation, breakage, and nucleation source terms are
# expressed as moment transforms via the QMoM (McGraw 1997,
# Marchisio & Fox 2013) quadrature closure:
#
#   ∫ g(L) n(L) dL ≈ Σ_i w_i · g(L_i)
#
# where the N abscissae `L_i` and weights `w_i` are recovered from the
# 2N moments via the product-difference (PD) algorithm.
#
# This module provides:
# - `qmom_recover_abscissae_weights(moments, N)` — PD algorithm.
# - Moment-source evaluators for growth, binary aggregation kernel, and
#   binary breakage kernel.
#
# Reference: Marchisio & Fox (2013), Computational Models for
# Polydisperse Particulate and Multiphase Systems (Cambridge).
# Clean-room implementation from the algorithmic description.

using LinearAlgebra: eigen, SymTridiagonal

"""
    qmom_recover_abscissae_weights(moments::AbstractVector{T}, N::Int)
        -> (abscissae::Vector{T}, weights::Vector{T})

Recover `N` abscissae and weights from the first `2N` moments of a
distribution via the Wheeler / product-difference algorithm. Returns
sorted abscissae.

The input `moments` must contain at least `2N` entries (indices 1..2N
correspond to `m_0, m_1, …, m_{2N-1}`).

Throws if the moment sequence is not realizable (e.g. if the PD
Chebyshev coefficients yield negative values, which happens when the
moments don't come from any real distribution).
"""
function qmom_recover_abscissae_weights(moments::AbstractVector{T}, N::Int) where {T}
    length(moments) >= 2 * N ||
        error("qmom requires $(2N) moments, got $(length(moments))")
    moments[1] > zero(T) ||
        error("qmom: m_0 must be positive (got $(moments[1]))")

    # Wheeler algorithm: compute recurrence coefficients α, β of the
    # orthogonal polynomials associated with the moment sequence.
    # Build the Hankel matrix via PD recurrence.
    sigma = zeros(T, 2 * N + 1, 2 * N)
    for i in 1:(2 * N)
        sigma[2, i] = moments[i]
    end
    a = zeros(T, N)
    b = zeros(T, N)
    a[1] = moments[2] / moments[1]
    b[1] = zero(T)

    for k in 2:N
        for j in k:(2 * N - k + 1)
            sigma[k + 1, j] =
                sigma[k, j + 1] - a[k - 1] * sigma[k, j] - b[k - 1] * sigma[k - 1, j]
        end
        a[k] = sigma[k + 1, k + 1] / sigma[k + 1, k] -
            sigma[k, k] / sigma[k, k - 1]
        b[k] = sigma[k + 1, k] / sigma[k, k - 1]
        b[k] >= zero(T) || error(
            "qmom: moment sequence not realizable (β_$k = $(b[k]) < 0)"
        )
    end

    # Jacobi matrix: symmetric tridiagonal with diagonal = α and
    # off-diagonal = sqrt(β). Its eigenvalues are the abscissae;
    # weights come from the first eigenvector components squared,
    # scaled by m_0.
    sub = T[sqrt(max(b[k + 1], zero(T))) for k in 1:(N - 1)]
    J = SymTridiagonal(a, sub)
    F = eigen(J)
    abscissae = F.values
    v1 = F.vectors[1, :]
    weights = moments[1] .* (v1 .^ 2)

    # Sort by abscissa for stable downstream usage
    perm = sortperm(abscissae)
    return abscissae[perm], weights[perm]
end

"""
    qmom_moment_source_growth(weights, abscissae, G::Function, k::Int) -> T

Growth-term contribution to the `k`-th moment:

    (dm_k / dt)_growth = ∫ k L^(k-1) G(L) n(L) dL
                      ≈ k · Σ_i w_i · L_i^(k-1) · G(L_i)

where `G(L)` is the size-dependent growth rate.
"""
function qmom_moment_source_growth(
        weights::AbstractVector{T}, abscissae::AbstractVector{T},
        G::Function, k::Int,
    ) where {T}
    s = zero(T)
    @inbounds for i in eachindex(abscissae)
        Li = abscissae[i]
        s += weights[i] * Li^(k - 1) * G(Li)
    end
    return T(k) * s
end

"""
    qmom_moment_source_aggregation(weights, abscissae, beta::Function, k::Int) -> T

Aggregation-kernel contribution to the `k`-th moment (Smoluchowski-type
binary aggregation):

    (dm_k/dt)_agg = (1/2) · Σ_i Σ_j w_i w_j ·
        [(L_i^3 + L_j^3)^(k/3) − L_i^k − L_j^k] · β(L_i, L_j)

where `β(L1, L2)` is the symmetric aggregation kernel and the volume-
preserving merging rule `L_new^3 = L_i^3 + L_j^3` is assumed.
"""
function qmom_moment_source_aggregation(
        weights::AbstractVector{T}, abscissae::AbstractVector{T},
        beta::Function, k::Int,
    ) where {T}
    s = zero(T)
    @inbounds for i in eachindex(abscissae)
        Li = abscissae[i]
        for j in eachindex(abscissae)
            Lj = abscissae[j]
            L_merge = (Li^3 + Lj^3)^(T(k) / T(3))
            s += weights[i] * weights[j] *
                (L_merge - Li^k - Lj^k) * beta(Li, Lj)
        end
    end
    return T(0.5) * s
end

"""
    qmom_moment_source_breakage(weights, abscissae, Kb::Function, daughter_pdf::Function, k::Int) -> T

Breakage contribution to the `k`-th moment with selection function
`Kb(L)` and daughter distribution `daughter_pdf(L_child, L_parent)`.
Assumes binary breakage (one parent → two children) with
mass-conservation-consistent daughter pdf.

    (dm_k/dt)_break = Σ_i w_i · Kb(L_i) ·
        (∫ L^k daughter_pdf(L, L_i) dL − L_i^k)

The daughter integral is evaluated by the caller-supplied
`daughter_pdf` which should return the expected kth moment directly
for efficiency:

    daughter_pdf(L_parent, k) → expected value of L^k over children.

i.e. this routine treats `daughter_pdf` as a function of `(L_parent, k)`.
"""
function qmom_moment_source_breakage(
        weights::AbstractVector{T}, abscissae::AbstractVector{T},
        Kb::Function, daughter_pdf::Function, k::Int,
    ) where {T}
    s = zero(T)
    @inbounds for i in eachindex(abscissae)
        Li = abscissae[i]
        # expected L^k over children - parent contribution
        s += weights[i] * Kb(Li) * (daughter_pdf(Li, k) - Li^k)
    end
    return s
end

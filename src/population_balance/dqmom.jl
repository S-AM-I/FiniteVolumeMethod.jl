# population_balance/dqmom.jl — Direct Quadrature Method of Moments
#
# DQMoM (Marchisio & Fox, JAS 2005; MF13 ch. 7) transports the weights
# w_i and weighted abscissae ζ_i = w_i · L_i of the quadrature
# approximation to n(L, x, t) directly:
#
#   ∂w_i/∂t      + ∇·(U_i · w_i)      = a_i
#   ∂(w_i L_i)/∂t + ∇·(U_i · w_i L_i) = b_i
#
# The sources `(a_i, b_i)` are linked to the usual moment sources
# `s_k = dm_k/dt` (as produced by QMoM moment-source closures) via a
# linear system derived by taking the k-th moment of the DQMoM
# transport equations and equating to the moment source:
#
#   sum_i [(1-k) · L_i^k · a_i + k · L_i^(k-1) · b_i] = s_k,
#       k = 0, …, 2N-1.
#
# This yields a 2N × 2N linear system `A · x = s` with
#   x = [a_1, …, a_N, b_1, …, b_N]
#   A[k, i]     = (1 - k) · L_i^k            (a-block)
#   A[k, N+i]   = k · L_i^(k-1)              (b-block)
# (with k = 0, …, 2N-1 as the row index and using L_i^{-1} := 0 when
# k = 0 so the b-block row 0 is all zero).
#
# Reference: Marchisio & Fox (2013), Computational Models for
# Polydisperse Particulate and Multiphase Systems (Cambridge).

using LinearAlgebra: lu, LinearAlgebra

"""
    dqmom_kernel(abscissae::AbstractVector{T}) -> Matrix{T}

Build the `2N × 2N` linear-system kernel `A` that relates DQMoM
source pairs `(a_i, b_i)` to moment sources `(s_0, …, s_{2N-1})`:

    A * [a; b] = s.

Row `k` (0-indexed) encodes the k-th moment of the DQMoM transport
equations:

    sum_i (1 - k) L_i^k a_i + sum_i k L_i^(k-1) b_i = s_k.
"""
function dqmom_kernel(abscissae::AbstractVector{T}) where {T}
    N = length(abscissae)
    A = zeros(T, 2 * N, 2 * N)
    for k in 0:(2 * N - 1)
        row = k + 1
        for i in 1:N
            Li = abscissae[i]
            A[row, i] = T(1 - k) * Li^k
            if k == 0
                A[row, N + i] = zero(T)
            else
                A[row, N + i] = T(k) * Li^(k - 1)
            end
        end
    end
    return A
end

"""
    dqmom_sources(weights, abscissae, moment_sources) -> (a_sources, b_sources)

Solve the DQMoM linear system to recover the weight sources `a_i` and
the weighted-abscissa sources `b_i = d(w_i · L_i)/dt` from the first
`2N` moment sources `s_k = dm_k/dt`.

The `weights` argument is unused by the algebraic formulation (since
the moment equations are homogeneous in the quadrature pairs) but is
accepted for API symmetry with QMoM and to allow future
regularization (e.g. Tikhonov) of near-degenerate abscissae.

Throws if the kernel is singular — the usual cause is equal abscissae
(collapsed quadrature nodes), which reduces the rank of the
Vandermonde kernel.
"""
function dqmom_sources(
        weights::AbstractVector{T},
        abscissae::AbstractVector{T},
        moment_sources::AbstractVector{T},
    ) where {T}
    N = length(abscissae)
    length(weights) == N ||
        error("dqmom_sources: weights and abscissae length mismatch")
    length(moment_sources) == 2 * N ||
        error(
        "dqmom_sources: need $(2N) moment sources, got $(length(moment_sources))"
    )

    # Reject degenerate quadrature (equal abscissae collapse the kernel).
    for i in 1:N, j in (i + 1):N
        if abs(abscissae[i] - abscissae[j]) < eps(T) * max(abs(abscissae[i]), one(T))
            error(
                "dqmom_sources: degenerate abscissae (L_$i ≈ L_$j) — " *
                    "linear system is singular; regularize quadrature before solving"
            )
        end
    end

    A = dqmom_kernel(abscissae)
    x = A \ collect(moment_sources)
    a_sources = x[1:N]
    b_sources = x[(N + 1):(2 * N)]
    return a_sources, b_sources
end

"""
    dqmom_moment_residual(abscissae, a_sources, b_sources) -> Vector{T}

Forward map: given DQMoM source pairs, reconstruct the first `2N`
moment sources via

    s_k = sum_i [(1-k) L_i^k a_i + k L_i^(k-1) b_i].

Used for round-trip verification in V&V.
"""
function dqmom_moment_residual(
        abscissae::AbstractVector{T},
        a_sources::AbstractVector{T},
        b_sources::AbstractVector{T},
    ) where {T}
    N = length(abscissae)
    A = dqmom_kernel(abscissae)
    x = vcat(collect(a_sources), collect(b_sources))
    return A * x
end

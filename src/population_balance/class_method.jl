# population_balance/class_method.jl — Sectional / class method
#
# The class method (Hounslow, Ryall, & Marshall 1988; Kumar & Ramkrishna
# 1996) solves the PBE by discretising the internal coordinate `L` into
# `N_class` bins with edges `L_edges` (length N+1). Each bin stores a
# single number density `n_i(x, t)` per CFD cell, and the aggregation /
# breakage kernels reduce to pair-wise bin interactions.
#
# Aggregation (Smoluchowski, mass-conserving volume merging
# `V_new = V_i + V_j`):
#
#   dn_i/dt = (1/2) Σ_{j+k → i} η_{jk,i} β_{jk} n_j n_k
#           − n_i Σ_k β_{ik} n_k
#
# where `η_{jk,i} ∈ [0, 1]` is the fraction of the merged particle
# volume that falls into bin `i` (split between the nearest two bins
# for geometric mass conservation per Hounslow et al.).
#
# Breakage (binary, rate Kb(L)):
#
#   dn_i/dt = Σ_{j ≥ i} f_{j→i} · Kb(L_j) · n_j − Kb(L_i) · n_i
#
# where `f_{j→i}` is the daughter-bin distribution (number of children
# of bin `j` that land in bin `i`, mass-conserving).
#
# This file contains the two primitive sources (`aggregate_classes!`
# and `breakage_classes!`) and a moment utility
# `class_moments(n_bins, cm)` for V&V and post-processing.
#
# Reference: Marchisio & Fox (2013), Computational Models for
# Polydisperse Particulate and Multiphase Systems (Cambridge).

"""
    aggregate_classes!(dn_bins, n_bins, cm::ClassMethod{T}, kernel::Function) -> dn_bins

Compute the rate of change `dn_bins[i] = dn_i/dt` of each bin due to
binary aggregation. The merged-particle volume
`V_new = V_i + V_j` is split between the two bins `m, m+1` whose
centers bracket it, with weights chosen to conserve both total number
and total volume on the coarse bins (Hounslow volume-splitting
scheme).

`kernel(L_i, L_j)` must be non-negative and symmetric. `dn_bins` is
overwritten with the aggregation contribution; call
`fill!(dn_bins, 0)` before accumulating sources from multiple
mechanisms.

Returns `dn_bins` for chaining.
"""
function aggregate_classes!(
        dn_bins::AbstractVector{T},
        n_bins::AbstractVector{T},
        cm::ClassMethod{T},
        kernel::Function,
    ) where {T}
    N = cm.N_class
    length(n_bins) == N || error("aggregate_classes!: n_bins length mismatch")
    length(dn_bins) == N || error("aggregate_classes!: dn_bins length mismatch")

    V = cm.V_centers
    L = cm.L_centers
    fill!(dn_bins, zero(T))

    # Birth: every unordered pair (i ≤ j) creates V_new that is split
    # between bracketing bins.
    for i in 1:N, j in i:N
        β = T(kernel(L[i], L[j]))
        β >= zero(T) || error("aggregate_classes!: kernel returned negative value")
        rate = β * n_bins[i] * n_bins[j]
        rate == zero(T) && continue

        V_new = V[i] + V[j]
        if V_new <= V[1]
            # Lands entirely in first bin
            contribution = i == j ? T(0.5) * rate : rate
            dn_bins[1] += contribution
        elseif V_new >= V[N]
            # Lands entirely in last bin
            contribution = i == j ? T(0.5) * rate : rate
            dn_bins[N] += contribution
        else
            # Locate bracketing bins m, m+1 with V[m] ≤ V_new < V[m+1]
            m = 1
            for k in 1:(N - 1)
                if V[k] <= V_new < V[k + 1]
                    m = k
                    break
                end
            end
            # Hounslow volume-splitting: weight to conserve number and
            # total volume on the coarse bins.
            ξ = (V[m + 1] - V_new) / (V[m + 1] - V[m])
            contribution = i == j ? T(0.5) * rate : rate
            dn_bins[m] += ξ * contribution
            dn_bins[m + 1] += (one(T) - ξ) * contribution
        end

        # Death: both parent bins lose one particle per collision; the
        # diagonal (i == j) loses two from bin i.
        if i == j
            dn_bins[i] -= rate
        else
            dn_bins[i] -= rate
            dn_bins[j] -= rate
        end
    end
    return dn_bins
end

"""
    breakage_classes!(dn_bins, n_bins, cm::ClassMethod{T},
        Kb::Function, fragment_distribution::Function) -> dn_bins

Compute the rate of change of each bin due to binary breakage. `Kb(L)`
is the selection rate and `fragment_distribution(L_parent, L_child) →
f` returns the expected number of children with characteristic length
`L_child` produced per parent of length `L_parent`. The function
assumes binary breakage so mass-conservation is enforced per caller
(not checked).

Returns `dn_bins` for chaining.
"""
function breakage_classes!(
        dn_bins::AbstractVector{T},
        n_bins::AbstractVector{T},
        cm::ClassMethod{T},
        Kb::Function,
        fragment_distribution::Function,
    ) where {T}
    N = cm.N_class
    L = cm.L_centers
    fill!(dn_bins, zero(T))
    for j in 1:N
        rate_j = T(Kb(L[j])) * n_bins[j]
        rate_j >= zero(T) || error("breakage_classes!: selection rate negative")
        rate_j == zero(T) && continue
        dn_bins[j] -= rate_j
        for i in 1:j
            f = T(fragment_distribution(L[j], L[i]))
            dn_bins[i] += f * rate_j
        end
    end
    return dn_bins
end

"""
    class_moments(n_bins, cm::ClassMethod{T}, order::Int = 0) -> T

Compute the `order`-th moment of the discretised number density:

    m_k ≈ Σ_i n_i · L_i^k.

Useful for V&V (m_0 = total count, m_3 · (π/6) · ρ = total mass).
"""
function class_moments(
        n_bins::AbstractVector{T}, cm::ClassMethod{T}, order::Int = 0
    ) where {T}
    length(n_bins) == cm.N_class || error("class_moments: n_bins length mismatch")
    s = zero(T)
    @inbounds for i in 1:(cm.N_class)
        s += n_bins[i] * cm.L_centers[i]^order
    end
    return s
end

"""
    class_total_volume(n_bins, cm::ClassMethod{T}) -> T

Total dispersed-phase volume Σ_i n_i · V_i (sphere-equivalent from the
bin centers). Exact invariant under aggregation when bins are chosen
such that V_new lands on a bin center; approximately invariant with
the Hounslow volume-splitting used in `aggregate_classes!`.
"""
function class_total_volume(
        n_bins::AbstractVector{T}, cm::ClassMethod{T}
    ) where {T}
    s = zero(T)
    @inbounds for i in 1:(cm.N_class)
        s += n_bins[i] * cm.V_centers[i]
    end
    return s
end

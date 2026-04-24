# lagrangian/agglomeration.jl — Coalescence (agglomeration) of Lagrangian particles
#
# A lightweight stochastic coalescence model: when two particles collide
# they merge with probability `p_c`. Volume and momentum are conserved
# exactly:
#
#   d⁺ = (d₁³ + d₂³)^(1/3)
#   m⁺ = m₁ + m₂
#   U⁺ = (m₁·U₁ + m₂·U₂) / (m₁ + m₂)
#
# This module provides algebraic primitives so higher-level collision
# loops (or V&V tests) can compose them with their own detection logic.

using LinearAlgebra: norm

"""
    CoalescenceModel{T}

Coalescence / agglomeration outcome model.

# Fields
- `p_c::T` — capture probability ∈ `[0, 1]`. When a collision is
  detected, the pair merges with probability `p_c`.
"""
struct CoalescenceModel{T}
    p_c::T
end

CoalescenceModel(; p_c::Real = 1.0) = CoalescenceModel{Float64}(Float64(p_c))

"""
    coalesce_pair(d1, d2, m1, m2, U1, U2) -> (d_new, m_new, U_new)

Compute the deterministic post-coalescence diameter, mass and
velocity of a pair of particles. Volume and momentum are conserved.
"""
function coalesce_pair(
        d1::T, d2::T, m1::T, m2::T,
        U1::SVector{Dim, T}, U2::SVector{Dim, T},
    ) where {Dim, T}
    m_new = m1 + m2
    d_new = (d1^3 + d2^3)^(one(T) / T(3))
    U_new = (m1 * U1 + m2 * U2) / m_new
    return d_new, m_new, U_new
end

"""
    try_coalesce!(p1, p2, model) -> Bool

Attempt a stochastic coalescence between `p1` and `p2`. Returns
`true` when the merge was accepted (in which case `p1` holds the
merged particle and `p2.active = false`), and `false` otherwise.

Both particles must carry `:diameter` and `:mass` in their property
bag. When accepted the merged particle receives the momentum-weighted
centre-of-mass position.
"""
function try_coalesce!(
        p1::LagrangianParticle{Dim, T},
        p2::LagrangianParticle{Dim, T},
        model::CoalescenceModel{T},
    ) where {Dim, T}
    (!p1.active || !p2.active) && return false
    if model.p_c < one(T) && rand(T) > model.p_c
        return false
    end
    d1 = T(p1.properties[:diameter])
    d2 = T(p2.properties[:diameter])
    m1 = T(p1.properties[:mass])
    m2 = T(p2.properties[:mass])

    d_new, m_new, U_new = coalesce_pair(d1, d2, m1, m2, p1.velocity, p2.velocity)
    x_new = (m1 * p1.position + m2 * p2.position) / m_new

    p1.position = x_new
    p1.velocity = U_new
    p1.properties[:diameter] = d_new
    p1.properties[:mass] = m_new
    p2.active = false
    return true
end

"""
    apply_agglomeration!(tracker, mesh, model) -> Int

Walk every pair of active particles that share a mesh cell and
attempt a stochastic coalescence. Returns the number of accepted
merges.
"""
function apply_agglomeration!(
        tracker::ParticleTracker{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        model::CoalescenceModel{T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    cell_particles = [Int[] for _ in 1:nc]
    for (i, p) in enumerate(tracker.particles)
        if p.active && 1 <= p.cell_index <= nc
            push!(cell_particles[p.cell_index], i)
        end
    end

    n_merged = 0
    for c in 1:nc
        idxs = cell_particles[c]
        n_p = length(idxs)
        n_p < 2 && continue
        for ii in 1:(n_p - 1), jj in (ii + 1):n_p
            p1 = tracker.particles[idxs[ii]]
            p2 = tracker.particles[idxs[jj]]
            if try_coalesce!(p1, p2, model)
                n_merged += 1
            end
        end
    end
    return n_merged
end

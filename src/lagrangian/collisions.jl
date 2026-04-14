# lagrangian/collisions.jl — O'Rourke stochastic particle collision model
#
# Implements probabilistic collision detection and outcome (coalescence or
# bounce) for Lagrangian particles sharing the same mesh cell.

using LinearAlgebra: norm

"""
    AbstractCollisionModel

Supertype for particle-particle collision models.
"""
abstract type AbstractCollisionModel end

"""
    ORourkeCollision{T} <: AbstractCollisionModel

O'Rourke stochastic collision model.

Particles sharing the same mesh cell are tested for collision
probability based on their relative velocity and diameters. If a
collision occurs, the outcome is coalescence (if Weber number below
threshold) or bounce (elastic reflection of relative velocity).

# Fields
- `We_crit::T` — critical Weber number for coalescence vs bounce (default 12.0)
"""
struct ORourkeCollision{T} <: AbstractCollisionModel
    We_crit::T
end

ORourkeCollision(; We_crit::Real = 12.0) = ORourkeCollision{Float64}(Float64(We_crit))

"""
    apply_collisions!(tracker, mesh, dt, model; rho_f, sigma_s)

Apply stochastic collisions to all active particle pairs sharing a cell.

For each cell containing >1 active particle, compute collision probability
per pair and apply coalescence or bounce outcome.

# Arguments
- `tracker::ParticleTracker` — particle tracker (modified in-place)
- `mesh::UnstructuredFVMMesh` — mesh (cell volumes for collision rate)
- `dt::T` — time step
- `model::ORourkeCollision` — collision model parameters
- `rho_f::T` — fluid density (for Weber number)
- `sigma_s::T` — surface tension coefficient [N/m]
"""
function apply_collisions!(
        tracker::ParticleTracker{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
        model::ORourkeCollision{T};
        rho_f::T = one(T),
        sigma_s::T = T(0.072),
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    # Build cell → particle index map
    cell_particles = [Int[] for _ in 1:nc]
    for (i, p) in enumerate(tracker.particles)
        if p.active && 1 <= p.cell_index <= nc
            push!(cell_particles[p.cell_index], i)
        end
    end

    for c in 1:nc
        idxs = cell_particles[c]
        n_p = length(idxs)
        n_p < 2 && continue

        V_cell = mesh.cell_volumes[c]

        # Check each pair
        for ii in 1:(n_p - 1), jj in (ii + 1):n_p
            p1 = tracker.particles[idxs[ii]]
            p2 = tracker.particles[idxs[jj]]
            (!p1.active || !p2.active) && continue

            d1 = T(p1.properties[:diameter])
            d2 = T(p2.properties[:diameter])
            d_max = max(d1, d2)

            # Relative velocity
            v_rel = norm(p1.velocity - p2.velocity)
            v_rel < eps(T) && continue

            # Collision cross section
            sigma_coll = T(pi) / 4 * d_max^2

            # Collision probability (O'Rourke)
            P_coll = sigma_coll * v_rel * dt / V_cell
            P_coll = min(P_coll, one(T))

            # Stochastic test
            rand(T) > P_coll && continue

            # Weber number based on relative velocity
            We = rho_f * v_rel^2 * d_max / max(sigma_s, eps(T))

            if We < model.We_crit
                # Coalescence: merge into the larger particle
                m1 = T(p1.properties[:mass])
                m2 = T(p2.properties[:mass])
                m_total = m1 + m2

                # Momentum-weighted merge
                v_merged = (m1 * p1.velocity + m2 * p2.velocity) / m_total
                x_merged = (m1 * p1.position + m2 * p2.position) / m_total

                # Update larger particle, deactivate smaller
                if m1 >= m2
                    p1.velocity = v_merged
                    p1.position = x_merged
                    p1.properties[:mass] = m_total
                    p1.properties[:diameter] = (d1^3 + d2^3)^(one(T) / 3)
                    p2.active = false
                else
                    p2.velocity = v_merged
                    p2.position = x_merged
                    p2.properties[:mass] = m_total
                    p2.properties[:diameter] = (d1^3 + d2^3)^(one(T) / 3)
                    p1.active = false
                end
            else
                # Bounce: elastic reflection of relative velocity
                m1 = T(p1.properties[:mass])
                m2 = T(p2.properties[:mass])
                m_total = m1 + m2

                v_cm = (m1 * p1.velocity + m2 * p2.velocity) / m_total
                p1.velocity = v_cm + m2 / m_total * (p1.velocity - p2.velocity)
                p2.velocity = v_cm - m1 / m_total * (p1.velocity - p2.velocity)
            end
        end
    end

    return nothing
end

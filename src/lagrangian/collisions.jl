# lagrangian/collisions.jl — Particle collision models
#
# Three collision families are provided:
#
# 1. **O'Rourke stochastic** — cell-local probabilistic collision detection
#    with coalescence/bounce outcome (legacy; used by the spray solver).
# 2. **Hard-sphere DEM** — deterministic impulsive binary collision between
#    particles in contact, parameterised by a coefficient of restitution.
# 3. **Soft-sphere DEM** — Hertzian/linear spring-dashpot contact force with
#    Coulomb-capped tangential friction.

using LinearAlgebra: norm, dot

"""
    AbstractCollisionModel

Supertype for particle-particle collision models.
"""
abstract type AbstractCollisionModel end

# ═════════════════════════════════════════════════════════════════════════
# O'Rourke stochastic collision model
# ═════════════════════════════════════════════════════════════════════════

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

# ═════════════════════════════════════════════════════════════════════════
# Hard-sphere DEM collision model
# ═════════════════════════════════════════════════════════════════════════

"""
    HardSphereCollision{T} <: AbstractCollisionModel

Deterministic hard-sphere DEM collision. Given two particles in
contact and a line-of-centres unit normal `n̂`, the post-collision
velocities follow from the impulsive equation with coefficient of
restitution `e`:

```
U₁⁺ = U₁ − (1 + e)·m₂/(m₁ + m₂) · ((U₁ − U₂)·n̂) · n̂
U₂⁺ = U₂ + (1 + e)·m₁/(m₁ + m₂) · ((U₁ − U₂)·n̂) · n̂
```

Linear momentum is conserved exactly. `e = 1` is perfectly elastic
(kinetic energy preserved); `e = 0` is perfectly inelastic (the
components of velocity along `n̂` collapse to a common value).

# Fields
- `e::T` — coefficient of restitution ∈ `[0, 1]` (default `1.0`)
"""
struct HardSphereCollision{T} <: AbstractCollisionModel
    e::T
end

HardSphereCollision(; e::Real = 1.0) = HardSphereCollision{Float64}(Float64(e))

"""
    apply_hard_sphere_collision!(p1, p2, model; normal = nothing) -> nothing

Resolve a single binary hard-sphere collision between `p1` and `p2` in
place. When `normal` is `nothing` (default), the line-of-centres normal
is computed from the particle positions `(p2 - p1)`. If the particles
are co-located, the caller must pass an explicit `normal`.

Particle masses are read from `p.properties[:mass]`.
"""
function apply_hard_sphere_collision!(
        p1::LagrangianParticle{Dim, T},
        p2::LagrangianParticle{Dim, T},
        model::HardSphereCollision{T};
        normal::Union{Nothing, SVector{Dim, T}} = nothing,
    ) where {Dim, T}
    m1 = T(p1.properties[:mass])
    m2 = T(p2.properties[:mass])

    n = if normal === nothing
        Δ = p2.position - p1.position
        nrm = norm(Δ)
        nrm < eps(T) && return nothing
        Δ / nrm
    else
        normal
    end

    v_rel = p1.velocity - p2.velocity
    v_rel_n = dot(v_rel, n)
    # Only resolve if approaching (v_rel · n > 0).
    v_rel_n <= zero(T) && return nothing

    e = model.e
    factor = (one(T) + e) * v_rel_n / (m1 + m2)
    p1.velocity = p1.velocity - (m2 * factor) * n
    p2.velocity = p2.velocity + (m1 * factor) * n
    return nothing
end

# ═════════════════════════════════════════════════════════════════════════
# Soft-sphere DEM collision model
# ═════════════════════════════════════════════════════════════════════════

"""
    SoftSphereCollision{T} <: AbstractCollisionModel

Linear spring-dashpot soft-sphere DEM contact model. For two
particles with overlap `δ > 0` along the contact normal:

```
F_n = k·δ  −  γ·δ̇         (normal; spring + viscous damping)
F_t = −min(|μ_f·F_n|, k_t·|u_t|) · t̂   (tangential; Coulomb-capped)
```

The force is applied equal and opposite to the two particles. Use
this in the particle integration step as `F_contact += F_n·n̂ + F_t·t̂`.

# Fields
- `k::T`   — normal spring stiffness [N/m]
- `gamma::T` — normal damping coefficient [kg/s]
- `mu_f::T` — Coulomb friction coefficient (default `0.3`)
- `k_t::T` — tangential spring stiffness [N/m] (default `k`)
"""
struct SoftSphereCollision{T} <: AbstractCollisionModel
    k::T
    gamma::T
    mu_f::T
    k_t::T
end

function SoftSphereCollision(;
        k::Real = 1.0e3,
        gamma::Real = 1.0,
        mu_f::Real = 0.3,
        k_t::Real = k,
    )
    T = Float64
    return SoftSphereCollision{T}(T(k), T(gamma), T(mu_f), T(k_t))
end

"""
    soft_sphere_force(p1, p2, model) -> (F_on_p1, F_on_p2)

Return the contact force exerted by the soft-sphere model on each
particle. If there is no overlap the returned forces are zero.

The overlap is `δ = (r₁ + r₂) − |x₂ − x₁|`; when `δ > 0` a normal
spring–dashpot force plus a Coulomb-capped tangential spring force is
returned.
"""
function soft_sphere_force(
        p1::LagrangianParticle{Dim, T},
        p2::LagrangianParticle{Dim, T},
        model::SoftSphereCollision{T},
    ) where {Dim, T}
    r1 = T(p1.properties[:diameter]) / 2
    r2 = T(p2.properties[:diameter]) / 2
    Δ = p2.position - p1.position
    dist = norm(Δ)
    # No overlap (or co-located) ⇒ zero force.
    if dist < eps(T) || dist >= r1 + r2
        return zero(SVector{Dim, T}), zero(SVector{Dim, T})
    end
    δ = (r1 + r2) - dist
    n = Δ / dist
    v_rel = p1.velocity - p2.velocity
    v_rel_n = dot(v_rel, n)              # closing speed (positive ⇒ approach)

    # Normal force on p1 points from p2 → p1 (i.e. −n):
    F_n_mag = model.k * δ + model.gamma * v_rel_n
    F_n_on_p1 = -F_n_mag * n

    # Tangential relative velocity
    v_t = v_rel - v_rel_n * n
    v_t_mag = norm(v_t)
    F_t_on_p1 = if v_t_mag < eps(T)
        zero(SVector{Dim, T})
    else
        t = v_t / v_t_mag
        F_t_mag = min(abs(model.mu_f * F_n_mag), model.k_t * v_t_mag)
        -F_t_mag * t
    end
    F_on_p1 = F_n_on_p1 + F_t_on_p1
    return F_on_p1, -F_on_p1
end

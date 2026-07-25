# lagrangian/injection.jl — Injector patterns for Lagrangian particle seeding
#
# Four common spray-pattern injectors are provided:
#
# - `ConeInjector`       — solid-cone pattern with random angular / radial
#                          distribution inside a half-angle θ (2D and 3D).
# - `HollowConeInjector` — annular pattern at a prescribed half-angle
#                          θ ± Δθ, used for pressure-swirl atomisers.
# - `FlatFanInjector`    — flat-fan pattern, used for agricultural
#                          spray nozzles; particles emitted within a
#                          planar sector.
# - `SolidConeInjector`  — uniform solid-cone pattern for diesel-jet
#                          atomisers (volume-uniform sampling).
#
# Each injector exposes `inject_particles!(tracker, injector, n, t) -> nothing`.
# Stochastic injectors draw from the Julia global RNG; tests should
# `Random.seed!(...)` inside the testset for reproducibility.

using LinearAlgebra: norm

"""
    AbstractInjector{Dim, T}

Supertype for injector patterns. Concrete subtypes must implement
`inject_particles!`.
"""
abstract type AbstractInjector{Dim, T} end

# ─── helpers ─────────────────────────────────────────────────────────

# Build an orthonormal frame (t1, t2) orthogonal to the unit vector `axis`.
function _orthonormal_frame(axis::SVector{3, T}) where {T}
    # Pick a helper vector that is least aligned with the axis
    helper = abs(axis[1]) < T(0.9) ? SVector{3, T}(one(T), zero(T), zero(T)) :
        SVector{3, T}(zero(T), one(T), zero(T))
    t1 = helper - axis * (helper[1] * axis[1] + helper[2] * axis[2] + helper[3] * axis[3])
    t1 = t1 / norm(t1)
    t2 = SVector{3, T}(
        axis[2] * t1[3] - axis[3] * t1[2],
        axis[3] * t1[1] - axis[1] * t1[3],
        axis[1] * t1[2] - axis[2] * t1[1],
    )
    return t1, t2
end

# 2-D perpendicular (π/2 rotation).
function _orthonormal_frame(axis::SVector{2, T}) where {T}
    return SVector{2, T}(-axis[2], axis[1])
end

# Push a new particle onto the tracker (mirroring `inject_particles!` in
# parabolic/particles.jl — kept here to avoid a cross-module dependency).
function _push_particle!(
        tracker::ParticleTracker{Dim, T},
        position::SVector{Dim, T},
        velocity::SVector{Dim, T};
        diameter::T = T(1.0e-4),
        density::T = T(1000),
    ) where {Dim, T}
    id = tracker.next_id[]
    tracker.next_id[] = id + 1
    mass = T(pi) / T(6) * diameter^3 * density
    props = Dict{Symbol, Any}(
        :diameter => diameter,
        :density => density,
        :mass => mass,
        :temperature => T(300),
        :Cp => T(4180),
    )
    p = LagrangianParticle{Dim, T}(position, velocity, 0, id, true, props)
    push!(tracker.particles, p)
    return p
end

# ═════════════════════════════════════════════════════════════════════════
# ConeInjector — solid-cone (uniform in cos θ)
# ═════════════════════════════════════════════════════════════════════════

"""
    ConeInjector{Dim, T} <: AbstractInjector{Dim, T}

Solid-cone injector. Particles are emitted from `origin` into a cone
of half-angle `half_angle` about the unit vector `axis`, with
magnitude `speed` and diameter `diameter`.

In 3-D the angular distribution is uniform on the spherical cap; in
2-D it is uniform in the planar sector `[−θ, +θ]` about `axis`.

The optional `r_inj` field lets the injector seed particles from a
disc of radius `r_inj` perpendicular to `axis` (set to `0` for a point
source).

# Fields
- `origin::SVector{Dim, T}` — nozzle origin
- `axis::SVector{Dim, T}` — unit vector along the centre-line
- `half_angle::T` — cone half-angle [rad]
- `r_inj::T` — injector disc radius [m]
- `speed::T` — particle speed [m/s]
- `diameter::T` — particle diameter [m]
- `density::T` — particle density [kg/m³]
"""
struct ConeInjector{Dim, T} <: AbstractInjector{Dim, T}
    origin::SVector{Dim, T}
    axis::SVector{Dim, T}
    half_angle::T
    r_inj::T
    speed::T
    diameter::T
    density::T
end

function ConeInjector(;
        origin::SVector{Dim, T},
        axis::SVector{Dim, T},
        half_angle::Real,
        r_inj::Real = 0.0,
        speed::Real = 1.0,
        diameter::Real = 1.0e-4,
        density::Real = 1000.0,
    ) where {Dim, T}
    ax = axis / norm(axis)
    return ConeInjector{Dim, T}(
        origin, ax, T(half_angle), T(r_inj),
        T(speed), T(diameter), T(density),
    )
end

function inject_particles!(
        tracker::ParticleTracker{3, T},
        injector::ConeInjector{3, T},
        n_particles::Int,
        t::T = zero(T),
    ) where {T}
    t1, t2 = _orthonormal_frame(injector.axis)
    θ_max = injector.half_angle
    for _ in 1:n_particles
        # Uniform on spherical cap: cos θ ∈ [cos θ_max, 1], φ ∈ [0, 2π]
        cos_θ = one(T) - rand(T) * (one(T) - cos(θ_max))
        sin_θ = sqrt(max(zero(T), one(T) - cos_θ^2))
        φ = T(2) * T(pi) * rand(T)
        dir = cos_θ * injector.axis + sin_θ * (cos(φ) * t1 + sin(φ) * t2)
        # Random disc position
        r_disc = injector.r_inj * sqrt(rand(T))
        φ_disc = T(2) * T(pi) * rand(T)
        pos = injector.origin + r_disc * (cos(φ_disc) * t1 + sin(φ_disc) * t2)
        vel = injector.speed * dir
        _push_particle!(
            tracker, pos, vel;
            diameter = injector.diameter,
            density = injector.density,
        )
    end
    return nothing
end

function inject_particles!(
        tracker::ParticleTracker{2, T},
        injector::ConeInjector{2, T},
        n_particles::Int,
        t::T = zero(T),
    ) where {T}
    t1 = _orthonormal_frame(injector.axis)
    θ_max = injector.half_angle
    for _ in 1:n_particles
        θ = (T(2) * rand(T) - one(T)) * θ_max
        dir = cos(θ) * injector.axis + sin(θ) * t1
        s = (T(2) * rand(T) - one(T)) * injector.r_inj
        pos = injector.origin + s * t1
        vel = injector.speed * dir
        _push_particle!(
            tracker, pos, vel;
            diameter = injector.diameter,
            density = injector.density,
        )
    end
    return nothing
end

# ═════════════════════════════════════════════════════════════════════════
# HollowConeInjector — annular spray sheet (pressure-swirl atomiser)
# ═════════════════════════════════════════════════════════════════════════

"""
    HollowConeInjector{Dim, T} <: AbstractInjector{Dim, T}

Hollow-cone injector. Particles are emitted at a half-angle drawn
uniformly from `[half_angle - Δθ, half_angle + Δθ]` and at a radial
position in the annulus `[r_i, r_o]` on the plane normal to `axis`.

# Fields
- `origin::SVector{Dim, T}`
- `axis::SVector{Dim, T}`  — centre-line (unit vector)
- `half_angle::T` — nominal cone half-angle [rad]
- `delta_theta::T` — angular spread about `half_angle` [rad]
- `r_i::T` — inner injector radius [m]
- `r_o::T` — outer injector radius [m]
- `speed::T`, `diameter::T`, `density::T`
"""
struct HollowConeInjector{Dim, T} <: AbstractInjector{Dim, T}
    origin::SVector{Dim, T}
    axis::SVector{Dim, T}
    half_angle::T
    delta_theta::T
    r_i::T
    r_o::T
    speed::T
    diameter::T
    density::T
end

function HollowConeInjector(;
        origin::SVector{Dim, T},
        axis::SVector{Dim, T},
        half_angle::Real,
        delta_theta::Real = 0.0,
        r_i::Real,
        r_o::Real,
        speed::Real = 1.0,
        diameter::Real = 1.0e-4,
        density::Real = 1000.0,
    ) where {Dim, T}
    ax = axis / norm(axis)
    return HollowConeInjector{Dim, T}(
        origin, ax, T(half_angle), T(delta_theta),
        T(r_i), T(r_o), T(speed), T(diameter), T(density),
    )
end

function inject_particles!(
        tracker::ParticleTracker{3, T},
        injector::HollowConeInjector{3, T},
        n_particles::Int,
        t::T = zero(T),
    ) where {T}
    t1, t2 = _orthonormal_frame(injector.axis)
    for _ in 1:n_particles
        θ = injector.half_angle + (T(2) * rand(T) - one(T)) * injector.delta_theta
        φ = T(2) * T(pi) * rand(T)
        dir = cos(θ) * injector.axis + sin(θ) * (cos(φ) * t1 + sin(φ) * t2)
        # Radial position in annulus: area-uniform sampling via √u.
        u = rand(T)
        r = sqrt(injector.r_i^2 + u * (injector.r_o^2 - injector.r_i^2))
        pos = injector.origin + r * (cos(φ) * t1 + sin(φ) * t2)
        vel = injector.speed * dir
        _push_particle!(
            tracker, pos, vel;
            diameter = injector.diameter,
            density = injector.density,
        )
    end
    return nothing
end

function inject_particles!(
        tracker::ParticleTracker{2, T},
        injector::HollowConeInjector{2, T},
        n_particles::Int,
        t::T = zero(T),
    ) where {T}
    t1 = _orthonormal_frame(injector.axis)
    for _ in 1:n_particles
        θ = injector.half_angle + (T(2) * rand(T) - one(T)) * injector.delta_theta
        # Random side (±t1)
        side = rand(Bool) ? one(T) : -one(T)
        dir = cos(θ) * injector.axis + side * sin(θ) * t1
        # Radial position in [r_i, r_o] (linear 1-D annulus)
        r = injector.r_i + rand(T) * (injector.r_o - injector.r_i)
        pos = injector.origin + side * r * t1
        vel = injector.speed * dir
        _push_particle!(
            tracker, pos, vel;
            diameter = injector.diameter,
            density = injector.density,
        )
    end
    return nothing
end

# ═════════════════════════════════════════════════════════════════════════
# FlatFanInjector — planar fan
# ═════════════════════════════════════════════════════════════════════════

"""
    FlatFanInjector{Dim, T} <: AbstractInjector{Dim, T}

Flat-fan injector. Particles are emitted in the plane spanned by
`axis` and `fan_dir` — the "fan plane" — within a half-angle
`half_angle` about `axis`. The particles are seeded at positions
`origin + u·fan_dir` with `u ∈ [-L/2, +L/2]` (linear nozzle length
`L`) and `width` thickness along the out-of-plane direction (3-D
only).

# Fields
- `origin::SVector{Dim, T}`
- `axis::SVector{Dim, T}` — injection direction (unit)
- `fan_dir::SVector{Dim, T}` — in-plane direction perpendicular to `axis` (unit)
- `half_angle::T` — spray half-angle in the fan plane [rad]
- `length::T` — slot length along `fan_dir` [m]
- `width::T` — slot thickness along the out-of-plane direction [m] (3-D only)
- `speed::T`, `diameter::T`, `density::T`
"""
struct FlatFanInjector{Dim, T} <: AbstractInjector{Dim, T}
    origin::SVector{Dim, T}
    axis::SVector{Dim, T}
    fan_dir::SVector{Dim, T}
    half_angle::T
    length::T
    width::T
    speed::T
    diameter::T
    density::T
end

function FlatFanInjector(;
        origin::SVector{Dim, T},
        axis::SVector{Dim, T},
        fan_dir::SVector{Dim, T},
        half_angle::Real,
        length::Real,
        width::Real = 0.0,
        speed::Real = 1.0,
        diameter::Real = 1.0e-4,
        density::Real = 1000.0,
    ) where {Dim, T}
    ax = axis / norm(axis)
    # Orthogonalise fan_dir against axis
    fd = fan_dir - ax * sum(ax[i] * fan_dir[i] for i in 1:Dim)
    fd = fd / norm(fd)
    return FlatFanInjector{Dim, T}(
        origin, ax, fd, T(half_angle), T(length), T(width),
        T(speed), T(diameter), T(density),
    )
end

function inject_particles!(
        tracker::ParticleTracker{3, T},
        injector::FlatFanInjector{3, T},
        n_particles::Int,
        t::T = zero(T),
    ) where {T}
    # Normal to fan plane
    n = SVector{3, T}(
        injector.axis[2] * injector.fan_dir[3] - injector.axis[3] * injector.fan_dir[2],
        injector.axis[3] * injector.fan_dir[1] - injector.axis[1] * injector.fan_dir[3],
        injector.axis[1] * injector.fan_dir[2] - injector.axis[2] * injector.fan_dir[1],
    )
    for _ in 1:n_particles
        θ = (T(2) * rand(T) - one(T)) * injector.half_angle
        dir = cos(θ) * injector.axis + sin(θ) * injector.fan_dir
        u = (T(2) * rand(T) - one(T)) * injector.length / 2
        w = (T(2) * rand(T) - one(T)) * injector.width / 2
        pos = injector.origin + u * injector.fan_dir + w * n
        vel = injector.speed * dir
        _push_particle!(
            tracker, pos, vel;
            diameter = injector.diameter,
            density = injector.density,
        )
    end
    return nothing
end

function inject_particles!(
        tracker::ParticleTracker{2, T},
        injector::FlatFanInjector{2, T},
        n_particles::Int,
        t::T = zero(T),
    ) where {T}
    for _ in 1:n_particles
        θ = (T(2) * rand(T) - one(T)) * injector.half_angle
        dir = cos(θ) * injector.axis + sin(θ) * injector.fan_dir
        u = (T(2) * rand(T) - one(T)) * injector.length / 2
        pos = injector.origin + u * injector.fan_dir
        vel = injector.speed * dir
        _push_particle!(
            tracker, pos, vel;
            diameter = injector.diameter,
            density = injector.density,
        )
    end
    return nothing
end

# ═════════════════════════════════════════════════════════════════════════
# SolidConeInjector — volume-uniform solid cone (diesel jet)
# ═════════════════════════════════════════════════════════════════════════

"""
    SolidConeInjector{Dim, T} <: AbstractInjector{Dim, T}

Volume-uniform solid-cone injector, intended for dense diesel-jet
atomisers. Unlike [`ConeInjector`](@ref) — which distributes directions
uniformly on the spherical cap — this injector distributes directions
*inside* the cone with a density proportional to `sin θ` so that the
resulting point cloud is uniform per unit solid angle weighted by the
cone volume at each radius. In practice it gives a visually uniform
"filled cone" pattern.

# Fields
- `origin::SVector{Dim, T}`
- `axis::SVector{Dim, T}` (unit)
- `half_angle::T` [rad]
- `r_inj::T` — inlet disc radius [m]
- `speed::T`, `diameter::T`, `density::T`
"""
struct SolidConeInjector{Dim, T} <: AbstractInjector{Dim, T}
    origin::SVector{Dim, T}
    axis::SVector{Dim, T}
    half_angle::T
    r_inj::T
    speed::T
    diameter::T
    density::T
end

function SolidConeInjector(;
        origin::SVector{Dim, T},
        axis::SVector{Dim, T},
        half_angle::Real,
        r_inj::Real = 0.0,
        speed::Real = 1.0,
        diameter::Real = 1.0e-4,
        density::Real = 1000.0,
    ) where {Dim, T}
    ax = axis / norm(axis)
    return SolidConeInjector{Dim, T}(
        origin, ax, T(half_angle), T(r_inj),
        T(speed), T(diameter), T(density),
    )
end

function inject_particles!(
        tracker::ParticleTracker{3, T},
        injector::SolidConeInjector{3, T},
        n_particles::Int,
        t::T = zero(T),
    ) where {T}
    t1, t2 = _orthonormal_frame(injector.axis)
    θ_max = injector.half_angle
    for _ in 1:n_particles
        # Uniform-in-angle sampling (linearly distributed θ).
        θ = θ_max * rand(T)
        φ = T(2) * T(pi) * rand(T)
        dir = cos(θ) * injector.axis + sin(θ) * (cos(φ) * t1 + sin(φ) * t2)
        r_disc = injector.r_inj * sqrt(rand(T))
        φ_disc = T(2) * T(pi) * rand(T)
        pos = injector.origin + r_disc * (cos(φ_disc) * t1 + sin(φ_disc) * t2)
        vel = injector.speed * dir
        _push_particle!(
            tracker, pos, vel;
            diameter = injector.diameter,
            density = injector.density,
        )
    end
    return nothing
end

function inject_particles!(
        tracker::ParticleTracker{2, T},
        injector::SolidConeInjector{2, T},
        n_particles::Int,
        t::T = zero(T),
    ) where {T}
    t1 = _orthonormal_frame(injector.axis)
    θ_max = injector.half_angle
    for _ in 1:n_particles
        θ = (T(2) * rand(T) - one(T)) * θ_max
        dir = cos(θ) * injector.axis + sin(θ) * t1
        s = (T(2) * rand(T) - one(T)) * injector.r_inj
        pos = injector.origin + s * t1
        vel = injector.speed * dir
        _push_particle!(
            tracker, pos, vel;
            diameter = injector.diameter,
            density = injector.density,
        )
    end
    return nothing
end

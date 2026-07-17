# test/v_and_v_injection_patterns.jl — Injection-pattern V&V
#
# Geometric-invariant tests for the four spray-pattern injectors:
#
# - ConeInjector:       every particle has velocity lying inside the cone
#                       half-angle about the injector axis.
# - HollowConeInjector: every particle sits in the annulus [r_i, r_o] on
#                       the plane perpendicular to the axis.
# - FlatFanInjector:    every particle lies in the fan plane (zero
#                       component along the out-of-plane normal).
# - SolidConeInjector:  particles cover the cone uniformly (angle
#                       histogram fills the range [0, θ_max]).
#
# All stochastic tests seed the RNG inside the testset with Random.seed!
# for reproducibility.

using LinearAlgebra: cross, dot, norm
using FiniteVolumeMethod: AbstractParticle, LagrangianParticle
using Random
using StaticArrays
using Test

# Self-contained stubs — the V&V uses the injector geometry only, so we
# reproduce the minimal particle / tracker types required by injection.jl.
abstract type AbstractParticle end
mutable struct LagrangianParticle{N, T} <: AbstractParticle
    position::SVector{N, T}
    velocity::SVector{N, T}
    cell_index::Int
    id::Int
    active::Bool
    properties::Dict{Symbol, Any}
end
struct ParticleTracker{N, T}
    particles::Vector{LagrangianParticle{N, T}}
    next_id::Ref{Int}
end
ParticleTracker{N, T}() where {N, T} = ParticleTracker(Vector{LagrangianParticle{N, T}}(), Ref(1))

include(joinpath(@__DIR__, "..", "src", "collocated", "lagrangian", "injection.jl"))

const TOL = 1.0e-10

@testset "V&V: ConeInjector — all velocities inside cone half-angle (3D)" begin
    Random.seed!(12345)
    θ = 0.3
    axis = SVector(0.0, 0.0, 1.0)
    injector = ConeInjector(;
        origin = SVector(0.0, 0.0, 0.0),
        axis = axis,
        half_angle = θ,
        r_inj = 1.0e-4,
        speed = 10.0,
    )
    tracker = ParticleTracker{3, Float64}()
    inject_particles!(tracker, injector, 500)
    @test length(tracker.particles) == 500
    cos_θ_max = cos(θ) - TOL
    for p in tracker.particles
        u = p.velocity / norm(p.velocity)
        cos_α = dot(u, axis)
        @test cos_α >= cos_θ_max
    end
end

@testset "V&V: ConeInjector — all velocities inside half-angle (2D)" begin
    Random.seed!(999)
    θ = 0.4
    axis = SVector(1.0, 0.0)
    injector = ConeInjector(;
        origin = SVector(0.0, 0.0),
        axis = axis,
        half_angle = θ,
    )
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, injector, 300)
    cos_θ_max = cos(θ) - TOL
    for p in tracker.particles
        u = p.velocity / norm(p.velocity)
        @test dot(u, axis) >= cos_θ_max
    end
end

@testset "V&V: HollowConeInjector — particles in annulus [r_i, r_o]" begin
    Random.seed!(7)
    axis = SVector(0.0, 0.0, 1.0)
    r_i = 1.0e-3
    r_o = 2.0e-3
    injector = HollowConeInjector(;
        origin = SVector(0.0, 0.0, 0.0),
        axis = axis,
        half_angle = 0.4,
        r_i = r_i,
        r_o = r_o,
        speed = 5.0,
    )
    tracker = ParticleTracker{3, Float64}()
    inject_particles!(tracker, injector, 400)
    for p in tracker.particles
        # Radial distance in the plane normal to axis.
        proj = p.position - dot(p.position, axis) * axis
        r = norm(proj)
        @test r >= r_i - TOL
        @test r <= r_o + TOL
    end
end

@testset "V&V: HollowConeInjector — particles near mean half-angle" begin
    Random.seed!(11)
    axis = SVector(0.0, 0.0, 1.0)
    θ = 0.5
    injector = HollowConeInjector(;
        origin = SVector(0.0, 0.0, 0.0),
        axis = axis,
        half_angle = θ,
        delta_theta = 0.05,
        r_i = 1.0e-3,
        r_o = 1.5e-3,
        speed = 5.0,
    )
    tracker = ParticleTracker{3, Float64}()
    inject_particles!(tracker, injector, 300)
    for p in tracker.particles
        u = p.velocity / norm(p.velocity)
        α = acos(clamp(dot(u, axis), -1.0, 1.0))
        @test abs(α - θ) <= 0.05 + TOL
    end
end

@testset "V&V: FlatFanInjector — particles in fan plane (3D)" begin
    Random.seed!(31)
    axis = SVector(0.0, 0.0, 1.0)
    fan_dir = SVector(1.0, 0.0, 0.0)
    injector = FlatFanInjector(;
        origin = SVector(0.0, 0.0, 0.0),
        axis = axis,
        fan_dir = fan_dir,
        half_angle = 0.3,
        length = 1.0e-3,
        width = 0.0,
        speed = 5.0,
    )
    # With width = 0, every particle must lie exactly in the fan plane.
    normal_to_plane = cross(axis, fan_dir)
    tracker = ParticleTracker{3, Float64}()
    inject_particles!(tracker, injector, 200)
    for p in tracker.particles
        # Position must satisfy dot(pos, normal) == 0.
        @test abs(dot(p.position, normal_to_plane)) <= TOL
        # Velocity also lies in the fan plane.
        @test abs(dot(p.velocity, normal_to_plane)) <= TOL * norm(p.velocity)
    end
end

@testset "V&V: FlatFanInjector — half-angle bound (2D)" begin
    Random.seed!(13)
    θ = 0.25
    axis = SVector(0.0, 1.0)
    fan_dir = SVector(1.0, 0.0)
    injector = FlatFanInjector(;
        origin = SVector(0.0, 0.0),
        axis = axis,
        fan_dir = fan_dir,
        half_angle = θ,
        length = 2.0e-3,
        speed = 4.0,
    )
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, injector, 200)
    cos_θ = cos(θ) - TOL
    for p in tracker.particles
        u = p.velocity / norm(p.velocity)
        @test dot(u, axis) >= cos_θ
        # Particle position is on the fan_dir line through origin.
        @test abs(p.position[2]) <= TOL
    end
end

@testset "V&V: SolidConeInjector — all inside cone, uniformly sampled" begin
    Random.seed!(19)
    axis = SVector(0.0, 0.0, 1.0)
    θ_max = 0.5
    injector = SolidConeInjector(;
        origin = SVector(0.0, 0.0, 0.0),
        axis = axis,
        half_angle = θ_max,
        r_inj = 1.0e-4,
        speed = 50.0,
    )
    tracker = ParticleTracker{3, Float64}()
    inject_particles!(tracker, injector, 2_000)
    cos_θ = cos(θ_max) - TOL
    angles = Float64[]
    for p in tracker.particles
        u = p.velocity / norm(p.velocity)
        cos_α = dot(u, axis)
        @test cos_α >= cos_θ
        push!(angles, acos(clamp(cos_α, -1.0, 1.0)))
    end
    # Coverage check: max sampled angle should fill the cone.
    @test maximum(angles) > 0.9 * θ_max
    @test minimum(angles) < 0.1 * θ_max
end

@testset "V&V: SolidConeInjector — 2D half-angle bound" begin
    Random.seed!(23)
    axis = SVector(1.0, 0.0)
    θ_max = 0.35
    injector = SolidConeInjector(;
        origin = SVector(0.0, 0.0),
        axis = axis,
        half_angle = θ_max,
        r_inj = 1.0e-4,
        speed = 30.0,
    )
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, injector, 500)
    cos_θ = cos(θ_max) - TOL
    for p in tracker.particles
        u = p.velocity / norm(p.velocity)
        @test dot(u, axis) >= cos_θ
    end
end

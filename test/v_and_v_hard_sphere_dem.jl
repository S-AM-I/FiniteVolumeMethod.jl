# test/v_and_v_hard_sphere_dem.jl — Hard-sphere DEM V&V
#
# Algebraic verification of the impulsive binary-collision formula:
#
#     U₁⁺ = U₁ − (1 + e)·m₂/(m₁ + m₂) · ((U₁ − U₂)·n̂) · n̂
#     U₂⁺ = U₂ + (1 + e)·m₁/(m₁ + m₂) · ((U₁ − U₂)·n̂) · n̂
#
# Primitive invariants verified:
# - Linear momentum is conserved exactly (rtol 1e-14)
# - e = 1 conserves kinetic energy (rtol 1e-14)
# - e = 0 yields a common normal velocity
# - Heavy-light mass ratio limit (m₁ → ∞) reproduces wall-bounce physics
# - Off-centre collision preserves tangential velocities

using LinearAlgebra: dot, norm
using StaticArrays
using Test

# The DEM models live in src/lagrangian/collisions.jl. This V&V is kept
# self-contained by defining minimal `AbstractParticle` / `LagrangianParticle`
# stubs in the test module and then including the source directly. When the
# main thread wires the types into `FiniteVolumeMethod`, the full package
# test-suite re-uses the same `LagrangianParticle` definition.
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
struct UnstructuredFVMMesh{N, T}
    cell_volumes::Vector{T}
    cell_centers::Matrix{T}
end

include(joinpath(@__DIR__, "..", "src", "collocated", "lagrangian", "collisions.jl"))

function _make_particle(pos::SVector{2, Float64}, vel::SVector{2, Float64}; d = 1.0e-3, rho = 1000.0)
    mass = pi / 6 * d^3 * rho
    props = Dict{Symbol, Any}(
        :diameter => d,
        :density => rho,
        :mass => mass,
        :temperature => 300.0,
        :Cp => 4180.0,
    )
    return LagrangianParticle{2, Float64}(pos, vel, 0, 0, true, props)
end

@testset "V&V: hard-sphere DEM — momentum conservation (pair)" begin
    model = HardSphereCollision(; e = 0.7)
    p1 = _make_particle(SVector(0.0, 0.0), SVector(3.0, 1.0))
    p2 = _make_particle(SVector(1.0e-3, 0.0), SVector(-2.0, -0.5); d = 2.0e-3)

    m1 = p1.properties[:mass]
    m2 = p2.properties[:mass]
    P_before = m1 * p1.velocity + m2 * p2.velocity

    apply_hard_sphere_collision!(p1, p2, model)
    P_after = m1 * p1.velocity + m2 * p2.velocity
    @test isapprox(P_before[1], P_after[1]; rtol = 1.0e-14, atol = 1.0e-14)
    @test isapprox(P_before[2], P_after[2]; rtol = 1.0e-14, atol = 1.0e-14)
end

@testset "V&V: hard-sphere DEM — elastic (e=1) conserves KE" begin
    model = HardSphereCollision(; e = 1.0)
    p1 = _make_particle(SVector(0.0, 0.0), SVector(5.0, 0.0))
    p2 = _make_particle(SVector(1.0e-3, 0.0), SVector(-1.0, 2.0); d = 1.0e-3)

    m1 = p1.properties[:mass]
    m2 = p2.properties[:mass]
    KE_before = 0.5 * m1 * dot(p1.velocity, p1.velocity) +
        0.5 * m2 * dot(p2.velocity, p2.velocity)

    apply_hard_sphere_collision!(p1, p2, model)
    KE_after = 0.5 * m1 * dot(p1.velocity, p1.velocity) +
        0.5 * m2 * dot(p2.velocity, p2.velocity)
    @test isapprox(KE_before, KE_after; rtol = 1.0e-14, atol = 1.0e-14)
end

@testset "V&V: hard-sphere DEM — perfectly inelastic (e=0)" begin
    # With e = 0 the two particles must share a common velocity along n̂.
    model = HardSphereCollision(; e = 0.0)
    n = SVector(1.0, 0.0)
    p1 = _make_particle(SVector(0.0, 0.0), SVector(4.0, 1.0))
    p2 = _make_particle(SVector(1.0e-3, 0.0), SVector(-2.0, 0.5))

    apply_hard_sphere_collision!(p1, p2, model)
    v1n = dot(p1.velocity, n)
    v2n = dot(p2.velocity, n)
    @test isapprox(v1n, v2n; rtol = 1.0e-14, atol = 1.0e-14)
end

@testset "V&V: hard-sphere DEM — heavy-light limit (m1 → ∞)" begin
    # Very heavy p1 hit by a light p2 should leave p1 nearly unchanged and
    # p2's normal component reversed (wall-bounce) at e = 1.
    model = HardSphereCollision(; e = 1.0)
    # m1 ≫ m2 via a large diameter ratio.
    p1 = _make_particle(SVector(0.0, 0.0), SVector(0.0, 0.0); d = 1.0)
    p2 = _make_particle(SVector(1.0, 0.0), SVector(-3.0, 0.0); d = 1.0e-3)

    v1_before = p1.velocity
    v2_before = p2.velocity
    apply_hard_sphere_collision!(p1, p2, model)

    # p1 essentially unchanged (mass ratio ≈ 1e9)
    @test norm(p1.velocity - v1_before) < 1.0e-8
    # p2 normal velocity flipped
    @test isapprox(p2.velocity[1], -v2_before[1]; rtol = 1.0e-6)
end

@testset "V&V: hard-sphere DEM — off-centre preserves tangential velocity" begin
    # Normal = x̂. The pre-collision velocities have non-zero y-components
    # that must be unchanged (impulse is purely along n̂).
    model = HardSphereCollision(; e = 0.8)
    # Place centres along x-axis so the computed normal is (1, 0).
    p1 = _make_particle(SVector(0.0, 0.0), SVector(2.0, 1.7))
    p2 = _make_particle(SVector(1.0e-3, 0.0), SVector(-1.0, -0.4))

    vy1_before = p1.velocity[2]
    vy2_before = p2.velocity[2]
    apply_hard_sphere_collision!(p1, p2, model)
    @test isapprox(p1.velocity[2], vy1_before; rtol = 1.0e-14, atol = 1.0e-14)
    @test isapprox(p2.velocity[2], vy2_before; rtol = 1.0e-14, atol = 1.0e-14)
end

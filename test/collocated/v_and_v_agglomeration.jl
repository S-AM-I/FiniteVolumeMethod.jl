# test/v_and_v_agglomeration.jl — Coalescence / agglomeration V&V
#
# Verifies the deterministic mass, volume, and momentum invariants of the
# `CoalescenceModel` on a pair of Lagrangian particles.
#
# Invariants:
# - p_c = 0 ⇒ no merge occurs
# - p_c = 1 ⇒ deterministic merge succeeds
# - Post-merge mass = m₁ + m₂ (exact)
# - Post-merge diameter = (d₁³ + d₂³)^(1/3) (volume conservation, rtol 1e-14)
# - Post-merge velocity = (m₁·U₁ + m₂·U₂) / (m₁ + m₂) (momentum mean, rtol 1e-14)

using LinearAlgebra: norm
using Random
using StaticArrays
using Test

# Self-contained stubs — mirrors the package definition so the V&V runs
# even before the main thread exports the coalescence API.
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

include(joinpath(@__DIR__, "..", "..", "src", "collocated", "lagrangian", "agglomeration.jl"))

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

@testset "V&V: coalescence — p_c = 0 never merges" begin
    Random.seed!(42)
    model = CoalescenceModel(; p_c = 0.0)
    n_trials = 2_000
    n_merged = 0
    for _ in 1:n_trials
        p1 = _make_particle(SVector(0.0, 0.0), SVector(1.0, 0.0))
        p2 = _make_particle(SVector(1.0e-4, 0.0), SVector(-1.0, 0.0))
        if try_coalesce!(p1, p2, model)
            n_merged += 1
        end
    end
    @test n_merged == 0
end

@testset "V&V: coalescence — p_c = 1 always merges deterministically" begin
    model = CoalescenceModel(; p_c = 1.0)
    p1 = _make_particle(SVector(0.0, 0.0), SVector(3.0, 0.0); d = 1.0e-3)
    p2 = _make_particle(SVector(1.0e-3, 0.0), SVector(-2.0, 1.0); d = 2.0e-3)

    m1_before = p1.properties[:mass]
    m2_before = p2.properties[:mass]
    d1_before = p1.properties[:diameter]
    d2_before = p2.properties[:diameter]
    U1_before = p1.velocity
    U2_before = p2.velocity

    merged = try_coalesce!(p1, p2, model)
    @test merged == true
    @test p2.active == false

    # Mass
    @test isapprox(p1.properties[:mass], m1_before + m2_before; rtol = 1.0e-14, atol = 0.0)
    # Diameter (volume conservation)
    d_expected = (d1_before^3 + d2_before^3)^(1.0 / 3.0)
    @test isapprox(p1.properties[:diameter], d_expected; rtol = 1.0e-14)
    # Velocity (momentum-weighted)
    U_expected = (m1_before * U1_before + m2_before * U2_before) /
        (m1_before + m2_before)
    @test isapprox(p1.velocity[1], U_expected[1]; rtol = 1.0e-14, atol = 1.0e-14)
    @test isapprox(p1.velocity[2], U_expected[2]; rtol = 1.0e-14, atol = 1.0e-14)
end

@testset "V&V: coalescence — coalesce_pair algebra" begin
    d1 = 1.0e-3
    d2 = 2.0e-3
    rho = 1000.0
    m1 = pi / 6 * d1^3 * rho
    m2 = pi / 6 * d2^3 * rho
    U1 = SVector(5.0, 0.0)
    U2 = SVector(-3.0, 2.0)
    d_new, m_new, U_new = coalesce_pair(d1, d2, m1, m2, U1, U2)
    @test isapprox(m_new, m1 + m2; rtol = 1.0e-14)
    @test isapprox(d_new, (d1^3 + d2^3)^(1.0 / 3.0); rtol = 1.0e-14)
    U_expected = (m1 * U1 + m2 * U2) / (m1 + m2)
    @test isapprox(U_new[1], U_expected[1]; rtol = 1.0e-14)
    @test isapprox(U_new[2], U_expected[2]; rtol = 1.0e-14)
end

@testset "V&V: coalescence — momentum conservation under merge" begin
    # ∑ m·U is preserved by the merge.
    model = CoalescenceModel(; p_c = 1.0)
    p1 = _make_particle(SVector(0.0, 0.0), SVector(4.2, 0.3); d = 1.0e-3)
    p2 = _make_particle(SVector(1.0e-3, 0.0), SVector(-1.1, 0.9); d = 1.5e-3)
    m1 = p1.properties[:mass]
    m2 = p2.properties[:mass]
    P_before = m1 * p1.velocity + m2 * p2.velocity

    try_coalesce!(p1, p2, model)
    m_new = p1.properties[:mass]
    P_after = m_new * p1.velocity
    @test isapprox(P_before[1], P_after[1]; rtol = 1.0e-14)
    @test isapprox(P_before[2], P_after[2]; rtol = 1.0e-14)
end

@testset "V&V: coalescence — stochastic probability (p_c = 0.5)" begin
    Random.seed!(2025)
    model = CoalescenceModel(; p_c = 0.5)
    n_trials = 10_000
    n_merged = 0
    for _ in 1:n_trials
        p1 = _make_particle(SVector(0.0, 0.0), SVector(1.0, 0.0))
        p2 = _make_particle(SVector(1.0e-4, 0.0), SVector(-1.0, 0.0))
        if try_coalesce!(p1, p2, model)
            n_merged += 1
        end
    end
    # 3σ bound: σ ≈ √(N·p·(1−p)) = √2500 = 50 ⇒ 150 wide window.
    @test abs(n_merged - n_trials / 2) < 200
end

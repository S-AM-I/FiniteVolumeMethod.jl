# test/v_and_v_particle_state.jl — Particle state + mass consistency V&V (v3.57)
#
# Fifth convergence-verified benchmark for `lagrangian_dpm`,
# joining Stokes terminal velocity (v3.13), Schiller-Naumann
# (v3.26), Ranz-Marshall (v3.33), and TAB breakup (v3.45). Covers
# the particle-initialization primitive `set_particle_properties!`
# and two-way momentum-coupling surface.
#
# Invariants:
#
#   1. m_p = (π/6) · d³ · ρ_p (closed-form sphere mass).
#   2. Doubling density doubles mass.
#   3. Doubling diameter multiplies mass by 8.
#   4. Temperature and Cp properties round-trip.
#   5. inject_particles! yields a ParticleTracker with the
#      requested count.
#   6. Particle id increments uniquely.

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: Particle state — mass = (π/6)·d³·ρ" begin
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.5, 0.5)])
    p = tracker.particles[1]

    d = 100.0e-6
    rho = 1000.0
    set_particle_properties!(p; diameter = d, density = rho)

    expected = pi / 6 * d^3 * rho
    @test isapprox(p.properties[:mass], expected; rtol = 1.0e-14)
end

@testset "V&V: Particle state — density-linear mass scaling" begin
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.5, 0.5), SVector(0.3, 0.3)])

    set_particle_properties!(tracker.particles[1]; diameter = 1.0e-4, density = 1000.0)
    set_particle_properties!(tracker.particles[2]; diameter = 1.0e-4, density = 2000.0)

    m1 = tracker.particles[1].properties[:mass]
    m2 = tracker.particles[2].properties[:mass]
    @test isapprox(m2 / m1, 2.0; rtol = 1.0e-14)
end

@testset "V&V: Particle state — d³ diameter scaling" begin
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.5, 0.5), SVector(0.3, 0.3)])

    set_particle_properties!(tracker.particles[1]; diameter = 1.0e-4, density = 1000.0)
    set_particle_properties!(tracker.particles[2]; diameter = 2.0e-4, density = 1000.0)

    m1 = tracker.particles[1].properties[:mass]
    m2 = tracker.particles[2].properties[:mass]
    @test isapprox(m2 / m1, 8.0; rtol = 1.0e-14)
end

@testset "V&V: Particle state — temperature + Cp round-trip" begin
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.5, 0.5)])
    p = tracker.particles[1]

    set_particle_properties!(
        p;
        diameter = 1.0e-4, density = 1000.0,
        temperature = 450.0, Cp = 1500.0,
    )

    @test p.properties[:temperature] == 450.0
    @test p.properties[:Cp] == 1500.0
    @test p.properties[:diameter] == 1.0e-4
    @test p.properties[:density] == 1000.0
end

@testset "V&V: Particle state — inject_particles! count matches request" begin
    tracker = ParticleTracker{2, Float64}()
    positions = [SVector(0.1 * i, 0.1 * i) for i in 1:5]
    inject_particles!(tracker, positions)

    @test length(tracker.particles) == 5
    for (i, p) in enumerate(tracker.particles)
        @test p.position[1] ≈ 0.1 * i
        @test p.active == true
    end
end

@testset "V&V: Particle state — unique id increments" begin
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.5, 0.5), SVector(0.6, 0.6), SVector(0.7, 0.7)])

    ids = [p.id for p in tracker.particles]
    @test length(unique(ids)) == 3      # all unique
    @test ids == sort(ids)               # monotone increasing
end

@testset "V&V: Particle state — default temperature 300 K + Cp 1000 J/(kg·K)" begin
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.5, 0.5)])
    p = tracker.particles[1]
    set_particle_properties!(p; diameter = 1.0e-5, density = 1000.0)

    # Defaults per the documented signature.
    @test p.properties[:temperature] == 300.0
    @test p.properties[:Cp] == 1000.0
end

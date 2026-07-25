# test/v_and_v_dpm_dispatch.jl — DPM drag dispatch across particles V&V (v3.70)

using FiniteVolumeMethod
using FiniteVolumeMethod: SchillerNaumann, compute_drag_force
using LinearAlgebra: norm, dot
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: DPM — drag force sign opposes slip" begin
    # Drag always decelerates the particle relative to the fluid.
    # Slip = U_f - U_p; F_drag should point in the same direction
    # as slip (accelerating the particle toward U_f).
    for slip_vec in (
            SVector(1.0, 0.0), SVector(0.0, 1.0),
            SVector(-1.0, 0.0), SVector(1.0, 1.0),
        )
        U_f = SVector(0.0, 0.0)
        U_p = U_f - slip_vec
        F = compute_drag_force(StokesDrag(), U_f, U_p, 1.0e-4, 1000.0, 1.2, 1.8e-5)
        # F should be parallel to slip_vec.
        if norm(slip_vec) > 1.0e-10
            slip_hat = slip_vec / norm(slip_vec)
            F_hat = F / norm(F)
            @test isapprox(dot(slip_hat, F_hat), 1.0; rtol = 1.0e-10)
        end
    end
end

@testset "V&V: DPM — Stokes drag zero at zero slip" begin
    U = SVector(1.0, 0.5)
    F = compute_drag_force(StokesDrag(), U, U, 1.0e-4, 1000.0, 1.2, 1.8e-5)
    @test isapprox(F[1], 0.0; atol = 1.0e-14)
    @test isapprox(F[2], 0.0; atol = 1.0e-14)
end

@testset "V&V: DPM — mass-diameter consistency across multiple particles" begin
    tracker = ParticleTracker{2, Float64}()
    positions = [SVector(0.1 * i, 0.5) for i in 1:5]
    inject_particles!(tracker, positions)

    for (i, p) in enumerate(tracker.particles)
        d = 1.0e-5 * i
        rho = 1000.0
        set_particle_properties!(p; diameter = d, density = rho)
        expected = pi / 6 * d^3 * rho
        @test isapprox(p.properties[:mass], expected; rtol = 1.0e-14)
    end
end

@testset "V&V: DPM — inject at distinct positions retains IDs" begin
    tracker = ParticleTracker{2, Float64}()
    inject_particles!(tracker, [SVector(0.1, 0.2), SVector(0.3, 0.4)])
    inject_particles!(tracker, [SVector(0.5, 0.6)])  # second batch

    @test length(tracker.particles) == 3
    ids = [p.id for p in tracker.particles]
    @test length(unique(ids)) == 3
    # Second batch must have an ID strictly greater than all first-batch IDs.
    @test ids[3] > ids[2] > ids[1]
end

@testset "V&V: DPM — Schiller-Naumann Stokes agreement at low Re" begin
    d_p = 1.0e-6   # 1 micron
    rho_p = 1000.0
    rho_f = 1.2
    mu_f = 1.8e-5
    slip = 0.01
    U_f = SVector(0.0, 0.0)
    U_p = SVector(-slip, 0.0)

    Re = rho_f * slip * d_p / mu_f
    @test Re < 0.01   # deep Stokes regime
    F_sn = compute_drag_force(SchillerNaumann(), U_f, U_p, d_p, rho_p, rho_f, mu_f)
    F_st = compute_drag_force(StokesDrag(), U_f, U_p, d_p, rho_p, rho_f, mu_f)
    @test isapprox(F_sn[1], F_st[1]; rtol = 1.0e-2)
end

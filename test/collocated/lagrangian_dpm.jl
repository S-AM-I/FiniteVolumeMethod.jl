using FiniteVolumeMethod
using FiniteVolumeMethod: RanzMarshall, SchillerNaumann, compute_drag_force, compute_energy_source, compute_momentum_source, compute_particle_heat_transfer
using Test
using LinearAlgebra
using StaticArrays

# ── Mesh builder (shared helper) ─────────────────────────────────────
include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

# ── Tests ──────────────────────────────────────────────────────────────

@testset "Lagrangian DPM" begin
    # Common parameters
    d_p = 1.0e-4          # 100 um particle
    rho_p = 2500.0      # glass bead
    rho_f = 1.2         # air
    mu_f = 1.8e-5       # air viscosity
    k_f = 0.026         # air thermal conductivity
    Pr_f = 0.7

    # ── 1. StokesDrag force ─────────────────────────────────────────
    @testset "StokesDrag force" begin
        U_f = SVector(1.0, 0.0)
        U_p = SVector(0.0, 0.0)
        F = compute_drag_force(StokesDrag(), U_f, U_p, d_p, rho_p, rho_f, mu_f)

        mass = pi / 6 * d_p^3 * rho_p
        tau_p = rho_p * d_p^2 / (18 * mu_f)
        F_expected = (mass / tau_p) * (U_f - U_p)

        @test F[1] > 0               # force in flow direction
        @test F[2] == 0.0            # no transverse force
        @test F ≈ F_expected atol = 1.0e-20
    end

    # ── 2. SchillerNaumann force ────────────────────────────────────
    @testset "SchillerNaumann force" begin
        U_f = SVector(1.0, 0.0)
        U_p = SVector(0.0, 0.0)
        F_stokes = compute_drag_force(StokesDrag(), U_f, U_p, d_p, rho_p, rho_f, mu_f)
        F_sn = compute_drag_force(SchillerNaumann(), U_f, U_p, d_p, rho_p, rho_f, mu_f)

        # Correction factor > 1 for Re > 0, so |F_sn| > |F_stokes|
        @test norm(F_sn) > norm(F_stokes)
        @test F_sn[1] > 0  # same direction as flow
    end

    # ── 3. Particle Reynolds number ─────────────────────────────────
    @testset "Particle Reynolds number" begin
        U_f = SVector(2.0, 0.0)
        U_p = SVector(0.0, 0.0)
        Re = FiniteVolumeMethod._particle_reynolds(U_f, U_p, d_p, rho_f, mu_f)
        Re_expected = rho_f * 2.0 * d_p / mu_f
        @test Re ≈ Re_expected atol = 1.0e-12
    end

    # ── 4. RanzMarshall heat transfer ───────────────────────────────
    @testset "RanzMarshall heat transfer" begin
        T_f = 400.0
        T_p = 300.0
        U_f = SVector(1.0, 0.0)
        U_p = SVector(0.0, 0.0)
        q = compute_particle_heat_transfer(
            RanzMarshall(), T_f, T_p, U_f, U_p, d_p, rho_f, mu_f, k_f, Pr_f,
        )
        @test q > 0  # particle heats up

        # Verify Nu >= 2 (minimum Ranz-Marshall value)
        Re_p = FiniteVolumeMethod._particle_reynolds(U_f, U_p, d_p, rho_f, mu_f)
        Nu = 2.0 + 0.6 * Re_p^0.5 * Pr_f^0.33
        @test Nu >= 2.0

        q_expected = pi * d_p * k_f * Nu * (T_f - T_p)
        @test q ≈ q_expected rtol = 1.0e-10
    end

    # ── 5. set_particle_properties! ─────────────────────────────────
    @testset "set_particle_properties!" begin
        tracker = ParticleTracker{2, Float64}()
        inject_particles!(tracker, [SVector(0.5, 0.5)])
        p = tracker.particles[1]
        set_particle_properties!(p; diameter = d_p, density = rho_p, temperature = 350.0, Cp = 800.0)

        @test p.properties[:diameter] == d_p
        @test p.properties[:density] == rho_p
        @test p.properties[:temperature] == 350.0
        @test p.properties[:Cp] == 800.0
        @test p.properties[:mass] ≈ pi / 6 * d_p^3 * rho_p atol = 1.0e-25
    end

    # ── 6. compute_momentum_source ──────────────────────────────────
    @testset "compute_momentum_source" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))

        tracker = ParticleTracker{2, Float64}()
        # Place particle at center of cell 1 (0.125, 0.125)
        inject_particles!(tracker, [SVector(0.125, 0.125)])
        p = tracker.particles[1]
        p.cell_index = 1
        set_particle_properties!(p; diameter = d_p, density = rho_p)

        source = compute_momentum_source(tracker, StokesDrag(), U, rho_f, mu_f, mesh)

        # Drag force on particle
        F_drag = compute_drag_force(
            StokesDrag(), SVector(1.0, 0.0), SVector(0.0, 0.0),
            d_p, rho_p, rho_f, mu_f,
        )
        V_c = mesh.cell_volumes[1]
        expected = -F_drag / V_c

        @test source[1] ≈ expected atol = 1.0e-20
        # Other cells should be zero
        for c in 2:length(source)
            @test source[c] == zero(SVector{2, Float64})
        end
    end

    # ── 7. compute_energy_source ────────────────────────────────────
    @testset "compute_energy_source" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
        T_field = CollocatedScalarField(:T, mesh; value = 400.0)

        tracker = ParticleTracker{2, Float64}()
        inject_particles!(tracker, [SVector(0.125, 0.125)])
        p = tracker.particles[1]
        p.cell_index = 1
        set_particle_properties!(p; diameter = d_p, density = rho_p, temperature = 300.0)

        source = compute_energy_source(
            tracker, RanzMarshall(), T_field, U, rho_f, mu_f, k_f, Pr_f, mesh,
        )

        q = compute_particle_heat_transfer(
            RanzMarshall(), 400.0, 300.0, SVector(1.0, 0.0), SVector(0.0, 0.0),
            d_p, rho_f, mu_f, k_f, Pr_f,
        )
        V_c = mesh.cell_volumes[1]
        expected = -q / V_c

        @test source[1] ≈ expected rtol = 1.0e-10
        for c in 2:length(source)
            @test source[c] == 0.0
        end
    end

    # ── 8. advance_particles! basic ─────────────────────────────────
    @testset "advance_particles! basic" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))

        tracker = ParticleTracker{2, Float64}()
        inject_particles!(tracker, [SVector(0.125, 0.125)])
        p = tracker.particles[1]
        p.cell_index = 1
        set_particle_properties!(p; diameter = d_p, density = rho_p)

        x0 = p.position
        v0 = p.velocity

        dt = 1.0e-6
        advance_particles!(
            tracker, U, mesh, dt;
            drag_model = SchillerNaumann(), rho_f = rho_f, mu_f = mu_f
        )

        # Position should have changed
        @test p.position != x0
        # Velocity should have moved toward fluid velocity (1, 0)
        @test p.velocity[1] > v0[1]
        @test p.active
    end

    # ── 9. advance_particles! with gravity ──────────────────────────
    @testset "advance_particles! with gravity" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        # Quiescent fluid
        U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))

        tracker = ParticleTracker{2, Float64}()
        inject_particles!(tracker, [SVector(0.5, 0.5)])
        p = tracker.particles[1]
        p.cell_index = 6  # cell near center
        set_particle_properties!(p; diameter = d_p, density = rho_p)

        gravity = SVector(0.0, -9.81)
        dt = 1.0e-4
        advance_particles!(
            tracker, U, mesh, dt;
            drag_model = StokesDrag(), rho_f = rho_f, mu_f = mu_f, gravity = gravity
        )

        # Particle should have gained downward velocity
        @test p.velocity[2] < 0.0
        # Position should have moved down
        @test p.position[2] < 0.5
        @test p.active
    end

    # ── 10. advance_particles! deactivation ─────────────────────────
    @testset "advance_particles! deactivation" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        # Strong flow pushing particle out of domain
        U = CollocatedVectorField(:U, mesh; value = SVector(100.0, 0.0))

        tracker = ParticleTracker{2, Float64}()
        # Place particle near right boundary
        inject_particles!(tracker, [SVector(0.875, 0.125)])
        p = tracker.particles[1]
        p.cell_index = 4  # rightmost cell in bottom row
        set_particle_properties!(p; diameter = d_p, density = rho_p)

        # Large dt to push particle well past domain
        dt = 1.0
        advance_particles!(
            tracker, U, mesh, dt;
            drag_model = StokesDrag(), rho_f = rho_f, mu_f = mu_f
        )

        @test !p.active
    end
end

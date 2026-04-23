# test/v_and_v_psi_cell.jl — PSI-cell two-way coupling V&V (v3.94)

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

const _compute_momentum_source = FiniteVolumeMethod.compute_momentum_source
const _set_particle_properties! = FiniteVolumeMethod.set_particle_properties!

function _make_particle(pos::SVector{2, Float64}, vel::SVector{2, Float64}, cell::Int, id::Int)
    return LagrangianParticle{2, Float64}(pos, vel, cell, id, true, Dict{Symbol, Any}())
end

@testset "V&V: PSI-cell — empty tracker ⇒ zero source" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    tracker = ParticleTracker{2, Float64}()
    drag = StokesDrag()
    U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
    source = _compute_momentum_source(tracker, drag, U, 1.0, 1.0e-3, mesh)
    @test length(source) == nc
    for c in 1:nc
        @test source[c] == SVector(0.0, 0.0)
    end
end

@testset "V&V: PSI-cell — zero slip ⇒ zero source" begin
    # If every particle moves with the fluid (U_p = U_f), drag is zero,
    # so the reaction on the fluid is zero.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    tracker = ParticleTracker{2, Float64}()
    drag = StokesDrag()
    U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
    # Inject a particle in every cell with the fluid velocity.
    for c in 1:nc
        p = _make_particle(
            SVector(mesh.cell_centers[1, c], mesh.cell_centers[2, c]),
            SVector(1.0, 0.0), c, c,
        )
        _set_particle_properties!(p; diameter = 1.0e-4, density = 1000.0)
        push!(tracker.particles, p)
    end
    source = _compute_momentum_source(tracker, drag, U, 1.0, 1.0e-3, mesh)
    for c in 1:nc
        @test isapprox(source[c][1], 0.0; atol = 1.0e-12)
        @test isapprox(source[c][2], 0.0; atol = 1.0e-12)
    end
end

@testset "V&V: PSI-cell — inactive particles don't contribute" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    tracker = ParticleTracker{2, Float64}()
    drag = StokesDrag()
    U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
    # Inject with non-trivial slip but mark inactive.
    p = _make_particle(SVector(0.5, 0.5), SVector(0.0, 0.0), 1, 1)
    p.active = false
    _set_particle_properties!(p; diameter = 1.0e-4, density = 1000.0)
    push!(tracker.particles, p)
    source = _compute_momentum_source(tracker, drag, U, 1.0, 1.0e-3, mesh)
    for c in 1:length(mesh.cell_volumes)
        @test source[c] == SVector(0.0, 0.0)
    end
end

@testset "V&V: PSI-cell — out-of-range cell index skipped" begin
    # Particle with cell_index = 0 (before locator) or > nc should not
    # contribute nor error.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    tracker = ParticleTracker{2, Float64}()
    drag = StokesDrag()
    U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
    p0 = _make_particle(SVector(0.5, 0.5), SVector(0.0, 0.0), 0, 1)
    _set_particle_properties!(p0; diameter = 1.0e-4, density = 1000.0)
    push!(tracker.particles, p0)
    p_hi = _make_particle(SVector(0.5, 0.5), SVector(0.0, 0.0), nc + 1, 2)
    _set_particle_properties!(p_hi; diameter = 1.0e-4, density = 1000.0)
    push!(tracker.particles, p_hi)
    source = _compute_momentum_source(tracker, drag, U, 1.0, 1.0e-3, mesh)
    for c in 1:nc
        @test source[c] == SVector(0.0, 0.0)
    end
end

@testset "V&V: PSI-cell — drag reaction opposes slip direction" begin
    # Particle is stationary, fluid moves at +x. Drag on particle is
    # along +x (fluid drags particle forward). Reaction on fluid is −x.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    tracker = ParticleTracker{2, Float64}()
    drag = StokesDrag()
    U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
    p = _make_particle(
        SVector(mesh.cell_centers[1, 1], mesh.cell_centers[2, 1]),
        SVector(0.0, 0.0), 1, 1,
    )
    _set_particle_properties!(p; diameter = 1.0e-4, density = 1000.0)
    push!(tracker.particles, p)
    source = _compute_momentum_source(tracker, drag, U, 1.0, 1.0e-3, mesh)
    # Reaction on fluid at cell 1 is in -x direction.
    @test source[1][1] < 0.0
    @test isapprox(source[1][2], 0.0; atol = 1.0e-12)
    # All other cells zero.
    for c in 2:nc
        @test source[c] == SVector(0.0, 0.0)
    end
end

@testset "V&V: PSI-cell — two particles in same cell sum up linearly" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    tracker = ParticleTracker{2, Float64}()
    drag = StokesDrag()
    U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
    # One particle.
    p1 = _make_particle(
        SVector(mesh.cell_centers[1, 1], mesh.cell_centers[2, 1]),
        SVector(0.0, 0.0), 1, 1,
    )
    _set_particle_properties!(p1; diameter = 1.0e-4, density = 1000.0)
    push!(tracker.particles, p1)
    source1 = copy(_compute_momentum_source(tracker, drag, U, 1.0, 1.0e-3, mesh))
    # Add a second identical particle in the same cell.
    p2 = _make_particle(
        SVector(mesh.cell_centers[1, 1], mesh.cell_centers[2, 1]),
        SVector(0.0, 0.0), 1, 2,
    )
    _set_particle_properties!(p2; diameter = 1.0e-4, density = 1000.0)
    push!(tracker.particles, p2)
    source2 = _compute_momentum_source(tracker, drag, U, 1.0, 1.0e-3, mesh)
    @test isapprox(source2[1][1], 2.0 * source1[1][1]; rtol = 1.0e-12)
    for c in 2:nc
        @test source2[c] == SVector(0.0, 0.0)
    end
end

@testset "V&V: PSI-cell — slip-direction reversal flips source sign" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    tracker = ParticleTracker{2, Float64}()
    drag = StokesDrag()
    U_pos = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
    U_neg = CollocatedVectorField(:U, mesh; value = SVector(-1.0, 0.0))
    p = _make_particle(
        SVector(mesh.cell_centers[1, 1], mesh.cell_centers[2, 1]),
        SVector(0.0, 0.0), 1, 1,
    )
    _set_particle_properties!(p; diameter = 1.0e-4, density = 1000.0)
    push!(tracker.particles, p)
    source_pos = _compute_momentum_source(tracker, drag, U_pos, 1.0, 1.0e-3, mesh)
    source_neg = _compute_momentum_source(tracker, drag, U_neg, 1.0, 1.0e-3, mesh)
    # Flipping slip reverses both drag and reaction.
    @test isapprox(source_neg[1][1], -source_pos[1][1]; rtol = 1.0e-12)
end

@testset "V&V: PSI-cell — set_particle_properties! mass closed form" begin
    p = _make_particle(SVector(0.0, 0.0), SVector(0.0, 0.0), 1, 1)
    _set_particle_properties!(p; diameter = 1.0e-3, density = 2000.0)
    expected_mass = pi / 6 * (1.0e-3)^3 * 2000.0
    @test p.properties[:mass] ≈ expected_mass rtol = 1.0e-14
    @test p.properties[:diameter] == 1.0e-3
    @test p.properties[:density] == 2000.0
    @test p.properties[:temperature] == 300.0   # default
    @test p.properties[:Cp] == 1000.0           # default
end

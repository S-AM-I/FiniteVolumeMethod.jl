# test/v_and_v_mesh_update.jl — update_mesh! geometry-update V&V (v3.80)

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: update_mesh! — V_old copied before cell-center move" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    V_before = copy(mesh.cell_volumes)

    # Uniform translation; call update_mesh!
    for c in 1:length(mesh.cell_volumes)
        ms.displacement[c] = SVector(0.1, 0.0)
    end
    update_mesh!(mesh, ms, 0.1)

    # V_old should hold the pre-update volumes.
    for c in 1:length(V_before)
        @test isapprox(ms.V_old[c], V_before[c]; rtol = 1.0e-14)
    end
end

@testset "V&V: update_mesh! — cell centers shift by prescribed displacement" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    d0 = SVector(0.25, -0.1)
    for c in 1:length(mesh.cell_volumes)
        ms.displacement[c] = d0
    end
    x_before = copy(mesh.cell_centers)

    update_mesh!(mesh, ms, 0.1)

    for c in 1:length(mesh.cell_volumes)
        @test isapprox(mesh.cell_centers[1, c], x_before[1, c] + d0[1]; rtol = 1.0e-14)
        @test isapprox(mesh.cell_centers[2, c], x_before[2, c] + d0[2]; rtol = 1.0e-14)
    end
end

@testset "V&V: update_mesh! — phi_mesh populated after call" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    @test all(==(0.0), ms.phi_mesh)   # initial

    for c in 1:length(mesh.cell_volumes)
        ms.displacement[c] = SVector(0.2, 0.0)
    end
    update_mesh!(mesh, ms, 0.1)

    # After update_mesh!, phi_mesh should have non-zero values on
    # streamwise faces (where dot(d, S_f) ≠ 0).
    non_zero_count = count(!=(0.0), ms.phi_mesh)
    @test non_zero_count > 0
end

@testset "V&V: update_mesh! — zero dt leaves phi_mesh = 0" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    for c in 1:length(mesh.cell_volumes)
        ms.displacement[c] = SVector(0.1, 0.0)
    end
    # compute_mesh_flux! handles dt ≤ 0 by returning early with
    # phi_mesh zeroed.
    FiniteVolumeMethod.compute_mesh_flux!(ms, mesh, 0.0)
    @test all(==(0.0), ms.phi_mesh)
end

@testset "V&V: update_mesh! — uniform translation preserves volumes exactly" begin
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    V_before = copy(mesh.cell_volumes)

    d0 = SVector(0.3, 0.2)
    for c in 1:length(mesh.cell_volumes)
        ms.displacement[c] = d0
    end
    update_mesh!(mesh, ms, 0.1)

    for c in 1:length(V_before)
        @test isapprox(mesh.cell_volumes[c], V_before[c]; atol = 1.0e-12)
    end
end

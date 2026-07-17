# test/v_and_v_courant_edge.jl — Courant number edge-case V&V (v3.84)

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_courant_number
using Test

include("TestHelpers.jl")

@testset "V&V: Courant — dt-linear scaling" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    phi = FaceFluxField(:phi, mesh)
    for f in 1:size(mesh.face_cells, 2)
        phi.values[f] = 0.5
    end
    Co_1 = compute_courant_number(phi, mesh, 0.01)
    Co_2 = compute_courant_number(phi, mesh, 0.02)
    for c in 1:length(mesh.cell_volumes)
        @test isapprox(Co_2[c] / Co_1[c], 2.0; rtol = 1.0e-12)
    end
end

@testset "V&V: Courant — sign-symmetric (|phi|)" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    phi_pos = FaceFluxField(:phi, mesh)
    phi_neg = FaceFluxField(:phi, mesh)
    for f in 1:size(mesh.face_cells, 2)
        phi_pos.values[f] = 0.3
        phi_neg.values[f] = -0.3
    end
    Co_pos = compute_courant_number(phi_pos, mesh, 0.01)
    Co_neg = compute_courant_number(phi_neg, mesh, 0.01)
    # Co uses |phi|, so sign flip gives identical Co.
    for c in 1:length(mesh.cell_volumes)
        @test isapprox(Co_pos[c], Co_neg[c]; rtol = 1.0e-14)
    end
end

@testset "V&V: Courant — zero dt ⇒ Co ≡ 0" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    phi = FaceFluxField(:phi, mesh; value = 1.0)
    Co = compute_courant_number(phi, mesh, 0.0)
    for c in 1:length(mesh.cell_volumes)
        @test Co[c] == 0.0
    end
end

@testset "V&V: Courant — Co ≥ 0 for arbitrary phi" begin
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    phi = FaceFluxField(:phi, mesh)
    for f in 1:size(mesh.face_cells, 2)
        phi.values[f] = sin(0.2 * f) * (-1)^f
    end
    Co = compute_courant_number(phi, mesh, 0.005)
    for c in 1:length(mesh.cell_volumes)
        @test Co[c] >= 0.0
    end
end

@testset "V&V: Courant — V_c inverse scaling (finer mesh ⇒ bigger Co)" begin
    # For same flux, halving V_c doubles Co.
    phi_val = 0.25
    dt = 0.01

    # Larger mesh: larger V_c ⇒ smaller Co.
    mesh_a = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    mesh_b = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    phi_a = FaceFluxField(:phi, mesh_a; value = phi_val)
    phi_b = FaceFluxField(:phi, mesh_b; value = phi_val)

    Co_a = compute_courant_number(phi_a, mesh_a, dt)
    Co_b = compute_courant_number(phi_b, mesh_b, dt)

    # Max over interior cells.
    # V_b = V_a / 4 ⇒ Co_b = 4 · Co_a (approximately).
    m_a = maximum(Co_a)
    m_b = maximum(Co_b)
    @test m_b > m_a
end

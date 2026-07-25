# test/v_and_v_ale_flux.jl — ALE-corrected flux V&V (v3.49)
#
# Fourth convergence-verified benchmark for `dynamic_mesh`.
# Complements three-pattern GCL (v3.14), rotational GCL (v3.29),
# and mesh sweep-flux (v3.34) with the transport-side primitive
# `ale_corrected_flux`:
#
#   φ_ale[f] = φ[f] − φ_mesh[f]
#
# This is the flux consumed by transport equations on a moving
# mesh: the effective flux a scalar sees after removing the
# mesh-velocity contribution. Five invariants verified.

using FiniteVolumeMethod
using FiniteVolumeMethod: ale_corrected_flux
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: ALE flux — zero φ_mesh ⇒ φ_ale = φ" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)

    phi = FaceFluxField(:phi, mesh)
    for f in 1:nf
        phi.values[f] = Float64(f) * 0.01
    end

    phi_mesh = zeros(Float64, nf)
    phi_ale = ale_corrected_flux(phi, phi_mesh)

    for f in 1:nf
        @test isapprox(phi_ale.values[f], phi.values[f]; rtol = 1.0e-14)
    end
end

@testset "V&V: ALE flux — zero φ ⇒ φ_ale = -φ_mesh" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)

    phi = FaceFluxField(:phi, mesh; value = 0.0)
    phi_mesh = fill(0.3, nf)

    phi_ale = ale_corrected_flux(phi, phi_mesh)

    for f in 1:nf
        @test isapprox(phi_ale.values[f], -0.3; rtol = 1.0e-14)
    end
end

@testset "V&V: ALE flux — algebraic identity at every face" begin
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)

    phi = FaceFluxField(:phi, mesh)
    phi_mesh = zeros(Float64, nf)
    for f in 1:nf
        phi.values[f] = sin(0.1 * f) * 2.5
        phi_mesh[f] = cos(0.1 * f) * 0.7
    end

    phi_ale = ale_corrected_flux(phi, phi_mesh)

    for f in 1:nf
        @test isapprox(phi_ale.values[f], phi.values[f] - phi_mesh[f]; rtol = 1.0e-14)
    end
end

@testset "V&V: ALE flux — Eulerian limit (moving-frame = same-frame)" begin
    # When flow and mesh move with the same velocity (φ == φ_mesh),
    # ALE-corrected flux must be zero — the fluid is at rest in
    # the moving frame.
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)

    phi = FaceFluxField(:phi, mesh)
    phi_mesh = zeros(Float64, nf)
    for f in 1:nf
        phi.values[f] = 0.15 * sin(f * 0.2)
        phi_mesh[f] = phi.values[f]
    end

    phi_ale = ale_corrected_flux(phi, phi_mesh)

    for f in 1:nf
        @test isapprox(phi_ale.values[f], 0.0; atol = 1.0e-14)
    end
end

@testset "V&V: ALE flux — dimension mismatch errors" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)

    phi = FaceFluxField(:phi, mesh; value = 1.0)
    phi_mesh_wrong = zeros(Float64, nf + 1)

    @test_throws ErrorException ale_corrected_flux(phi, phi_mesh_wrong)
end

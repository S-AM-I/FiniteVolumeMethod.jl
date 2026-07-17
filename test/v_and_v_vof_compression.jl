# test/v_and_v_vof_compression.jl — VOF compression flux V&V (v3.59)
#
# Fifth convergence-verified benchmark for `multiphase_vof`,
# joining disc translation (v3.16), plane-wave (v3.24), mixture
# blending (v3.36), and CSF surface tension (v3.46). Covers the
# interface-compression flux primitive used to counter upwind
# smearing:
#
#   φ_c[f] = C_α · |φ_f| · (n̂ · S_f) / |S_f|
#
# Four invariants verified.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_compression_flux, face_normal_area
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: VOF compression — C_α = 0 ⇒ φ_c ≡ 0 (but nonzero Cα path)" begin
    # `compute_compression_flux` with Cα=0 returns all-zero
    # compression flux (multiplier is zero).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        alpha.internal[c] = mesh.cell_centers[1, c]    # non-trivial field
    end

    nf = size(mesh.face_cells, 2)
    phi = FaceFluxField(:phi, mesh)
    for f in 1:nf
        phi.values[f] = face_normal_area(mesh, f)[1]
    end

    phi_c = compute_compression_flux(alpha, phi, mesh; C_alpha = 0.0)
    for f in 1:nf
        @test phi_c[f] == 0.0
    end
end

@testset "V&V: VOF compression — uniform α ⇒ φ_c ≡ 0 (no interface)" begin
    # With uniform α, ∇α = 0 everywhere, so interface normal is
    # undefined (falls back to zero) and φ_c = 0 at every face.
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    alpha = CollocatedScalarField(:alpha, mesh; value = 0.3)

    phi = FaceFluxField(:phi, mesh)
    for f in 1:nf
        phi.values[f] = face_normal_area(mesh, f)[1]
    end

    phi_c = compute_compression_flux(alpha, phi, mesh; C_alpha = 1.0)
    for f in 1:nf
        @test isapprox(phi_c[f], 0.0; atol = 1.0e-12)
    end
end

@testset "V&V: VOF compression — C_α linear scaling" begin
    # At fixed α and φ, doubling C_α doubles φ_c at every face
    # that has non-trivial compression.
    mesh = build_cartesian_unstructured_mesh(20, 20, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        alpha.internal[c] = 0.5 * (1 + tanh(20.0 * (x - 0.5)))
    end

    phi = FaceFluxField(:phi, mesh)
    for f in 1:nf
        phi.values[f] = face_normal_area(mesh, f)[1]
    end

    phi_c_1 = compute_compression_flux(alpha, phi, mesh; C_alpha = 0.5)
    phi_c_2 = compute_compression_flux(alpha, phi, mesh; C_alpha = 1.0)
    phi_c_4 = compute_compression_flux(alpha, phi, mesh; C_alpha = 2.0)

    count_checked = 0
    for f in 1:nf
        if abs(phi_c_1[f]) > 1.0e-10
            @test isapprox(phi_c_2[f] / phi_c_1[f], 2.0; rtol = 1.0e-12)
            @test isapprox(phi_c_4[f] / phi_c_1[f], 4.0; rtol = 1.0e-12)
            count_checked += 1
        end
    end
    @test count_checked > 0   # at least some faces have compression
end

@testset "V&V: VOF compression — |φ_f| ∝ scaling at fixed α, Cα" begin
    # The compression flux is proportional to |φ_f| (flux
    # magnitude). Doubling all φ values doubles φ_c.
    mesh = build_cartesian_unstructured_mesh(20, 20, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        alpha.internal[c] = 0.5 * (1 + tanh(15.0 * (x - 0.5)))
    end

    phi_1 = FaceFluxField(:phi, mesh)
    phi_2 = FaceFluxField(:phi, mesh)
    for f in 1:nf
        phi_1.values[f] = face_normal_area(mesh, f)[1]
        phi_2.values[f] = 2.0 * phi_1.values[f]
    end

    phi_c_1 = compute_compression_flux(alpha, phi_1, mesh; C_alpha = 1.0)
    phi_c_2 = compute_compression_flux(alpha, phi_2, mesh; C_alpha = 1.0)

    count_checked = 0
    for f in 1:nf
        if abs(phi_c_1[f]) > 1.0e-10
            @test isapprox(phi_c_2[f] / phi_c_1[f], 2.0; rtol = 1.0e-12)
            count_checked += 1
        end
    end
    @test count_checked > 0
end

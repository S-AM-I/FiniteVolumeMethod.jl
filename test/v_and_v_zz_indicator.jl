# test/v_and_v_zz_indicator.jl — ZZ recovery-based error indicator V&V.

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

@testset "V&V: ZZ indicator — constant field ⇒ η ≈ 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi = CollocatedScalarField(:phi, mesh; value = 3.0)
    eta = zz_error_indicator(phi, mesh)
    @test length(eta) == nc
    for v in eta
        @test v < 1.0e-10
    end
end

@testset "V&V: ZZ indicator — linear field ⇒ small η" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi = CollocatedScalarField(:phi, mesh; value = 0.0)
    for c in 1:nc
        phi.internal[c] = 2.0 * mesh.cell_centers[1, c] + 3.0 * mesh.cell_centers[2, c]
    end
    eta = zz_error_indicator(phi, mesh)
    # Linear-field discrepancy should be small compared to the field scale.
    @test maximum(eta) < 1.0
end

@testset "V&V: ZZ indicator — non-negative everywhere" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi = CollocatedScalarField(:phi, mesh; value = 0.0)
    for c in 1:nc
        phi.internal[c] = sin(4 * mesh.cell_centers[1, c]) * cos(4 * mesh.cell_centers[2, c])
    end
    eta = zz_error_indicator(phi, mesh)
    for v in eta
        @test v >= 0.0
    end
end

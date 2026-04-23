# test/v_and_v_les_dispatch.jl — LES model output-consistency V&V (v3.73)

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: LES — all three models return ν_t of right length" begin
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        U.internal[c] = SVector(mesh.cell_centers[2, c], 0.0)
    end

    for model in (
            Smagorinsky(mesh; Cs = 0.1), WALE(mesh; Cw = 0.325),
            DynamicSmagorinsky(mesh),
        )
        nu_t = zeros(Float64, nc)
        FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)
        @test length(nu_t) == nc
        @test all(isfinite, nu_t)
    end
end

@testset "V&V: LES — ν_t finite across broad flow magnitudes" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = Smagorinsky(mesh; Cs = 0.1)

    for A in (1.0e-3, 1.0, 1.0e3, 1.0e6)
        U = CollocatedVectorField(:U, mesh)
        for c in 1:nc
            U.internal[c] = SVector(A * mesh.cell_centers[2, c], 0.0)
        end
        nu_t = zeros(Float64, nc)
        FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)
        @test all(isfinite, nu_t)
        @test all(>=(0.0), nu_t)
    end
end

@testset "V&V: LES — ν_t ≥ 0 on biaxial stretching" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(0.5 * x, -0.5 * y)
    end
    for model in (Smagorinsky(mesh; Cs = 0.1), WALE(mesh; Cw = 0.325))
        nu_t = zeros(Float64, nc)
        FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)
        @test all(>=(0.0), nu_t)
    end
end

@testset "V&V: LES — Smagorinsky Cs=0 ⇒ ν_t = 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = Smagorinsky(mesh; Cs = 0.0)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        U.internal[c] = SVector(mesh.cell_centers[2, c], 0.0)
    end
    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)
    @test all(==(0.0), nu_t)
end

@testset "V&V: LES — WALE Cw=0 ⇒ ν_t = 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = WALE(mesh; Cw = 0.0)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(x * y, 0.0)
    end
    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)
    @test all(==(0.0), nu_t)
end

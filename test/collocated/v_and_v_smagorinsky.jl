# test/v_and_v_smagorinsky.jl — Smagorinsky LES V&V (v3.19)
#
# Verifies the Smagorinsky subgrid-scale viscosity formula
#
#   ν_t = (C_s · Δ)² · |S|
#
# against its analytical value on two prescribed velocity fields:
#
#   1. Zero velocity  → |S| = 0 → ν_t ≡ 0 (trivial invariance).
#   2. Linear shear U = (A·y, 0) → |S| = A (exact on a Cartesian
#      mesh with the present gradient discretization) →
#      ν_t = (C_s · Δ)² · A uniform on the interior.
#
# A third gate establishes the cubic scaling ν_t ∝ C_s² under C_s
# variation at fixed mesh, an algebraic invariant of the model.
# Evidence for promoting `turbulence_les` from `experimental`/
# `smoke_tested` to `provisional`/`convergence_verified`.

using FiniteVolumeMethod
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: Smagorinsky — zero velocity ⇒ ν_t ≡ 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = Smagorinsky(mesh; Cs = 0.1)

    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))

    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

    # |S| ≡ 0 ⇒ ν_t ≡ 0 to round-off.
    @test all(isapprox.(nu_t, 0.0; atol = 1.0e-14))
end

@testset "V&V: Smagorinsky — linear shear ν_t = (Cs·Δ)²·A" begin
    # U(x, y) = (A·y, 0). Strain tensor components:
    #   S_xy = S_yx = (1/2)(∂U/∂y + ∂V/∂x) = A/2
    #   all others zero.
    # |S| = √(2 S_ij S_ij) = √(2 · 2 · (A/2)²) = A.
    A = 3.0
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = Smagorinsky(mesh; Cs = 0.1)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(A * y, 0.0)
    end

    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

    # Expected: ν_t = (Cs · Δ)² · A per cell.
    # Δ varies per cell if the mesh is non-uniform; on a 16×16
    # Cartesian mesh with unit aspect ratio it is uniform.
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        # Interior cells only — boundary cells carry discretization
        # error in the gradient stencil which changes the apparent
        # ∂U/∂y slightly from A.
        if 0.2 < x < 0.8 && 0.2 < y < 0.8
            expected = (model.Cs * model.delta[c])^2 * A
            @test isapprox(nu_t[c], expected; rtol = 1.0e-8)
        end
    end

    # All ν_t ≥ 0 (realizability of LES closure).
    @test all(>=(0.0), nu_t)
end

@testset "V&V: Smagorinsky — ν_t ∝ C_s² at fixed flow" begin
    # At fixed A and Δ, varying C_s changes ν_t purely quadratically.
    # Scale invariance: ν_t(2·C_s) / ν_t(C_s) = 4.
    A = 2.0
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        U.internal[c] = SVector(A * mesh.cell_centers[2, c], 0.0)
    end

    models = (Smagorinsky(mesh; Cs = 0.05), Smagorinsky(mesh; Cs = 0.1), Smagorinsky(mesh; Cs = 0.2))
    nus = [zeros(Float64, nc) for _ in 1:3]
    for i in 1:3
        FiniteVolumeMethod.turbulent_viscosity!(nus[i], models[i], U, mesh)
    end

    # Pick an interior cell.
    c_probe = 0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.4 < x < 0.6 && 0.4 < y < 0.6
            c_probe = c
            break
        end
    end
    @test c_probe != 0

    r1 = nus[2][c_probe] / nus[1][c_probe]  # 0.10² / 0.05² = 4
    r2 = nus[3][c_probe] / nus[2][c_probe]  # 0.20² / 0.10² = 4

    @test isapprox(r1, 4.0; rtol = 1.0e-10)
    @test isapprox(r2, 4.0; rtol = 1.0e-10)
end

@testset "V&V: Smagorinsky — filter width scales as Δ²" begin
    # ν_t ∝ Δ² at fixed Cs, |S|. Compare coarse vs. fine mesh at
    # the same linear-shear flow; Δ_fine / Δ_coarse = 1/2 ⇒ the
    # ν_t ratio should be (1/2)² = 1/4.
    A = 1.0
    Cs = 0.1

    results = Float64[]
    for N in (8, 16)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        model = Smagorinsky(mesh; Cs = Cs)
        U = CollocatedVectorField(:U, mesh)
        for c in 1:nc
            U.internal[c] = SVector(A * mesh.cell_centers[2, c], 0.0)
        end
        nu_t = zeros(Float64, nc)
        FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

        # Interior sample.
        sample = 0.0
        count = 0
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            if 0.3 < x < 0.7 && 0.3 < y < 0.7
                sample += nu_t[c]
                count += 1
            end
        end
        push!(results, sample / count)
    end

    ratio = results[2] / results[1]  # fine / coarse
    @test isapprox(ratio, 0.25; rtol = 1.0e-8)
end

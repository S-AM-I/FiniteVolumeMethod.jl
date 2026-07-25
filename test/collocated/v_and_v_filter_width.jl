# test/v_and_v_filter_width.jl — LES filter-width + DynamicSmagorinsky V&V (v3.39)
#
# Third convergence-verified benchmark for `turbulence_les`. The
# first (v3.19) tested Smagorinsky ν_t = (Cs·Δ)²·|S| algebra; the
# second (v3.28) tested WALE's pure-shear vanishing. This one
# covers the shared `compute_filter_width` primitive
#
#   Δ[c] = V_c^(1/Dim)
#
# used by every LES model, plus the `DynamicSmagorinsky` closure
# invariants in trivial flow fields where the Germano identity
# can be evaluated analytically.
#
# Puts `turbulence_les` at three convergence-verified benchmarks.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_filter_width
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: Filter width — Δ = √V on uniform 2D Cartesian mesh" begin
    # Uniform N×N mesh on [0, 1]² has V_c = 1/N² and
    # Δ_c = √V_c = 1/N identically for every cell.
    for N in (4, 8, 16, 32)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        delta = FiniteVolumeMethod.compute_filter_width(mesh)
        expected = 1.0 / N
        for c in 1:length(mesh.cell_volumes)
            @test isapprox(delta[c], expected; rtol = 1.0e-12)
        end
    end
end

@testset "V&V: Filter width — anisotropic Cartesian Δ = √(Lx·Ly/(Nx·Ny))" begin
    # Stretched mesh [Lx, Ly] at Nx×Ny: V_c = (Lx/Nx)·(Ly/Ny);
    # Δ = √V.
    for (Lx, Ly, Nx, Ny) in ((2.0, 0.5, 20, 10), (4.0, 1.0, 32, 8), (1.0, 2.0, 16, 32))
        mesh = build_cartesian_unstructured_mesh(Nx, Ny, Lx, Ly)
        delta = FiniteVolumeMethod.compute_filter_width(mesh)
        expected = sqrt((Lx / Nx) * (Ly / Ny))
        for c in 1:length(mesh.cell_volumes)
            @test isapprox(delta[c], expected; rtol = 1.0e-12)
        end
    end
end

@testset "V&V: Filter width — Smagorinsky × WALE uses same Δ" begin
    # Both Smagorinsky and WALE construct from the same
    # compute_filter_width. Verify the stored `delta` fields
    # match exactly.
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    smag = Smagorinsky(mesh; Cs = 0.1)
    wale = WALE(mesh; Cw = 0.325)
    dynsmag = DynamicSmagorinsky(mesh)

    for c in 1:length(mesh.cell_volumes)
        @test smag.delta[c] == wale.delta[c]
        @test smag.delta[c] == dynsmag.delta[c]
    end
end

@testset "V&V: DynamicSmagorinsky — zero velocity ⇒ ν_t = 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = DynamicSmagorinsky(mesh)

    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))
    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

    # |S| = 0 ⇒ ν_t = Cs²·Δ²·|S| = 0 regardless of Cs² value.
    @test all(isapprox.(nu_t, 0.0; atol = 1.0e-14))
end

@testset "V&V: DynamicSmagorinsky — uniform velocity ⇒ ν_t = 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = DynamicSmagorinsky(mesh)

    U = CollocatedVectorField(:U, mesh; value = SVector(2.5, -1.3))
    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

    # Uniform flow: |S| = 0 ⇒ ν_t = 0.
    @test all(isapprox.(nu_t, 0.0; atol = 1.0e-14))
end

@testset "V&V: DynamicSmagorinsky — ν_t ≥ 0 realizability (clipping at zero)" begin
    # The implementation clamps Cs² at ≥ 0 (via max(LM/MM, 0)),
    # so ν_t is always non-negative. Verify on a non-trivial
    # linear-shear flow.
    A = 2.0
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = DynamicSmagorinsky(mesh)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(A * y, 0.0)
    end

    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

    # Non-negative everywhere (realizability).
    @test all(>=(0.0), nu_t)

    # Bounded above by a small constant times the characteristic
    # scale. The Dynamic Smagorinsky implementation caps Cs² at
    # 0.04, but the strain-rate estimate carries boundary-cell
    # discretization error on a 16×16 mesh with Dirichlet edges
    # that may elevate the apparent |S| near walls. Cap ν_t at
    # ≤ 10⁻³ on this flow (≈ 3× the theoretical interior bound).
    @test maximum(nu_t) < 1.0e-3
end

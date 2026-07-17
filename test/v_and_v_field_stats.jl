# test/v_and_v_field_stats.jl — Field statistics + TI V&V (v3.40)
#
# Third convergence-verified benchmark for `postprocessing`. The
# first (v3.20) tested vorticity + Q on canonical flows; the
# second (v3.30) tested Courant + Q-sign discrimination. This one
# completes coverage of the scalar-field statistics primitives:
#
#   • `field_average(field, mesh)` — volume-weighted mean.
#   • `field_min_max(field)` — extrema.
#   • `turbulence_intensity(k, U_mean)` — TI = √(2k/3) / U_mean.
#
# Puts `postprocessing` at three convergence-verified benchmarks,
# bringing every provisional physics feature in the repository
# to the 3-benchmark stable-review floor.

using FiniteVolumeMethod
using FiniteVolumeMethod: field_average, field_min_max, turbulence_intensity
using Test

include("TestHelpers.jl")

@testset "V&V: field_average — constant field returns constant" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    field = CollocatedScalarField(:f, mesh; value = 2.5)
    @test isapprox(field_average(field, mesh), 2.5; rtol = 1.0e-14)
end

@testset "V&V: field_average — linear field f(x) = x averages to 0.5 on [0, 1]²" begin
    mesh = build_cartesian_unstructured_mesh(32, 32, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    field = CollocatedScalarField(:f, mesh)
    for c in 1:nc
        field.internal[c] = mesh.cell_centers[1, c]
    end
    # Cell centers of a 32×32 uniform mesh lie at x = (i-0.5)/32,
    # whose cell-volume-weighted mean over [0, 1] is exactly 0.5.
    @test isapprox(field_average(field, mesh), 0.5; rtol = 1.0e-12)
end

@testset "V&V: field_average — volume-weighted (anisotropic domain)" begin
    # On a stretched mesh, the volume-weighted mean should still
    # collapse the scaling. f(x) = 2·x on [0, 3] × [0, 1] gives
    # mean = 2 · 1.5 = 3.
    mesh = build_cartesian_unstructured_mesh(30, 10, 3.0, 1.0)
    nc = length(mesh.cell_volumes)
    field = CollocatedScalarField(:f, mesh)
    for c in 1:nc
        field.internal[c] = 2 * mesh.cell_centers[1, c]
    end
    @test isapprox(field_average(field, mesh), 3.0; rtol = 1.0e-12)
end

@testset "V&V: field_min_max — recovers exact extrema" begin
    mesh = build_cartesian_unstructured_mesh(20, 20, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    field = CollocatedScalarField(:f, mesh)
    for c in 1:nc
        field.internal[c] = sin(2 * pi * mesh.cell_centers[1, c])
    end

    mn, mx = field_min_max(field)
    @test mn ≤ 0.0
    @test mx ≥ 0.0
    # Peaks of sin(2πx) at x = 0.25 and x = 0.75; cell-centered
    # samples miss the exact peaks but come within 1 − cos(2π/(2N)).
    @test mx > 0.95
    @test mn < -0.95
end

@testset "V&V: turbulence_intensity — √(2k/3)/U_mean identity" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    k_field = CollocatedScalarField(:k, mesh; value = 0.15)

    U_mean = 10.0
    TI = turbulence_intensity(k_field, U_mean)

    expected = sqrt(2 * 0.15 / 3) / U_mean  # ≈ 0.0316
    for c in 1:nc
        @test isapprox(TI[c], expected; rtol = 1.0e-12)
    end
end

@testset "V&V: turbulence_intensity — U_mean ≤ 0 returns zero" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    k_field = CollocatedScalarField(:k, mesh; value = 0.5)

    TI_zero = turbulence_intensity(k_field, 0.0)
    TI_neg = turbulence_intensity(k_field, -1.0)

    @test all(==(0.0), TI_zero)
    @test all(==(0.0), TI_neg)
end

@testset "V&V: turbulence_intensity — √k scaling at fixed U_mean" begin
    # TI ∝ √k. Quadrupling k doubles TI.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    k_a = CollocatedScalarField(:k, mesh; value = 0.25)
    k_b = CollocatedScalarField(:k, mesh; value = 1.0)

    TI_a = turbulence_intensity(k_a, 5.0)
    TI_b = turbulence_intensity(k_b, 5.0)

    for c in 1:length(TI_a)
        @test isapprox(TI_b[c] / TI_a[c], 2.0; rtol = 1.0e-12)
    end
end

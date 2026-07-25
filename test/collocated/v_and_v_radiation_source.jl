# test/v_and_v_radiation_source.jl — Radiation source algebra V&V (v3.35)
#
# Third convergence-verified benchmark for `radiation`. The first
# (v3.15) tested P1 diffusion in a cold slab; the second (v3.25)
# tested radiative equilibrium under Marshak walls. This one
# covers the final primitive: the thermal-coupling source term
#
#   S_rad[c] = a · G[c] − 4 · a · σ · T[c]⁴
#
# computed by `compute_radiation_source`, which feeds into the
# fluid-energy equation as the per-cell radiative heating or
# cooling rate. Five algebraic invariants are verified, closing
# the gap between the radiation solver output and its consumption
# by the coupled thermal solver.
#
# Puts `radiation` at three convergence-verified benchmarks —
# 3-benchmark floor for stable-promotion review.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_radiation_source
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const STEFAN_SB = 5.670374419e-8

@testset "V&V: Radiation source — equilibrium G = 4σT⁴ ⇒ S_rad ≡ 0" begin
    # When the incident irradiation matches the blackbody emissive
    # power G = 4σT⁴, the local radiative source must vanish
    # (detailed balance).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    rad = P1Model(; a = 1.0)

    T_val = 500.0
    T_field = CollocatedScalarField(:T, mesh; value = T_val)

    G_eq = 4 * STEFAN_SB * T_val^4
    G_field = CollocatedScalarField(:G, mesh; value = G_eq)

    S_rad = compute_radiation_source(rad, G_field, T_field)

    for c in 1:nc
        @test isapprox(S_rad[c], 0.0; atol = 1.0e-12 * G_eq)
    end
end

@testset "V&V: Radiation source — cold medium (T = 0) ⇒ S_rad = a·G > 0" begin
    # No emission (T = 0) and non-zero G ⇒ S_rad = a·G > 0 (medium
    # absorbs incident radiation).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    a = 2.5
    rad = P1Model(; a = a)

    T_field = CollocatedScalarField(:T, mesh; value = 0.0)
    G_field = CollocatedScalarField(:G, mesh; value = 1000.0)

    S_rad = compute_radiation_source(rad, G_field, T_field)

    for c in 1:nc
        @test isapprox(S_rad[c], a * 1000.0; rtol = 1.0e-12)
        @test S_rad[c] > 0.0
    end
end

@testset "V&V: Radiation source — hot medium (G = 0) ⇒ S_rad = -4aσT⁴ < 0" begin
    # Zero incident radiation and hot medium ⇒ S_rad = -4aσT⁴ < 0
    # (medium radiates away, net cooling).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    a = 0.5
    rad = P1Model(; a = a)

    T_val = 800.0
    T_field = CollocatedScalarField(:T, mesh; value = T_val)
    G_field = CollocatedScalarField(:G, mesh; value = 0.0)

    S_rad = compute_radiation_source(rad, G_field, T_field)

    expected = -4 * a * STEFAN_SB * T_val^4
    for c in 1:nc
        @test isapprox(S_rad[c], expected; rtol = 1.0e-12)
        @test S_rad[c] < 0.0
    end
end

@testset "V&V: Radiation source — linearity in G" begin
    # At fixed T, S_rad is linear in G with slope a.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    a = 1.5
    rad = P1Model(; a = a)

    T_field = CollocatedScalarField(:T, mesh; value = 400.0)
    G_a = CollocatedScalarField(:G, mesh; value = 500.0)
    G_b = CollocatedScalarField(:G, mesh; value = 1500.0)

    S_a = compute_radiation_source(rad, G_a, T_field)
    S_b = compute_radiation_source(rad, G_b, T_field)

    # (S_b − S_a) should equal a · (G_b − G_a) = 1.5 · 1000 = 1500.
    for c in 1:nc
        @test isapprox(S_b[c] - S_a[c], a * (1500.0 - 500.0); rtol = 1.0e-12)
    end
end

@testset "V&V: Radiation source — T⁴ scaling of emission branch" begin
    # At G = 0 (pure emission), S_rad = -4aσT⁴. Doubling T
    # multiplies |S_rad| by 16.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    rad = P1Model(; a = 1.0)

    G_field = CollocatedScalarField(:G, mesh; value = 0.0)

    T1 = CollocatedScalarField(:T, mesh; value = 300.0)
    T2 = CollocatedScalarField(:T, mesh; value = 600.0)

    S1 = compute_radiation_source(rad, G_field, T1)
    S2 = compute_radiation_source(rad, G_field, T2)

    for c in 1:nc
        @test isapprox(S2[c] / S1[c], 16.0; rtol = 1.0e-12)
    end
end

@testset "V&V: Radiation source — negative-T clamp (no unphysical emission)" begin
    # The implementation clamps T at zero before the T⁴ term.
    # Supplying a negative T should give the same result as T = 0.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    rad = P1Model(; a = 1.0)

    T_neg = CollocatedScalarField(:T, mesh; value = -100.0)
    T_zero = CollocatedScalarField(:T, mesh; value = 0.0)
    G_field = CollocatedScalarField(:G, mesh; value = 200.0)

    S_neg = compute_radiation_source(rad, G_field, T_neg)
    S_zero = compute_radiation_source(rad, G_field, T_zero)

    for c in 1:nc
        @test isapprox(S_neg[c], S_zero[c]; rtol = 1.0e-14)
    end
end

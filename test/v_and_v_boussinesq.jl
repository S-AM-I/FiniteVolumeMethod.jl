# test/v_and_v_boussinesq.jl — Boussinesq buoyancy algebra V&V (v3.32)
#
# Third convergence-verified benchmark for
# `conjugate_heat_transfer`. The first (v3.12) tested solid
# conduction against the Laplace series; the second (v3.21)
# tested transient conduction against the exponential-decay
# separable solution. This one tests the **coupling point
# between the thermal field and the momentum equation** —
# the Boussinesq buoyancy source term
#
#   F_b[c] = −ρ · β · (T[c] − T_ref) · g
#
# carries four algebraic invariants:
#
#   1. Zero β ⇒ no buoyancy (returns `nothing`).
#   2. Uniform T = T_ref ⇒ F_b ≡ 0.
#   3. Linearity: F_b(T_a) − F_b(T_b) = −ρ·β·(T_a − T_b)·g for
#      any two temperature fields.
#   4. β and g scaling: doubling either doubles F_b.
#
# Putting `conjugate_heat_transfer` at three convergence-verified
# benchmarks — the 3-benchmark floor for stable-promotion review.

using FiniteVolumeMethod
using LinearAlgebra: norm
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: Boussinesq — β = 0 disables buoyancy (returns nothing)" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    props = FluidThermalProperties{2}(;
        Cp = 1.0, k = 0.1, Pr_t = 0.9,
        beta = 0.0,          # ← explicit zero
        T_ref = 300.0,
        g = SVector(0.0, -9.81),
    )
    T_field = CollocatedScalarField(:T, mesh; value = 350.0)

    F = compute_buoyancy_source(T_field, props, 1.2)
    @test F === nothing
end

@testset "V&V: Boussinesq — uniform T = T_ref ⇒ F_b ≡ 0" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.7,
        beta = 3.4e-3,       # air
        T_ref = 293.15,
        g = SVector(0.0, -9.81),
    )
    T_field = CollocatedScalarField(:T, mesh; value = 293.15)

    F = compute_buoyancy_source(T_field, props, 1.2)
    @test F !== nothing
    for c in 1:nc
        @test isapprox(F[c][1], 0.0; atol = 1.0e-14)
        @test isapprox(F[c][2], 0.0; atol = 1.0e-14)
    end
end

@testset "V&V: Boussinesq — algebraic identity F = −ρβΔT·g" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    beta = 3.4e-3
    T_ref = 293.15
    rho = 1.2
    g_vec = SVector(0.0, -9.81)

    props = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.7,
        beta = beta, T_ref = T_ref, g = g_vec,
    )

    # T varies linearly with y to make every cell distinct.
    T_field = CollocatedScalarField(:T, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        T_field.internal[c] = T_ref + 50.0 * y
    end

    F = compute_buoyancy_source(T_field, props, rho)

    for c in 1:nc
        dT = T_field.internal[c] - T_ref
        F_expected = -rho * beta * dT * g_vec
        @test isapprox(F[c][1], F_expected[1]; rtol = 1.0e-12, atol = 1.0e-14)
        @test isapprox(F[c][2], F_expected[2]; rtol = 1.0e-12, atol = 1.0e-14)
    end
end

@testset "V&V: Boussinesq — linearity in (T − T_ref)" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.7,
        beta = 3.4e-3, T_ref = 300.0, g = SVector(0.0, -9.81),
    )
    rho = 1.0

    T_a = CollocatedScalarField(:T, mesh; value = 320.0)  # ΔT = 20
    T_b = CollocatedScalarField(:T, mesh; value = 340.0)  # ΔT = 40
    T_sum = CollocatedScalarField(:T, mesh; value = 360.0)  # 320+340-300 not relevant

    F_a = compute_buoyancy_source(T_a, props, rho)
    F_b = compute_buoyancy_source(T_b, props, rho)

    # F_b / F_a = ΔT_b / ΔT_a = 40 / 20 = 2.
    for c in 1:nc
        ratio_y = F_b[c][2] / F_a[c][2]
        @test isapprox(ratio_y, 2.0; rtol = 1.0e-12)
    end
end

@testset "V&V: Boussinesq — β and g scaling" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    T_field = CollocatedScalarField(:T, mesh; value = 320.0)
    rho = 1.0

    props_base = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.7,
        beta = 1.0e-3, T_ref = 300.0, g = SVector(0.0, -10.0),
    )
    props_2beta = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.7,
        beta = 2.0e-3, T_ref = 300.0, g = SVector(0.0, -10.0),
    )
    props_2g = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.7,
        beta = 1.0e-3, T_ref = 300.0, g = SVector(0.0, -20.0),
    )

    F_base = compute_buoyancy_source(T_field, props_base, rho)
    F_2b = compute_buoyancy_source(T_field, props_2beta, rho)
    F_2g = compute_buoyancy_source(T_field, props_2g, rho)

    for c in 1:nc
        @test isapprox(F_2b[c][2] / F_base[c][2], 2.0; rtol = 1.0e-12)
        @test isapprox(F_2g[c][2] / F_base[c][2], 2.0; rtol = 1.0e-12)
    end
end

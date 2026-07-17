# test/v_and_v_strain_rate.jl — Strain-rate magnitude primitive V&V (v3.53)
#
# Fourth convergence-verified benchmark for `turbulence_les`,
# joining Smagorinsky (v3.19), WALE (v3.28), and filter width +
# DynamicSmagorinsky (v3.39). Covers the `compute_strain_rate`
# primitive — the foundational kernel consumed by every LES and
# RANS production term:
#
#   |S| = √(2 S_ij S_ij)
#   S_ij = (1/2)(∂u_i/∂x_j + ∂u_j/∂x_i)
#
# Invariants:
#
#   1. Zero velocity ⇒ |S| = 0.
#   2. Uniform velocity ⇒ |S| = 0 (translation invariance).
#   3. Rigid rotation U = (−Ω·y, Ω·x) ⇒ |S| = 0
#      (rotations are strain-free).
#   4. Simple shear U = (A·y, 0) ⇒ |S| = |A|.
#   5. Biaxial stretching U = (α·x, −α·y) ⇒ |S| = 2·|α|.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_strain_rate
using StaticArrays
using Test

include("TestHelpers.jl")

function interior_cells(mesh, margin = 0.2)
    mask = falses(length(mesh.cell_volumes))
    for c in 1:length(mesh.cell_volumes)
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if margin < x < 1.0 - margin && margin < y < 1.0 - margin
            mask[c] = true
        end
    end
    return mask
end

@testset "V&V: Strain rate — zero velocity ⇒ |S| = 0" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))

    S_mag = compute_strain_rate(U, mesh)
    @test all(isapprox.(S_mag, 0.0; atol = 1.0e-14))
end

@testset "V&V: Strain rate — uniform velocity ⇒ |S| = 0" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh; value = SVector(3.7, -1.2))

    S_mag = compute_strain_rate(U, mesh)
    @test all(isapprox.(S_mag, 0.0; atol = 1.0e-12))
end

@testset "V&V: Strain rate — rigid rotation ⇒ |S| = 0" begin
    # Pure rotation has only an anti-symmetric velocity gradient,
    # so the symmetric part (strain rate) is identically zero.
    Omega = 2.5
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c] - 0.5
        y = mesh.cell_centers[2, c] - 0.5
        U.internal[c] = SVector(-Omega * y, Omega * x)
    end

    S_mag = compute_strain_rate(U, mesh)
    mask = interior_cells(mesh)
    for c in 1:nc
        if mask[c]
            @test isapprox(S_mag[c], 0.0; atol = 1.0e-10)
        end
    end
end

@testset "V&V: Strain rate — simple shear ⇒ |S| = |A|" begin
    # U = (A·y, 0): S_12 = S_21 = A/2, others zero.
    # 2 S_ij S_ij = 2·2·(A/2)² = A², so |S| = A.
    A = 3.0
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(A * y, 0.0)
    end

    S_mag = compute_strain_rate(U, mesh)
    mask = interior_cells(mesh)
    for c in 1:nc
        if mask[c]
            @test isapprox(S_mag[c], A; rtol = 1.0e-10)
        end
    end
end

@testset "V&V: Strain rate — biaxial stretching ⇒ |S| = 2·|α|" begin
    # U = (α·x, -α·y), ∂u/∂x = α, ∂v/∂y = -α, off-diag zero.
    # S_xx = α, S_yy = -α. 2 S_ij S_ij = 2(α² + α²) = 4α²,
    # |S| = 2|α|.
    alpha = 1.5
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(alpha * x, -alpha * y)
    end

    S_mag = compute_strain_rate(U, mesh)
    mask = interior_cells(mesh)
    for c in 1:nc
        if mask[c]
            @test isapprox(S_mag[c], 2 * abs(alpha); rtol = 1.0e-10)
        end
    end
end

@testset "V&V: Strain rate — A scaling (|S| linear in velocity magnitude)" begin
    # Doubling A doubles |S| for any linear velocity field.
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U_1 = CollocatedVectorField(:U, mesh)
    U_2 = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        U_1.internal[c] = SVector(y, 0.0)
        U_2.internal[c] = SVector(2 * y, 0.0)
    end

    S_1 = compute_strain_rate(U_1, mesh)
    S_2 = compute_strain_rate(U_2, mesh)
    mask = interior_cells(mesh)
    for c in 1:nc
        if mask[c]
            @test isapprox(S_2[c] / S_1[c], 2.0; rtol = 1.0e-10)
        end
    end
end

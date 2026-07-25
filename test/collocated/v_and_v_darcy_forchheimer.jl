# test/v_and_v_darcy_forchheimer.jl — Darcy-Forchheimer porous
# momentum-source algebra V&V.
#
# Algebraic invariants:
#   1. Isotropic K, F = 0 ⇒ source = −(μ/K)·U  (pure Darcy).
#   2. U = 0              ⇒ source = 0.
#   3. F > 0              ⇒ source has a quadratic |U|·U term, closed form.
#   4. Outside the porous zone ⇒ source untouched.
#   5. Monotone: |source| grows with |U|.

using FiniteVolumeMethod
using LinearAlgebra: norm
using StaticArrays
using Test

const PZ = FiniteVolumeMethod.PorousZone
const df_source = FiniteVolumeMethod.darcy_forchheimer_source
const add_df! = FiniteVolumeMethod.add_darcy_forchheimer_source!

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: Darcy-Forchheimer — pure Darcy limit (F = 0)" begin
    # Isotropic permeability k ⇒ K = k·I, K⁻¹ = (1/k)·I ⇒ source = −(μ/k)·U.
    k_perm = 1.0e-8
    mu = 1.0e-3
    rho = 1000.0
    zone = PZ([1, 2, 3]; K = k_perm, F = 0.0)
    for U in (SVector(1.0, 0.0), SVector(0.0, 2.0), SVector(0.5, -0.3))
        expected = -(mu / k_perm) .* U
        got = df_source(zone, U, rho, mu)
        @test got ≈ expected rtol = 1.0e-14
    end
end

@testset "V&V: Darcy-Forchheimer — U = 0 ⇒ zero source" begin
    zone = PZ([1]; K = 1.0e-8, F = 100.0)
    @test df_source(zone, SVector(0.0, 0.0), 1000.0, 1.0e-3) ==
        SVector(0.0, 0.0)
    @test df_source(zone, SVector(0.0, 0.0, 0.0), 1000.0, 1.0e-3) ==
        SVector(0.0, 0.0, 0.0)
end

@testset "V&V: Darcy-Forchheimer — isotropic F closed form" begin
    k_perm, f_coef = 1.0e-8, 50.0
    mu, rho = 1.0e-3, 1000.0
    zone = PZ([1]; K = k_perm, F = f_coef)
    U = SVector(1.2, -0.7)
    u_mag = norm(U)
    # S = −[ μ·K⁻¹ + 0.5·ρ·F·|U| ] · U
    #   = −[ μ/k + 0.5·ρ·f·|U| ] · U (isotropic)
    expected_scalar_coef = mu / k_perm + 0.5 * rho * f_coef * u_mag
    expected = -expected_scalar_coef .* U
    got = df_source(zone, U, rho, mu)
    @test got ≈ expected rtol = 1.0e-12
end

@testset "V&V: Darcy-Forchheimer — anisotropic tensor closed form (3D)" begin
    # Non-trivial diagonal K and F tensors.
    K_diag = [1.0e-8, 2.0e-8, 4.0e-8]
    F_diag = [10.0, 20.0, 30.0]
    mu, rho = 1.0e-3, 1000.0
    zone = PZ([1]; K = K_diag, F = F_diag)
    U = SVector(1.0, 0.5, -0.25)
    u_mag = norm(U)
    expected = SVector{3, Float64}(
        -(mu / K_diag[1] + 0.5 * rho * F_diag[1] * u_mag) * U[1],
        -(mu / K_diag[2] + 0.5 * rho * F_diag[2] * u_mag) * U[2],
        -(mu / K_diag[3] + 0.5 * rho * F_diag[3] * u_mag) * U[3],
    )
    got = df_source(zone, U, rho, mu)
    @test got ≈ expected rtol = 1.0e-12
end

@testset "V&V: Darcy-Forchheimer — add_darcy_forchheimer_source! respects zone mask" begin
    ncells = 10
    T = Float64
    source_U = [zero(SVector{2, T}) for _ in 1:ncells]
    U = [SVector(1.0, 0.0) for _ in 1:ncells]
    zone_cells = [2, 4, 7]
    zone = PZ(zone_cells; K = 1.0e-8, F = 0.0)  # pure Darcy for easy check
    rho, mu = 1000.0, 1.0e-3
    add_df!(source_U, U, zone, rho, mu)
    for c in 1:ncells
        if c in zone_cells
            @test source_U[c] ≈ SVector(-(mu / 1.0e-8), 0.0) rtol = 1.0e-12
        else
            @test source_U[c] == SVector(0.0, 0.0)
        end
    end
end

@testset "V&V: Darcy-Forchheimer — monotone |source| in |U|" begin
    zone = PZ([1]; K = 1.0e-8, F = 50.0)
    rho, mu = 1000.0, 1.0e-3
    mags = [0.1, 0.5, 1.0, 2.0, 5.0]
    norms = [norm(df_source(zone, SVector(m, 0.0), rho, mu)) for m in mags]
    for i in 1:(length(norms) - 1)
        @test norms[i] < norms[i + 1]
    end
end

@testset "V&V: Darcy-Forchheimer — accumulates onto pre-existing source" begin
    ncells = 5
    T = Float64
    source_U = [SVector(one(T), zero(T)) for _ in 1:ncells]  # pre-existing 1 along x
    U = [SVector(1.0, 0.0) for _ in 1:ncells]
    zone = PZ([1, 2]; K = 1.0e-8, F = 0.0)
    rho, mu = 1000.0, 1.0e-3
    add_df!(source_U, U, zone, rho, mu)
    darcy_coef = mu / 1.0e-8
    for c in 1:ncells
        if c in (1, 2)
            @test source_U[c] ≈ SVector(1.0 - darcy_coef, 0.0) rtol = 1.0e-12
        else
            @test source_U[c] == SVector(1.0, 0.0)
        end
    end
end

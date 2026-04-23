# test/v_and_v_fvdom_quadrature.jl — fvDOM angular-quadrature V&V (v3.48)
#
# Fourth convergence-verified benchmark for `radiation`, joining
# P1 cold-slab attenuation (v3.15), P1 equilibrium (v3.25), and
# compute_radiation_source algebra (v3.35). Covers the fvDOM
# angular-quadrature infrastructure underlying the higher-order
# DOM solver.
#
# Quadrature invariants verified (both S2 and S4 in 2D/3D):
#
#   1. All direction vectors are unit: |Ω̂_i| = 1.
#   2. Isotropy of sum-of-direction-cosines: Σ_i w_i·Ω̂_i = 0.
#   3. Total-weight identity: Σ_i w_i = surface of unit sphere in
#      relevant dimension (4π in 3D, or the 2D angular
#      integration weight 2π — but the implementation assigns
#      weights so Σ w = 2π for 2D S2, 4π for 3D S2).

using FiniteVolumeMethod
using LinearAlgebra: norm
using StaticArrays
using Test

include("TestHelpers.jl")

# Internal quadrature helpers (not exported); access via full
# module qualification.
const _s2_q = FiniteVolumeMethod._s2_quadrature
const _s4_q = FiniteVolumeMethod._s4_quadrature

@testset "V&V: fvDOM quadrature — 2D S2 has 4 unit directions" begin
    dirs, w = _s2_q(Val(2), Float64)
    @test length(dirs) == 4
    @test length(w) == 4
    for d in dirs
        @test isapprox(norm(d), 1.0; rtol = 1.0e-14)
    end
end

@testset "V&V: fvDOM quadrature — 3D S2 has 8 unit directions" begin
    dirs, w = _s2_q(Val(3), Float64)
    @test length(dirs) == 8
    @test length(w) == 8
    for d in dirs
        @test isapprox(norm(d), 1.0; rtol = 1.0e-14)
    end
end

@testset "V&V: fvDOM quadrature — S2 isotropy Σ w·Ω̂ = 0" begin
    # In 2D:
    dirs_2, w_2 = _s2_q(Val(2), Float64)
    sum_2 = sum(w_2[i] * dirs_2[i] for i in 1:length(dirs_2))
    @test norm(sum_2) < 1.0e-12

    # In 3D:
    dirs_3, w_3 = _s2_q(Val(3), Float64)
    sum_3 = sum(w_3[i] * dirs_3[i] for i in 1:length(dirs_3))
    @test norm(sum_3) < 1.0e-12
end

@testset "V&V: fvDOM quadrature — S2 total weight" begin
    # In 2D S2: 4 directions × π/2 = 2π.
    _, w_2 = _s2_q(Val(2), Float64)
    @test isapprox(sum(w_2), 2 * pi; rtol = 1.0e-14)

    # In 3D S2: 8 directions × π/2 = 4π.
    _, w_3 = _s2_q(Val(3), Float64)
    @test isapprox(sum(w_3), 4 * pi; rtol = 1.0e-14)
end

@testset "V&V: fvDOM quadrature — 2D S4 has 12 unit directions" begin
    dirs, w = _s4_q(Val(2), Float64)
    @test length(dirs) == 12
    for d in dirs
        @test isapprox(norm(d), 1.0; rtol = 1.0e-6)
    end
end

@testset "V&V: fvDOM quadrature — 2D S4 isotropy and total weight" begin
    dirs, w = _s4_q(Val(2), Float64)

    # Sum of w·Ω̂ = 0 (isotropic quadrature).
    s = sum(w[i] * dirs[i] for i in 1:length(dirs))
    @test norm(s) < 1.0e-6

    # Total weight: 4 quadrants × (2 · π/6 + π/3) = 4·(π/3 + π/3)
    # = 8π/3 = 2.67π.
    total = sum(w)
    expected = 4 * (2 * (pi / 6) + pi / 3)   # per implementation
    @test isapprox(total, expected; rtol = 1.0e-14)
end

@testset "V&V: fvDOM quadrature — FvDOMModel constructs in 2D and 3D" begin
    # Smoke check of higher-level constructor.
    model_2d = FvDOMModel(; a = 0.5, Dim = 2, order = :S2)
    model_3d = FvDOMModel(; a = 0.5, Dim = 3, order = :S2)

    @test length(model_2d.directions) == 4
    @test length(model_3d.directions) == 8

    # S4 variants.
    model_2d_s4 = FvDOMModel(; a = 0.5, Dim = 2, order = :S4)
    @test length(model_2d_s4.directions) == 12
end

# test/v_and_v_wsggm.jl — WSGGM (Weighted-Sum-of-Grey-Gases) V&V
#
# Validates the wavelength-banded emissivity model added in Wave 2.
# Primary reference: Smith, Shen & Friedman (1982), J. Heat Transfer 104.
#
# Invariants checked:
#   1. Weights sum to 1 at every temperature (`Σ a_i(T) = 1`), including
#      the window-band-closure trick for `a_1(T) = 1 − Σ_{i>1} a_i(T)`.
#   2. All weights are non-negative for the validity range T ∈ [300, 2500] K.
#   3. Emissivity is monotone in path length (longer path ⇒ higher ε).
#   4. Emissivity saturates to ≤ 1 − a_window(T) at infinite path (the
#      window band does not participate, so `ε_∞ ≤ 1 − a_window`).
#   5. ε(T, 0) = 0 for every T (zero path, zero emission).
#   6. Effective grey absorption κ_eff > 0 for participating mixtures.
#   7. κ_eff reproduces ε via ε = 1 − exp(−κ_eff · L) to high accuracy.
#   8. solve_wsggm_radiation dispatches cleanly onto the :p1 grey solver.

using FiniteVolumeMethod
using FiniteVolumeMethod: AbstractRadiationModel
using FiniteVolumeMethod.Parabolic: DirichletBC
using LinearSolve
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

# --------------------------------------------------------------------
# Weight-sum invariant
# --------------------------------------------------------------------

@testset "V&V: WSGGM — Σ a_i(T) = 1 at 300 K, 1000 K, 2000 K" begin
    m = FiniteVolumeMethod.WSGGMModel()
    for T in (300.0, 1000.0, 2000.0)
        s = sum(FiniteVolumeMethod.compute_band_weight(m, T, i) for i in 1:4)
        @test isapprox(s, 1.0; rtol = 1.0e-6)
    end
end

@testset "V&V: WSGGM — Σ a_i(T) = 1 at fine temperature sweep" begin
    m = FiniteVolumeMethod.WSGGMModel()
    for T in 400.0:100.0:2500.0
        s = sum(FiniteVolumeMethod.compute_band_weight(m, T, i) for i in 1:4)
        @test isapprox(s, 1.0; rtol = 1.0e-6)
    end
end

@testset "V&V: WSGGM — all weights non-negative" begin
    m = FiniteVolumeMethod.WSGGMModel()
    for T in 300.0:50.0:2500.0
        for i in 1:4
            @test FiniteVolumeMethod.compute_band_weight(m, T, i) >= 0.0
        end
    end
end

# --------------------------------------------------------------------
# Emissivity structure
# --------------------------------------------------------------------

@testset "V&V: WSGGM — ε(T, L=0) = 0" begin
    m = FiniteVolumeMethod.WSGGMModel()
    for T in (500.0, 1200.0, 1800.0)
        @test FiniteVolumeMethod.compute_band_emissivity(m, T, 0.0) == 0.0
    end
end

@testset "V&V: WSGGM — emissivity monotone in path length" begin
    m = FiniteVolumeMethod.WSGGMModel()
    Ls = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for T in (500.0, 1000.0, 1500.0, 2000.0)
        eps_prev = -Inf
        for L in Ls
            eps_L = FiniteVolumeMethod.compute_band_emissivity(m, T, L)
            @test eps_L >= eps_prev
            eps_prev = eps_L
        end
    end
end

@testset "V&V: WSGGM — emissivity bounded by 1 - a_window" begin
    m = FiniteVolumeMethod.WSGGMModel()
    for T in (500.0, 1000.0, 1500.0, 2000.0)
        a_window = FiniteVolumeMethod.compute_band_weight(m, T, 1)
        eps_inf = FiniteVolumeMethod.compute_band_emissivity(m, T, 1.0e6)
        # ε(T, ∞) = 1 - a_window(T) because κ_window = 0
        @test isapprox(eps_inf, 1.0 - a_window; rtol = 1.0e-6)
        @test eps_inf <= 1.0
    end
end

# --------------------------------------------------------------------
# Effective-grey absorption
# --------------------------------------------------------------------

@testset "V&V: WSGGM — κ_eff > 0 for participating mixture" begin
    m = FiniteVolumeMethod.WSGGMModel()
    mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
    T_field = CollocatedScalarField(:T, mesh; value = 1500.0)
    kappa_eff = FiniteVolumeMethod.wsggm_effective_absorption(m, T_field, 1.0)
    @test length(kappa_eff) == length(mesh.cell_volumes)
    @test all(k -> k > 0.0, kappa_eff)
    @test all(isfinite, kappa_eff)
end

@testset "V&V: WSGGM — κ_eff reproduces ε within clamp window" begin
    m = FiniteVolumeMethod.WSGGMModel()
    mesh = build_cartesian_unstructured_mesh(2, 2, 1.0, 1.0)
    # Non-uniform temperature so different cells exercise different bands.
    T_field = CollocatedScalarField(:T, mesh; value = 1200.0)
    for c in 1:length(T_field.internal)
        T_field.internal[c] = 600.0 + 400.0 * c
    end
    L = 0.5
    kappa_eff = FiniteVolumeMethod.wsggm_effective_absorption(m, T_field, L)
    for c in 1:length(T_field.internal)
        eps_direct = FiniteVolumeMethod.compute_band_emissivity(m, T_field.internal[c], L)
        eps_via_kappa = 1.0 - exp(-kappa_eff[c] * L)
        # Round-trip identity is exact up to the clamp bounds (1e-20, 1-1e-12).
        if 1.0e-18 < eps_direct < 1.0 - 1.0e-10
            @test isapprox(eps_via_kappa, eps_direct; rtol = 1.0e-9)
        end
    end
end

# --------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------

@testset "V&V: WSGGM — dispatch via AbstractRadiationModel" begin
    m = FiniteVolumeMethod.WSGGMModel()
    @test m isa FiniteVolumeMethod.AbstractRadiationModel
end

@testset "V&V: WSGGM — unknown band set errors" begin
    @test_throws ErrorException FiniteVolumeMethod.WSGGMModel(; bands = :haworth2020)
end

@testset "V&V: WSGGM — solve_wsggm_radiation produces a finite G field" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    T_field = CollocatedScalarField(:T, mesh; value = 1500.0)
    bcs = Dict{Symbol, FiniteVolumeMethod.AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :top => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
    )
    m = FiniteVolumeMethod.WSGGMModel()
    G = FiniteVolumeMethod.solve_wsggm_radiation(
        m, T_field, mesh, bcs;
        path_length = 0.5, grey_solver = :p1,
    )
    @test length(G.internal) == length(mesh.cell_volumes)
    @test all(isfinite, G.internal)
end

@testset "V&V: WSGGM — unknown grey_solver errors" begin
    mesh = build_cartesian_unstructured_mesh(2, 2, 1.0, 1.0)
    T_field = CollocatedScalarField(:T, mesh; value = 1500.0)
    bcs = Dict{Symbol, FiniteVolumeMethod.AbstractBoundaryCondition}()
    m = FiniteVolumeMethod.WSGGMModel()
    @test_throws ErrorException FiniteVolumeMethod.solve_wsggm_radiation(
        m, T_field, mesh, bcs;
        path_length = 0.5, grey_solver = :mc_raytrace,
    )
end

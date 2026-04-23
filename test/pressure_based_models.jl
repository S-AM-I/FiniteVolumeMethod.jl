# test/pressure_based_models.jl — Stage 3 thermo + rheology + non-orthogonal
# correction contract tests.

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

@testset "Stage 3a: AbstractThermoModel hierarchy" begin
    # Incompressible — density independent of p, T.
    inc = IncompressibleThermo(; rho = 1.225, mu = 1.789e-5, cp = 1005.0, beta = 0.0)
    @test inc isa AbstractThermoModel
    @test density_at(inc, 1.0e5, 300.0) == 1.225
    @test density_at(inc, 2.0e5, 500.0) == 1.225  # no dependence
    @test viscosity_at(inc, 500.0) == 1.789e-5
    @test cp_at(inc, 500.0) == 1005.0
    @test !is_compressible(inc)

    # Ideal gas — ρ = p/(R T).
    gas = IdealGas(; gamma = 1.4, R = 287.05, mu = 1.8e-5)
    @test gas isa AbstractThermoModel
    @test density_at(gas, 1.0e5, 300.0) ≈ 1.0e5 / (287.05 * 300.0)
    @test density_at(gas, 2.0e5, 300.0) ≈ 2 * density_at(gas, 1.0e5, 300.0)  # linear in p
    @test is_compressible(gas)

    # Boussinesq — ρ = ρ₀ (1 - β (T - T₀)).
    bou = BoussinesqThermo(; rho0 = 1.0, T0 = 300.0, beta = 3.33e-3)
    @test bou isa AbstractThermoModel
    @test density_at(bou, 0.0, 300.0) ≈ 1.0   # T = T₀
    @test density_at(bou, 0.0, 400.0) ≈ 1.0 * (1 - 3.33e-3 * 100.0)
    @test !is_compressible(bou)

    # Sutherland gas — μ(T) strictly increasing.
    sg = SutherlandGas()
    @test sg isa AbstractThermoModel
    @test viscosity_at(sg, 400.0) > viscosity_at(sg, 300.0)
    @test density_at(sg, 1.0e5, 300.0) ≈ 1.0e5 / (287.05 * 300.0)
    @test is_compressible(sg)
end

@testset "Stage 3b: AbstractRheology hierarchy" begin
    # Newtonian — constant, temperature-invariant.
    newt = NewtonianRheology(; mu = 1.0e-3)
    @test newt isa AbstractRheology
    @test viscosity_at(newt, 0.1, 300.0) == 1.0e-3
    @test viscosity_at(newt, 1.0e6, 300.0) == 1.0e-3

    # Power law — shear-thinning n < 1.
    pl = PowerLawRheology(; K = 1.0e-3, n = 0.5)
    @test pl isa AbstractRheology
    mu_1 = viscosity_at(pl, 1.0, 300.0)
    mu_10 = viscosity_at(pl, 10.0, 300.0)
    @test mu_10 < mu_1   # shear-thinning
    @test mu_1 ≈ 1.0e-3 * 1.0^(0.5 - 1)
    @test mu_10 ≈ 1.0e-3 * 10.0^(0.5 - 1)

    # Bird-Carreau — μ(0) → μ₀, μ(∞) → μ_inf (approach is slow with n=0.5).
    bc = BirdCarreauRheology(; mu_0 = 1.0, mu_inf = 1.0e-3, lambda = 1.0, n = 0.5)
    mu_zero = viscosity_at(bc, 1.0e-12, 300.0)
    mu_small = viscosity_at(bc, 0.1, 300.0)
    mu_huge = viscosity_at(bc, 1.0e12, 300.0)
    @test mu_zero ≈ 1.0
    @test mu_small < mu_zero       # shear-thinning below zero-shear plateau
    @test mu_huge ≈ 1.0e-3 atol = 1.0e-4
    @test mu_huge < mu_small       # monotone shear-thinning

    # Herschel-Bulkley — yield stress dominates at low γ̇.
    hb = HerschelBulkleyRheology(; tau_y = 0.5, K = 1.0e-3, n = 0.8)
    mu_low = viscosity_at(hb, 1.0e-4, 300.0)
    @test mu_low > 1000.0   # near-rigid at vanishing strain rate

    # Casson — finite limit from √μ = √μ∞ + √(τ_y / γ̇).
    cas = CassonRheology(; tau_y = 0.05, mu_inf = 4.0e-3)
    mu_cas = viscosity_at(cas, 10.0, 300.0)
    @test mu_cas > 4.0e-3   # greater than μ_inf due to yield contribution
end

@testset "Stage 3c: Over-relaxed non-orthogonal correction" begin
    mesh = build_cartesian_unstructured_mesh(8, 6, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(0.0),
        :right => ParabolicDirichlet(0.0),
        :bottom => ParabolicDirichlet(0.0),
        :top => ParabolicDirichlet(1.0),
    )

    # On a Cartesian mesh, S_f · d̂ = |S_f|, so all three modes produce the
    # same implicit coefficient. Sanity-check by assembling with each and
    # asserting equality.
    eq_min = CollocatedEquation(mesh)
    assemble_laplacian!(eq_min, 1.0, mesh, bcs; correction_mode = NON_ORTHO_MINIMUM)

    eq_orth = CollocatedEquation(mesh)
    assemble_laplacian!(eq_orth, 1.0, mesh, bcs; correction_mode = NON_ORTHO_ORTHOGONAL)

    eq_over = CollocatedEquation(mesh)
    assemble_laplacian!(eq_over, 1.0, mesh, bcs; correction_mode = NON_ORTHO_OVER_RELAXED)

    @test eq_min.A.nzval ≈ eq_orth.A.nzval atol = 1.0e-12
    @test eq_orth.A.nzval ≈ eq_over.A.nzval atol = 1.0e-12
    @test eq_min.b ≈ eq_orth.b
    @test eq_orth.b ≈ eq_over.b

    # Default mode is over-relaxed (see function signature).
    eq_default = CollocatedEquation(mesh)
    assemble_laplacian!(eq_default, 1.0, mesh, bcs)
    @test eq_default.A.nzval ≈ eq_over.A.nzval

    # On a NON-orthogonal stress mesh the three modes should differ: the
    # over-relaxed implicit coefficient scales by 1/cosθ. We mimic this
    # by skewing one row of cell centers — that changes d̂ without
    # changing the face normals, so S·d̂ < |S|. Use a small perturbation.
    nc = length(mesh.cell_volumes)
    skewed = deepcopy(mesh)
    for c in 1:nc
        skewed.cell_centers[1, c] += 0.03 * sin(2π * skewed.cell_centers[2, c])
    end
    eq_skew_min = CollocatedEquation(skewed)
    assemble_laplacian!(eq_skew_min, 1.0, skewed, bcs; correction_mode = NON_ORTHO_MINIMUM)
    eq_skew_over = CollocatedEquation(skewed)
    assemble_laplacian!(eq_skew_over, 1.0, skewed, bcs; correction_mode = NON_ORTHO_OVER_RELAXED)
    # Over-relaxed should increase |diagonal| relative to minimum (|E|
    # scales by 1/cosθ ≥ 1, and the diagonal is the sum of per-face
    # E-magnitude contributions).
    diag_min = sum(eq_skew_min.A[c, c] for c in 1:nc)
    diag_over = sum(eq_skew_over.A[c, c] for c in 1:nc)
    @test diag_over > diag_min
end

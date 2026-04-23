# test/v_and_v_edm.jl — EDM reaction-rate algebra V&V (v3.27)
#
# Second analytical benchmark for `combustion`. The first benchmark
# (v3.17, species AD) tested transport kinematics; this one tests
# the **turbulence-chemistry interaction source term** at the
# heart of the Magnussen-Hjertager Eddy Dissipation Model:
#
#   ω_fuel = -ρ · A · (ε/k) · min(Y_fuel, Y_ox/s)          (mixing-limited)
#   ω_fuel = max(ω_fuel_mix, ω_prod_mix)                   (product-limited
#                                                           bound if Y_prod > 0)
#   ω_ox    = s · ω_fuel
#   ω_prod  = -(1 + s) · ω_fuel
#
# These are algebraic identities: given prescribed Y, k, ε and
# stoichiometry s, the returned rates are determined to machine
# precision. Four invariants are verified.
#
# Evidence toward future `stable` promotion of `combustion`.

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

@testset "V&V: EDM — fuel-limited branch matches closed form" begin
    # Fuel-limited: Y_fuel / 1 < Y_ox / s ⇒ rate driven by Y_fuel.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    props = CombustionProperties(;
        species_names = (:fuel, :oxidizer, :product),
        molecular_weights = (16.0, 32.0, 44.0),
        stoich_ratio = 4.0,
        heat_of_combustion = 5.0e7,
    )
    # Fuel-limited: Y_fuel small compared to Y_ox/s.
    species = SpeciesState(mesh, props; fuel = 0.01, oxidizer = 0.25, product = 0.0)

    k = fill(1.0, nc)
    eps = fill(0.5, nc)   # mixing rate ε/k = 0.5 s⁻¹
    rho = 1.2

    edm = EddyDissipationModel(; A_edm = 4.0, B_edm = 0.5)
    omega = compute_edm_reaction_rates(edm, species, props, k, eps, rho, mesh)

    # Expected fuel consumption: -ρ·A·(ε/k)·min(Y_fuel, Y_ox/s)
    #                           = -1.2 · 4 · 0.5 · min(0.01, 0.0625)
    #                           = -2.4 · 0.01 = -0.024.
    omega_fuel_expected = -rho * 4.0 * 0.5 * 0.01

    for c in 1:nc
        @test isapprox(omega[1][c], omega_fuel_expected; rtol = 1.0e-12)
        # Stoichiometric rates.
        @test isapprox(omega[2][c], 4.0 * omega[1][c]; rtol = 1.0e-12)
        @test isapprox(omega[3][c], -5.0 * omega[1][c]; rtol = 1.0e-12)
    end
end

@testset "V&V: EDM — oxidizer-limited branch matches closed form" begin
    # Oxidizer-limited: Y_ox/s < Y_fuel ⇒ rate driven by Y_ox/s.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    props = CombustionProperties(; stoich_ratio = 4.0)
    species = SpeciesState(mesh, props; fuel = 0.5, oxidizer = 0.04, product = 0.0)

    k = fill(2.0, nc)
    eps = fill(1.0, nc)  # ε/k = 0.5
    rho = 1.0

    edm = EddyDissipationModel(; A_edm = 4.0)
    omega = compute_edm_reaction_rates(edm, species, props, k, eps, rho, mesh)

    # Expected: min(Y_fuel, Y_ox/s) = min(0.5, 0.04/4) = 0.01.
    omega_fuel_expected = -1.0 * 4.0 * 0.5 * 0.01

    for c in 1:nc
        @test isapprox(omega[1][c], omega_fuel_expected; rtol = 1.0e-12)
    end
end

@testset "V&V: EDM — stoichiometric mass-balance invariant" begin
    # Σ ω_i ≡ 0 (mass conservation under a complete-combustion
    # one-step mechanism with stoichiometric coefficients).
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    props = CombustionProperties(; stoich_ratio = 4.0)
    species = SpeciesState(mesh, props; fuel = 0.2, oxidizer = 0.3, product = 0.1)

    k = fill(1.5, nc)
    eps = fill(0.8, nc)
    rho = 1.2

    edm = EddyDissipationModel(; A_edm = 4.0, B_edm = 0.5)
    omega = compute_edm_reaction_rates(edm, species, props, k, eps, rho, mesh)

    # Mass conservation: ω_fuel + ω_ox + ω_prod ≡ 0 for every cell.
    # ω_fuel + s·ω_fuel - (1+s)·ω_fuel = 0 algebraically.
    for c in 1:nc
        total = omega[1][c] + omega[2][c] + omega[3][c]
        @test abs(total) < 1.0e-14
    end
end

@testset "V&V: EDM — mixing rate scaling (ε/k proportionality)" begin
    # ω_fuel ∝ ε/k at fixed A, Y, s, ρ. Verify by doubling ε/k.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    props = CombustionProperties(; stoich_ratio = 4.0)
    species = SpeciesState(mesh, props; fuel = 0.05, oxidizer = 0.3, product = 0.0)

    rho = 1.0
    edm = EddyDissipationModel(; A_edm = 4.0)

    # ε/k = 0.5 and ε/k = 1.0; ω should scale by 2×.
    k = fill(1.0, nc)
    eps_a = fill(0.5, nc)
    omega_a = compute_edm_reaction_rates(edm, species, props, k, eps_a, rho, mesh)

    eps_b = fill(1.0, nc)
    omega_b = compute_edm_reaction_rates(edm, species, props, k, eps_b, rho, mesh)

    for c in 1:nc
        @test isapprox(omega_b[1][c] / omega_a[1][c], 2.0; rtol = 1.0e-12)
    end
end

@testset "V&V: EDM — heat release ω_fuel · ΔH" begin
    # S_h = -ω_fuel · ΔH. With ω_fuel < 0 and ΔH > 0, S_h > 0
    # (exothermic release).
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    dH = 5.0e7  # 50 MJ/kg
    props = CombustionProperties(; stoich_ratio = 4.0, heat_of_combustion = dH)
    species = SpeciesState(mesh, props; fuel = 0.02, oxidizer = 0.2, product = 0.0)

    k = fill(1.0, nc)
    eps = fill(0.5, nc)
    rho = 1.0

    edm = EddyDissipationModel(; A_edm = 4.0)
    omega = compute_edm_reaction_rates(edm, species, props, k, eps, rho, mesh)
    S_h = compute_heat_release(omega, props)

    # Expected: -ω_fuel · ΔH > 0.
    for c in 1:nc
        @test S_h[c] > 0.0
        @test isapprox(S_h[c], -omega[1][c] * dH; rtol = 1.0e-12)
    end
end

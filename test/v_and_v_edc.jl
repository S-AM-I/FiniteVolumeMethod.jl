# test/v_and_v_edc.jl — EDC (Eddy Dissipation Concept) invariants V&V (v3.93)

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

const _edc_rates = FiniteVolumeMethod.compute_edc_reaction_rates
const _FALLBACK = FiniteVolumeMethod._EDC_FALLBACK_MIXING_RATE

@testset "V&V: EDC fallback constant" begin
    # When no turbulence fields are supplied, the model falls back to
    # a constant EDM-like mixing rate of 10 / s.
    @test _FALLBACK == 10.0
end

@testset "V&V: EDC EddyDissipationConcept default constants" begin
    # Magnussen (2005) default EDC constants.
    edc = EddyDissipationConcept()
    @test edc.C_gamma ≈ 2.1377 rtol = 1.0e-14
    @test edc.C_tau ≈ 0.4082 rtol = 1.0e-14
end

@testset "V&V: EDC fallback — stoichiometric mass balance Σ ω = 0" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    props = CombustionProperties(; stoich_ratio = 4.0)
    species = SpeciesState(mesh, props; fuel = 0.1, oxidizer = 0.23)
    edc = EddyDissipationConcept()
    rho = 1.2
    nu = 1.5e-5
    omega = _edc_rates(edc, species, props, nothing, nothing, rho, nu, mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        total = omega[1][c] + omega[2][c] + omega[3][c]
        @test abs(total) < 1.0e-10
    end
end

@testset "V&V: EDC fallback — ω_fuel closed form min(Y_f, Y_o/s)" begin
    # Fallback rate is ω_fuel = -ρ · 10 · min(Y_fuel, Y_ox/s).
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    props = CombustionProperties(; stoich_ratio = 4.0)
    rho = 1.2
    nu = 1.5e-5
    edc = EddyDissipationConcept()
    # Fuel-limited: Y_fuel = 0.05, Y_ox = 0.5 (0.5/4 = 0.125 > 0.05).
    species_f = SpeciesState(mesh, props; fuel = 0.05, oxidizer = 0.5)
    om_f = _edc_rates(edc, species_f, props, nothing, nothing, rho, nu, mesh)
    expected_f = -1.2 * 10.0 * 0.05
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        @test om_f[1][c] ≈ expected_f rtol = 1.0e-12
    end
    # Oxidizer-limited: Y_fuel = 0.5, Y_ox = 0.1 (0.1/4 = 0.025 < 0.5).
    species_o = SpeciesState(mesh, props; fuel = 0.5, oxidizer = 0.1)
    om_o = _edc_rates(edc, species_o, props, nothing, nothing, rho, nu, mesh)
    expected_o = -1.2 * 10.0 * 0.025
    for c in 1:nc
        @test om_o[1][c] ≈ expected_o rtol = 1.0e-12
    end
end

@testset "V&V: EDC fallback — species rates from stoichiometry" begin
    # ω_oxidizer = s·ω_fuel, ω_product = -(1+s)·ω_fuel.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    props = CombustionProperties(; stoich_ratio = 4.0)
    species = SpeciesState(mesh, props; fuel = 0.05, oxidizer = 0.5)
    edc = EddyDissipationConcept()
    omega = _edc_rates(edc, species, props, nothing, nothing, 1.2, 1.5e-5, mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        @test omega[2][c] ≈ 4.0 * omega[1][c] rtol = 1.0e-12
        @test omega[3][c] ≈ -5.0 * omega[1][c] rtol = 1.0e-12
    end
end

@testset "V&V: EDC fallback — density linearity of ω_fuel" begin
    # ω_fuel = -ρ · mixing_rate · min(Y_f, Y_o/s) ⇒ linear in ρ.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    props = CombustionProperties(; stoich_ratio = 4.0)
    species = SpeciesState(mesh, props; fuel = 0.05, oxidizer = 0.5)
    edc = EddyDissipationConcept()
    om1 = _edc_rates(edc, species, props, nothing, nothing, 1.0, 1.5e-5, mesh)
    om2 = _edc_rates(edc, species, props, nothing, nothing, 2.0, 1.5e-5, mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        @test om2[1][c] ≈ 2.0 * om1[1][c] rtol = 1.0e-12
    end
end

@testset "V&V: EDC fallback — zero fuel ⇒ zero rates" begin
    # Y_fuel = 0 ⇒ min(0, Y_ox/s) = 0 ⇒ ω_fuel = 0 ⇒ all ω_i = 0.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    props = CombustionProperties(; stoich_ratio = 4.0)
    species = SpeciesState(mesh, props; fuel = 0.0, oxidizer = 0.23)
    edc = EddyDissipationConcept()
    omega = _edc_rates(edc, species, props, nothing, nothing, 1.2, 1.5e-5, mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        @test omega[1][c] == 0.0
        @test omega[2][c] == 0.0
        @test omega[3][c] == 0.0
    end
end

@testset "V&V: EDC with turbulence — stoichiometric balance" begin
    # With k/ε turbulence fields the full EDC formula runs. Still must
    # satisfy ω_fuel + ω_ox + ω_prod ≡ 0 by construction.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)
    species = SpeciesState(mesh, props; fuel = 0.05, oxidizer = 0.5)
    edc = EddyDissipationConcept()
    k_field = fill(1.0, nc)
    eps_field = fill(10.0, nc)
    omega = _edc_rates(edc, species, props, k_field, eps_field, 1.2, 1.5e-5, mesh)
    for c in 1:nc
        total = omega[1][c] + omega[2][c] + omega[3][c]
        @test abs(total) < 1.0e-10
    end
    # Y_fuel > 0 ⇒ ω_fuel should be negative (consumption).
    for c in 1:nc
        @test omega[1][c] < 0.0
    end
end

@testset "V&V: EDC with turbulence — Y_fuel linearity at fixed k, ε" begin
    # ω_fuel = -ρ · γ*² / (τ*·(1-γ*³)) · Y_fuel ⇒ linear in Y_fuel at
    # fixed turbulence + density.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)
    edc = EddyDissipationConcept()
    k_field = fill(1.0, nc)
    eps_field = fill(10.0, nc)
    s1 = SpeciesState(mesh, props; fuel = 0.05, oxidizer = 0.5)
    s2 = SpeciesState(mesh, props; fuel = 0.1, oxidizer = 0.5)
    o1 = _edc_rates(edc, s1, props, k_field, eps_field, 1.2, 1.5e-5, mesh)
    o2 = _edc_rates(edc, s2, props, k_field, eps_field, 1.2, 1.5e-5, mesh)
    for c in 1:nc
        # Doubling Y_fuel should double |ω_fuel| (EDC turbulent branch
        # doesn't depend on Y_oxidizer).
        @test o2[1][c] ≈ 2.0 * o1[1][c] rtol = 1.0e-10
    end
end

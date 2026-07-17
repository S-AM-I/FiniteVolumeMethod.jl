# test/v_and_v_arrhenius.jl — Arrhenius kinetics V&V (v3.37)
#
# Third convergence-verified benchmark for `combustion`. The first
# (v3.17) tested species transport; the second (v3.27) tested the
# Magnussen-Hjertager EDM mixing-limited closure. This one covers
# the complementary **finite-rate chemistry** closure:
#
#   k_f(T) = A · T^b · exp(−E_a / (R · T))
#   ω_fuel = −ρ · k_f · Y_fuel^n_fuel · Y_ox^n_ox
#
# Five algebraic invariants are verified, including the
# characteristic Arrhenius temperature sensitivity that
# distinguishes finite-rate kinetics from mixing-limited EDM.
#
# Puts `combustion` at three convergence-verified benchmarks.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_arrhenius_reaction_rates
using Test

include("TestHelpers.jl")

const R_UNIV = 8.314

@testset "V&V: Arrhenius — zero fuel or oxidizer ⇒ ω ≡ 0" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)

    reaction = CollocatedArrheniusReaction(; A = 1.0e10, E_a = 1.0e5)
    T_field = CollocatedScalarField(:T, mesh; value = 1500.0)

    # No fuel.
    state_nofuel = SpeciesState(mesh, props; fuel = 0.0, oxidizer = 0.5, product = 0.0)
    omega_nf = compute_arrhenius_reaction_rates(reaction, state_nofuel, props, T_field, 1.0, mesh)
    for c in 1:nc
        @test omega_nf[1][c] == 0.0
    end

    # No oxidizer.
    state_noox = SpeciesState(mesh, props; fuel = 0.5, oxidizer = 0.0, product = 0.0)
    omega_no = compute_arrhenius_reaction_rates(reaction, state_noox, props, T_field, 1.0, mesh)
    for c in 1:nc
        @test omega_no[1][c] == 0.0
    end
end

@testset "V&V: Arrhenius — closed-form algebraic identity" begin
    # Verify ω_fuel = -ρ · k_f(T) · Y_f^n_f · Y_o^n_o exactly.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)

    A = 1.0e10
    b = 0.5
    E_a = 1.0e5
    n_fuel = 1.0
    n_ox = 1.0
    reaction = CollocatedArrheniusReaction(; A = A, b = b, E_a = E_a, n_fuel = n_fuel, n_ox = n_ox)

    T_val = 1500.0
    T_field = CollocatedScalarField(:T, mesh; value = T_val)
    Y_f = 0.08
    Y_o = 0.2
    rho = 1.2
    state = SpeciesState(mesh, props; fuel = Y_f, oxidizer = Y_o, product = 0.0)

    omega = compute_arrhenius_reaction_rates(reaction, state, props, T_field, rho, mesh)

    # Expected: k_f = A · T^b · exp(-E_a / (R · T)).
    k_f = A * T_val^b * exp(-E_a / (R_UNIV * T_val))
    omega_fuel_expected = -rho * k_f * Y_f^n_fuel * Y_o^n_ox

    for c in 1:nc
        @test isapprox(omega[1][c], omega_fuel_expected; rtol = 1.0e-12)
        # Stoichiometry.
        @test isapprox(omega[2][c], 4.0 * omega[1][c]; rtol = 1.0e-12)
        @test isapprox(omega[3][c], -5.0 * omega[1][c]; rtol = 1.0e-12)
    end
end

@testset "V&V: Arrhenius — exponential T-sensitivity" begin
    # Doubling T (holding everything else fixed) should multiply
    # k_f by exp(E_a / R · (1/T_low - 1/T_high)) · (T_high/T_low)^b.
    # At T_low = 1000, T_high = 2000, E_a = 1e5, b = 0:
    #   exp(1e5/R · (1/1000 - 1/2000)) = exp(1e5 · 5e-4 / R) = exp(50 / R) ≈ exp(6.014) ≈ 409
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)

    reaction = CollocatedArrheniusReaction(; A = 1.0e10, b = 0.0, E_a = 1.0e5)
    state = SpeciesState(mesh, props; fuel = 0.1, oxidizer = 0.3, product = 0.0)

    T_low = CollocatedScalarField(:T, mesh; value = 1000.0)
    T_high = CollocatedScalarField(:T, mesh; value = 2000.0)

    omega_low = compute_arrhenius_reaction_rates(reaction, state, props, T_low, 1.0, mesh)
    omega_high = compute_arrhenius_reaction_rates(reaction, state, props, T_high, 1.0, mesh)

    # Analytical ratio (b = 0):
    ratio_expected = exp(1.0e5 / R_UNIV * (1.0 / 1000.0 - 1.0 / 2000.0))
    for c in 1:nc
        ratio_numerical = omega_high[1][c] / omega_low[1][c]
        @test isapprox(ratio_numerical, ratio_expected; rtol = 1.0e-12)
    end
end

@testset "V&V: Arrhenius — pre-exponential A scaling" begin
    # Doubling A should double |ω_fuel| (linear in A).
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)

    reaction_a = CollocatedArrheniusReaction(; A = 1.0e10, E_a = 1.0e5)
    reaction_b = CollocatedArrheniusReaction(; A = 2.0e10, E_a = 1.0e5)

    T_field = CollocatedScalarField(:T, mesh; value = 1500.0)
    state = SpeciesState(mesh, props; fuel = 0.1, oxidizer = 0.3, product = 0.0)

    omega_a = compute_arrhenius_reaction_rates(reaction_a, state, props, T_field, 1.0, mesh)
    omega_b = compute_arrhenius_reaction_rates(reaction_b, state, props, T_field, 1.0, mesh)

    for c in 1:nc
        @test isapprox(omega_b[1][c] / omega_a[1][c], 2.0; rtol = 1.0e-12)
    end
end

@testset "V&V: Arrhenius — low-temperature floor (T < 200 K clamped)" begin
    # The implementation clamps T at 200 K to avoid numerical
    # underflow of exp(-E_a/(R·T)). Verify that supplying T < 200
    # yields the same rate as T = 200.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)

    reaction = CollocatedArrheniusReaction(; A = 1.0e10, E_a = 1.0e5)
    state = SpeciesState(mesh, props; fuel = 0.1, oxidizer = 0.3, product = 0.0)

    T_100 = CollocatedScalarField(:T, mesh; value = 100.0)
    T_200 = CollocatedScalarField(:T, mesh; value = 200.0)

    omega_100 = compute_arrhenius_reaction_rates(reaction, state, props, T_100, 1.0, mesh)
    omega_200 = compute_arrhenius_reaction_rates(reaction, state, props, T_200, 1.0, mesh)

    for c in 1:nc
        @test isapprox(omega_100[1][c], omega_200[1][c]; rtol = 1.0e-14)
    end
end

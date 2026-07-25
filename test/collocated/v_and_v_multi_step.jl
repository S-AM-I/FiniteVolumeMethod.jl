# test/v_and_v_multi_step.jl — Multi-step Arrhenius mechanism V&V
#
# Verifies that the generalised multi-step evaluator:
#   - reduces exactly to the existing one-step closure via
#     `one_step_arrhenius_mechanism`,
#   - preserves stoichiometric mass balance for balanced reactions,
#   - zeroes out a step whose reactant concentration vanishes,
#   - recovers the closed-form `k_f(T)` at multiple temperatures.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_arrhenius_reaction_rates, compute_multi_step_rates, one_step_arrhenius_mechanism
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const R_UNIV = 8.314

@testset "V&V: multi-step — one-step mechanism matches closed form" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)
    s = 4.0

    A = 1.0e10
    b = 0.5
    E_a = 1.0e5
    n_fuel = 1.0
    n_ox = 1.0
    reaction = CollocatedArrheniusReaction(; A = A, b = b, E_a = E_a, n_fuel = n_fuel, n_ox = n_ox)
    mechanism = one_step_arrhenius_mechanism(A, b, E_a, s; n_fuel = n_fuel, n_ox = n_ox)

    # Loop over three (T, Y) operating points.
    for (T_val, Y_f, Y_o, rho) in (
            (1500.0, 0.08, 0.2, 1.2),
            (1800.0, 0.05, 0.15, 0.8),
            (1200.0, 0.12, 0.3, 1.5),
        )
        T_field = CollocatedScalarField(:T, mesh; value = T_val)
        state = SpeciesState(mesh, props; fuel = Y_f, oxidizer = Y_o, product = 0.0)

        omega_arr = compute_arrhenius_reaction_rates(reaction, state, props, T_field, rho, mesh)
        omega_ms = compute_multi_step_rates(mechanism, state, T_field, rho, mesh)

        for c in 1:nc, i in 1:3
            @test isapprox(omega_ms[i][c], omega_arr[i][c]; rtol = 1.0e-12)
        end
    end
end

@testset "V&V: multi-step — stoichiometric mass balance Σν_net = 0" begin
    # For a balanced reaction 1·Fuel + s·Ox → (1 + s)·Prod, the net
    # stoichiometry (ν_p − ν_r) sums to zero per reaction, so Σω_i = 0.
    mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)
    mechanism = one_step_arrhenius_mechanism(1.0e10, 0.0, 1.0e5, 4.0)

    T_field = CollocatedScalarField(:T, mesh; value = 1500.0)
    state = SpeciesState(mesh, props; fuel = 0.1, oxidizer = 0.3, product = 0.05)

    omega = compute_multi_step_rates(mechanism, state, T_field, 1.0, mesh)
    for c in 1:nc
        total = omega[1][c] + omega[2][c] + omega[3][c]
        @test isapprox(total, 0.0; atol = 1.0e-10 * max(abs(omega[1][c]), 1.0))
    end
end

@testset "V&V: multi-step — zero concentration ⇒ zero rate" begin
    mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; stoich_ratio = 4.0)
    mechanism = one_step_arrhenius_mechanism(1.0e10, 0.0, 1.0e5, 4.0)

    T_field = CollocatedScalarField(:T, mesh; value = 1500.0)

    # Zero fuel ⇒ rate = 0 for all species of that reaction.
    state_nf = SpeciesState(mesh, props; fuel = 0.0, oxidizer = 0.5, product = 0.0)
    omega_nf = compute_multi_step_rates(mechanism, state_nf, T_field, 1.0, mesh)
    for c in 1:nc, i in 1:3
        @test omega_nf[i][c] == 0.0
    end

    # Zero oxidizer ⇒ rate = 0.
    state_no = SpeciesState(mesh, props; fuel = 0.5, oxidizer = 0.0, product = 0.0)
    omega_no = compute_multi_step_rates(mechanism, state_no, T_field, 1.0, mesh)
    for c in 1:nc, i in 1:3
        @test omega_no[i][c] == 0.0
    end
end

@testset "V&V: multi-step — k_f Arrhenius closed form across T" begin
    # Single-step, single-species-reactant mechanism:
    # Reaction: 1·A → 1·B with ν_r = (1, 0), ν_p = (0, 1).
    # ω_A = -ρ · k_f · Y_A; ω_B = +ρ · k_f · Y_A.
    A = 2.5e9
    b = 1.0
    E_a = 8.0e4

    nu_r = [1.0 0.0]
    nu_p = [0.0 1.0]
    mechanism = MultiStepMechanism(;
        A = (A,), b = (b,), E_a = (E_a,),
        nu_reactants = nu_r, nu_products = nu_p,
    )

    mesh = build_cartesian_unstructured_mesh(2, 2, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    # Use a 2-species `CombustionProperties` for this test.
    props = CombustionProperties(;
        species_names = (:fuel, :product),
        molecular_weights = (28.0, 44.0),
        diffusivities = (2.0e-5, 2.0e-5),
        stoich_ratio = 1.0, heat_of_combustion = 1.0,
    )
    Y_A = 0.1
    rho = 1.0
    state = SpeciesState(mesh, props; fuel = Y_A, product = 0.0)

    for T_val in (800.0, 1400.0, 2000.0)
        T_field = CollocatedScalarField(:T, mesh; value = T_val)
        omega = compute_multi_step_rates(mechanism, state, T_field, rho, mesh)
        k_f = A * T_val^b * exp(-E_a / (R_UNIV * T_val))
        expected_A = -rho * k_f * Y_A
        expected_B = +rho * k_f * Y_A
        for c in 1:nc
            @test isapprox(omega[1][c], expected_A; rtol = 1.0e-12)
            @test isapprox(omega[2][c], expected_B; rtol = 1.0e-12)
        end
    end
end

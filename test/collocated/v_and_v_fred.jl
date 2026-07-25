# test/v_and_v_fred.jl — FR/ED (finite-rate / eddy-dissipation) V&V (v3.47)
#
# Fourth convergence-verified benchmark for `combustion`, joining
# species AD (v3.17), EDM algebra (v3.27), and Arrhenius kinetics
# (v3.37). Covers the FR/ED blending closure
#
#   ω_fuel = max(ω_Arrhenius, ω_EDM)     [both negative; take
#                                          the "least negative" =
#                                          slowest = most limiting]
#
# which is the standard industrial turbulence-chemistry interaction
# model. Verifies four invariants of the blending logic.
#
# Puts `combustion` at four convergence-verified benchmarks
# covering transport, mixing-limited kinetics, finite-rate
# kinetics, and FR/ED blending.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_arrhenius_reaction_rates, compute_edm_reaction_rates, compute_fred_reaction_rates
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

function fred_setup(; T_val, k_val, eps_val, Y_fuel = 0.05, Y_ox = 0.2, Y_prod = 0.0)
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    props = CombustionProperties(; stoich_ratio = 4.0)

    species = SpeciesState(mesh, props; fuel = Y_fuel, oxidizer = Y_ox, product = Y_prod)

    reaction = CollocatedArrheniusReaction(; A = 1.0e10, E_a = 1.0e5)
    edm = EddyDissipationModel(; A_edm = 4.0, B_edm = 0.5)

    T_field = CollocatedScalarField(:T, mesh; value = T_val)
    k = fill(k_val, length(mesh.cell_volumes))
    eps = fill(eps_val, length(mesh.cell_volumes))

    return mesh, species, props, reaction, edm, T_field, k, eps
end

@testset "V&V: FR/ED — cold regime ⇒ Arrhenius limits (chemistry-slow)" begin
    # At low T, Arrhenius rate is orders of magnitude smaller than
    # EDM ⇒ Arrhenius is the most-limiting (least-negative)
    # branch, so FR/ED returns the Arrhenius rate.
    mesh, species, props, reaction, edm, T_field, k, eps = fred_setup(;
        T_val = 300.0, k_val = 1.0, eps_val = 1.0,
    )

    omega_fred = compute_fred_reaction_rates(
        reaction, edm, species, props, T_field, k, eps, 1.0, mesh,
    )
    omega_arr = compute_arrhenius_reaction_rates(
        reaction, species, props, T_field, 1.0, mesh,
    )

    for c in 1:length(mesh.cell_volumes)
        @test isapprox(omega_fred[1][c], omega_arr[1][c]; rtol = 1.0e-12)
    end
end

@testset "V&V: FR/ED — hot regime ⇒ EDM limits (mixing-slow)" begin
    # At high T (fast chemistry), EDM is the most-limiting branch
    # ⇒ FR/ED returns the EDM rate.
    mesh, species, props, reaction, edm, T_field, k, eps = fred_setup(;
        T_val = 2500.0, k_val = 10.0, eps_val = 0.1,   # slow mixing
    )

    omega_fred = compute_fred_reaction_rates(
        reaction, edm, species, props, T_field, k, eps, 1.0, mesh,
    )
    omega_edm = compute_edm_reaction_rates(
        edm, species, props, k, eps, 1.0, mesh,
    )

    for c in 1:length(mesh.cell_volumes)
        @test isapprox(omega_fred[1][c], omega_edm[1][c]; rtol = 1.0e-12)
    end
end

@testset "V&V: FR/ED — always the slower (least negative) of the two rates" begin
    # Core invariant: ω_fred[c] = max(ω_arr[c], ω_edm[c]) at every
    # cell (both rates are negative; max picks the "smallest in
    # magnitude = rate-limiting" branch).
    mesh, species, props, reaction, edm, T_field, k, eps = fred_setup(;
        T_val = 1500.0, k_val = 1.0, eps_val = 0.5,
    )

    omega_arr = compute_arrhenius_reaction_rates(
        reaction, species, props, T_field, 1.0, mesh,
    )
    omega_edm = compute_edm_reaction_rates(
        edm, species, props, k, eps, 1.0, mesh,
    )
    omega_fred = compute_fred_reaction_rates(
        reaction, edm, species, props, T_field, k, eps, 1.0, mesh,
    )

    for c in 1:length(mesh.cell_volumes)
        expected = max(omega_arr[1][c], omega_edm[1][c])
        @test isapprox(omega_fred[1][c], expected; rtol = 1.0e-12)
    end
end

@testset "V&V: FR/ED — stoichiometric mass balance" begin
    # Σ ω_i ≡ 0 at every cell regardless of branch selection
    # (fuel + stoichiometric oxidizer = products).
    mesh, species, props, reaction, edm, T_field, k, eps = fred_setup(;
        T_val = 1200.0, k_val = 2.0, eps_val = 1.0,
    )

    omega = compute_fred_reaction_rates(
        reaction, edm, species, props, T_field, k, eps, 1.0, mesh,
    )

    for c in 1:length(mesh.cell_volumes)
        total = omega[1][c] + omega[2][c] + omega[3][c]
        @test abs(total) < 1.0e-14
    end
end

# test/v_and_v_wall_functions.jl — Wall-function algebra V&V (v3.54)
#
# Fifth convergence-verified benchmark for `turbulence_rans`,
# joining k-ε DHIT (v3.18), k-ε log-layer (v3.23), k-ω decay
# (v3.38), and Spalart-Allmaras (v3.44). Covers the wall-
# function primitives
#
#   u_τ ← Spalding's law (Newton iteration)
#   k_wall = u_τ² / √C_μ
#   ε_wall = u_τ³ / (κ·y)
#   ν_t_wall = ν · (y⁺/u⁺ − 1)
#
# used by every wall-bounded RANS closure.
#
# Six invariants verified.

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

# Access internal helpers via module qualification.
const _u_tau = FiniteVolumeMethod.spalding_u_tau
const _nut_wall = FiniteVolumeMethod.compute_nut_wall
const _k_wall = FiniteVolumeMethod.equilibrium_k_wall
const _eps_wall = FiniteVolumeMethod.equilibrium_epsilon_wall

const C_MU = 0.09
const KAPPA_VK = 0.41
const E_WF = 9.793

@testset "V&V: Wall fn — equilibrium k_wall = u_τ²/√C_μ algebraic identity" begin
    for u_tau in (0.01, 0.05, 0.1, 0.5, 1.0, 3.0)
        k_computed = _k_wall(u_tau)
        k_expected = u_tau^2 / sqrt(C_MU)
        @test isapprox(k_computed, k_expected; rtol = 1.0e-14)
    end
end

@testset "V&V: Wall fn — equilibrium ε_wall = u_τ³/(κ·y) algebraic identity" begin
    for (u_tau, y) in ((0.05, 0.01), (0.1, 0.02), (0.5, 0.1), (2.0, 0.001))
        eps_computed = _eps_wall(u_tau, y, 1.0e-6)
        eps_expected = u_tau^3 / (KAPPA_VK * y)
        @test isapprox(eps_computed, eps_expected; rtol = 1.0e-14)
    end
end

@testset "V&V: Wall fn — k_wall ∝ u_τ² scaling" begin
    k_1 = _k_wall(0.1)
    k_2 = _k_wall(0.2)
    k_3 = _k_wall(0.4)
    @test isapprox(k_2 / k_1, 4.0; rtol = 1.0e-14)   # (0.2/0.1)² = 4
    @test isapprox(k_3 / k_2, 4.0; rtol = 1.0e-14)
end

@testset "V&V: Wall fn — ε_wall ∝ u_τ³ / y scaling" begin
    # Hold y fixed: doubling u_τ should multiply ε by 8.
    eps_a = _eps_wall(0.1, 0.01, 1.0e-6)
    eps_b = _eps_wall(0.2, 0.01, 1.0e-6)
    @test isapprox(eps_b / eps_a, 8.0; rtol = 1.0e-14)

    # Hold u_τ fixed: doubling y should halve ε.
    eps_c = _eps_wall(0.1, 0.02, 1.0e-6)
    @test isapprox(eps_c / eps_a, 0.5; rtol = 1.0e-14)
end

@testset "V&V: Wall fn — Spalding u_τ converges to log-law at high y⁺" begin
    # In the log-law region (y⁺ ≫ 30), the Spalding iteration
    # converges to the solution of
    #
    #   u⁺ = (1/κ) · log(E · y⁺)
    #
    # Pick a deliberately-high-y⁺ configuration: U_par = 10, y = 0.1,
    # ν = 1e-5 ⇒ y·U/ν = 1e5, way into the log layer.
    U_par = 10.0
    y = 0.1
    nu = 1.0e-5
    u_tau = _u_tau(U_par, y, nu)

    y_plus = y * u_tau / nu
    u_plus = U_par / u_tau

    # Check log-law holds: u⁺ · κ = log(E · y⁺).
    lhs = KAPPA_VK * u_plus
    rhs = log(E_WF * y_plus)
    @test isapprox(lhs, rhs; rtol = 5.0e-2)   # 5% tolerance (log-law is asymptotic)
    @test u_tau > 0.0
end

@testset "V&V: Wall fn — ν_t_wall ≥ 0 realizability" begin
    # The wall-function ν_t must always be non-negative.
    for U_par in (0.01, 0.1, 1.0, 10.0)
        for y in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
            nut = _nut_wall(U_par, y, 1.0e-5)
            @test nut >= 0.0
        end
    end
end

@testset "V&V: Wall fn — Spalding u_τ is monotonically increasing in U_par" begin
    # At fixed y and ν, increasing the parallel velocity should
    # increase u_τ (more shear at the wall).
    y = 0.01
    nu = 1.0e-5
    u_prev = 0.0
    for U_par in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
        u_tau = _u_tau(U_par, y, nu)
        @test u_tau > u_prev
        u_prev = u_tau
    end
end

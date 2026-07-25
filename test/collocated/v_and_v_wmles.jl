# test/v_and_v_wmles.jl — Equilibrium WMLES primitives V&V (v3.0 / Wave 1)
#
# Verifies the closed-form primitives
#
#     (ν_t, active) = wmles_wall_nut(U_par, y, ν; y_plus_switch)
#     τ_w           = wmles_wall_shear(U_par, y, ν, ρ)
#
# that sit on top of the Spalding Newton iteration already exercised
# by `v_and_v_wall_functions.jl`. The focus here is the WMLES-specific
# switch logic and the algebraic equivalence `τ_w = ρ u_τ²`.

using FiniteVolumeMethod
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const _u_tau = FiniteVolumeMethod.spalding_u_tau
const _wmles_nut = FiniteVolumeMethod.wmles_wall_nut
const _wmles_tau = FiniteVolumeMethod.wmles_wall_shear

@testset "V&V: WMLES — τ_w = ρ·u_τ² at three y⁺ values" begin
    nu = 1.0e-5
    for (U_par, y) in ((1.0, 1.0e-4), (5.0, 1.0e-3), (10.0, 1.0e-2))
        u_tau = _u_tau(U_par, y, nu)
        y_plus = y * u_tau / nu
        # Sanity: we picked configurations that span the log layer.
        @test y_plus > 1.0
        for rho in (1.0, 1.2, 1000.0)
            tau_w = _wmles_tau(U_par, y, nu, rho)
            @test isapprox(tau_w, rho * u_tau^2; rtol = 1.0e-12)
        end
    end
end

@testset "V&V: WMLES — zero velocity ⇒ zero τ_w" begin
    nu = 1.0e-5
    for y in (1.0e-5, 1.0e-3, 1.0e-1)
        for rho in (1.0, 1.2, 1000.0)
            @test _wmles_tau(0.0, y, nu, rho) == 0.0
        end
    end
end

@testset "V&V: WMLES — zero velocity ⇒ inactive wall branch" begin
    nu = 1.0e-5
    nut, active = _wmles_nut(0.0, 1.0e-3, nu; y_plus_switch = 30.0)
    @test nut == 0.0
    @test active == false
end

@testset "V&V: WMLES — low y⁺ branch is inactive (sublayer resolved)" begin
    # With ν = 1e-3 and (U_par, y) = (0.1, 1e-3) the Spalding iteration
    # yields y⁺ ≈ O(0.1), well below the switch. Branch must return
    # `active = false`.
    nu = 1.0e-3
    nut, active = _wmles_nut(0.1, 1.0e-3, nu; y_plus_switch = 30.0)
    @test active == false
    @test nut == 0.0
end

@testset "V&V: WMLES — high y⁺ branch is active and matches closed form" begin
    # Pick a config deep in the log layer. Check that `active = true`
    # and that the returned ν_t satisfies ν_t = ν·(y⁺/u⁺ − 1).
    U_par = 10.0
    y = 0.05
    nu = 1.0e-5
    nut, active = _wmles_nut(U_par, y, nu; y_plus_switch = 30.0)
    @test active == true
    u_tau = _u_tau(U_par, y, nu)
    y_plus = y * u_tau / nu
    u_plus = U_par / u_tau
    @test isapprox(nut, nu * (y_plus / u_plus - 1.0); rtol = 1.0e-12)
    @test nut > 0.0
end

@testset "V&V: WMLES — wall-parallel velocity bounded by U_par input" begin
    # Sanity: the Spalding solution gives u⁺ ≥ 1 in the log layer, so
    # u_τ ≤ U_par. Consequently τ_w ≤ ρ · U_par² — a useful bound for
    # downstream solver diagnostics.
    nu = 1.0e-5
    rho = 1.0
    for (U_par, y) in ((1.0, 1.0e-3), (5.0, 1.0e-3), (10.0, 1.0e-2))
        tau_w = _wmles_tau(U_par, y, nu, rho)
        @test 0.0 <= tau_w <= rho * U_par^2 + 1.0e-12
    end
end

@testset "V&V: WMLES — constructor wires wall cells and filter width" begin
    # Construct an EquilibriumWMLES on a Cartesian mesh and verify that
    # wall-cell bookkeeping is populated (nx entries along :bottom).
    nx = 4; ny = 4
    mesh = build_cartesian_unstructured_mesh(nx, ny, 1.0, 1.0)
    model = FiniteVolumeMethod.EquilibriumWMLES(mesh, Symbol[:bottom]; Cs = 0.1)
    @test length(model.wall_cells) == nx
    @test length(model.wall_faces) == nx
    @test length(model.delta) == nx * ny
    @test all(model.delta .> 0)
    @test model.y_plus_switch == 30.0
end

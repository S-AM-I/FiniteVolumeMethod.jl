# test/v_and_v_marshak.jl — Marshak wall BC + radiation inlet V&V (v3.60)
#
# Fifth convergence-verified benchmark for `radiation`, joining
# P1 slab attenuation (v3.15), radiative equilibrium (v3.25),
# source-term algebra (v3.35), and fvDOM quadrature (v3.48).
# Covers the `marshak_wall_bc` and `radiation_inlet_bc`
# constructor primitives.
#
# Six invariants verified.

using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: DirichletBC, RobinBC
using Test

include("TestHelpers.jl")

const SIGMA_SB = 5.670374419e-8

@testset "V&V: Marshak BC — RobinBC(a=1, b=2/(3a), c=4σT⁴)" begin
    # For P1 with absorption a_val, marshak BC is
    # G + (2/(3a)) · ∂G/∂n = 4σT⁴.
    # This maps to RobinBC(1, 2/(3a), 4σT⁴).
    rad = P1Model(; a = 0.5)
    T_wall = 500.0

    bc = marshak_wall_bc(rad, T_wall)

    @test bc isa RobinBC
    @test bc.a == 1.0
    @test isapprox(bc.b, 2.0 / (3 * 0.5); rtol = 1.0e-14)
    @test isapprox(bc.c, 4 * SIGMA_SB * T_wall^4; rtol = 1.0e-14)
end

@testset "V&V: Marshak BC — T⁴ scaling" begin
    # c = 4σT⁴; doubling T multiplies c by 16.
    rad = P1Model(; a = 1.0)
    bc_a = marshak_wall_bc(rad, 300.0)
    bc_b = marshak_wall_bc(rad, 600.0)
    @test isapprox(bc_b.c / bc_a.c, 16.0; rtol = 1.0e-14)
end

@testset "V&V: Marshak BC — 1/a scaling of b coefficient" begin
    # b = 2/(3a): doubling a halves b.
    bc_a = marshak_wall_bc(P1Model(; a = 0.1), 300.0)
    bc_b = marshak_wall_bc(P1Model(; a = 0.2), 300.0)
    bc_c = marshak_wall_bc(P1Model(; a = 1.0), 300.0)

    @test isapprox(bc_a.b / bc_b.b, 2.0; rtol = 1.0e-14)
    @test isapprox(bc_a.b / bc_c.b, 10.0; rtol = 1.0e-14)
end

@testset "V&V: radiation_inlet_bc — G = 4σT⁴ Dirichlet" begin
    T_inlet = 400.0
    bc = radiation_inlet_bc(T_inlet)
    @test bc isa DirichletBC
    @test isapprox(bc.value, 4 * SIGMA_SB * T_inlet^4; rtol = 1.0e-14)
end

@testset "V&V: radiation_inlet_bc — T = 0 ⇒ G = 0" begin
    bc = radiation_inlet_bc(0.0)
    @test isapprox(bc.value, 0.0; atol = 1.0e-14)
end

@testset "V&V: Marshak + inlet — consistency at matching T" begin
    # At equilibrium T_medium = T_wall, the steady P1 solution is
    # G ≡ 4σT⁴. The Marshak BC coefficient c = 4σT⁴ equals the
    # radiation_inlet_bc value at the same T — they encode the
    # same blackbody emissive power.
    T = 750.0
    rad = P1Model(; a = 0.5)

    marshak = marshak_wall_bc(rad, T)
    inlet = radiation_inlet_bc(T)

    @test isapprox(marshak.c, inlet.value; rtol = 1.0e-14)
end

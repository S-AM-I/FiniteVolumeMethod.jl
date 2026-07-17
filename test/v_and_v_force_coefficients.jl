# test/v_and_v_force_coefficients.jl — force_coefficients algebra V&V (v3.72)

using FiniteVolumeMethod
using FiniteVolumeMethod: force_coefficients
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: force_coefficients — zero force ⇒ zero coeffs" begin
    F0 = SVector(0.0, 0.0)
    result = force_coefficients(
        F0, F0;
        rho_ref = 1.2, U_ref = 10.0, A_ref = 1.0
    )
    @test result.Cd == 0.0
    @test result.Cl == 0.0
    @test result.Cd_pressure == 0.0
    @test result.Cd_viscous == 0.0
end

@testset "V&V: force_coefficients — Cd formula (F/qA)" begin
    F_p = SVector(100.0, 0.0)
    F_v = SVector(50.0, 0.0)
    rho = 1.2
    U = 10.0
    A = 2.0

    result = force_coefficients(F_p, F_v; rho_ref = rho, U_ref = U, A_ref = A)
    q = 0.5 * rho * U^2
    qA = q * A

    # Cd = (F_p + F_v).x / qA
    @test isapprox(result.Cd, 150.0 / qA; rtol = 1.0e-14)
    @test isapprox(result.Cd_pressure, 100.0 / qA; rtol = 1.0e-14)
    @test isapprox(result.Cd_viscous, 50.0 / qA; rtol = 1.0e-14)
    @test isapprox(result.Cl, 0.0; atol = 1.0e-14)
end

@testset "V&V: force_coefficients — Cl perpendicular to drag" begin
    F_p = SVector(0.0, 200.0)
    F_v = SVector(0.0, 0.0)
    result = force_coefficients(
        F_p, F_v;
        rho_ref = 1.0, U_ref = 10.0, A_ref = 1.0
    )
    q = 0.5 * 1.0 * 100.0  # = 50
    @test isapprox(result.Cd, 0.0; atol = 1.0e-14)
    @test isapprox(result.Cl, 200.0 / 50.0; rtol = 1.0e-14)
end

@testset "V&V: force_coefficients — U² inverse scaling" begin
    F_p = SVector(100.0, 0.0)
    F_v = SVector(0.0, 0.0)
    r1 = force_coefficients(F_p, F_v; rho_ref = 1.0, U_ref = 10.0, A_ref = 1.0)
    r2 = force_coefficients(F_p, F_v; rho_ref = 1.0, U_ref = 20.0, A_ref = 1.0)
    # Doubling U reduces Cd by 4× (since q ∝ U²).
    @test isapprox(r1.Cd / r2.Cd, 4.0; rtol = 1.0e-14)
end

@testset "V&V: force_coefficients — custom drag/lift directions" begin
    F_p = SVector(3.0, 4.0)
    F_v = SVector(0.0, 0.0)
    # Use a 45° direction as drag.
    drag = SVector(1.0, 1.0) / sqrt(2.0)
    lift = SVector(-1.0, 1.0) / sqrt(2.0)
    result = force_coefficients(
        F_p, F_v;
        rho_ref = 1.0, U_ref = 1.0, A_ref = 1.0,
        drag_direction = drag, lift_direction = lift
    )
    q = 0.5
    F_dot_drag = 3.0 / sqrt(2.0) + 4.0 / sqrt(2.0)
    F_dot_lift = -3.0 / sqrt(2.0) + 4.0 / sqrt(2.0)
    @test isapprox(result.Cd, F_dot_drag / q; rtol = 1.0e-14)
    @test isapprox(result.Cl, F_dot_lift / q; rtol = 1.0e-14)
end

@testset "V&V: force_coefficients — A_ref linear scaling" begin
    F_p = SVector(100.0, 0.0)
    F_v = SVector(0.0, 0.0)
    r1 = force_coefficients(F_p, F_v; rho_ref = 1.0, U_ref = 10.0, A_ref = 1.0)
    r2 = force_coefficients(F_p, F_v; rho_ref = 1.0, U_ref = 10.0, A_ref = 2.0)
    @test isapprox(r1.Cd / r2.Cd, 2.0; rtol = 1.0e-14)
end

@testset "V&V: force_coefficients — zero qA ⇒ zero fallback" begin
    F_p = SVector(100.0, 0.0)
    F_v = SVector(50.0, 0.0)
    r = force_coefficients(F_p, F_v; rho_ref = 0.0, U_ref = 0.0, A_ref = 0.0)
    @test r.Cd == 0.0
    @test r.Cl == 0.0
    @test r.Cd_pressure == 0.0
    @test r.Cd_viscous == 0.0
end

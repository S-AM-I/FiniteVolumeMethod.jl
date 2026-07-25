# test/v_and_v_expression_bc.jl — runtime expression BC parse + evaluate V&V.

using FiniteVolumeMethod
using FiniteVolumeMethod: StringExpressionBC
using Test

@testset "V&V: StringExpressionBC — sin(x) + cos(y) at (0, 0) = 1" begin
    bc = StringExpressionBC("sin(x) + cos(y)")
    v = evaluate(bc, 0.0, 0.0, 0.0, 0.0)
    @test isapprox(v, 1.0; rtol = 1.0e-14)
end

@testset "V&V: StringExpressionBC — 2·t evaluated at t = 3 = 6" begin
    bc = StringExpressionBC("2 * t")
    v = evaluate(bc, 0.0, 0.0, 0.0, 3.0)
    @test v == 6.0
end

@testset "V&V: StringExpressionBC — π·x² evaluated at x = 1 = π" begin
    bc = StringExpressionBC("pi * x^2")
    v = evaluate(bc, 1.0, 0.0, 0.0, 0.0)
    @test isapprox(v, π; rtol = 1.0e-14)
end

@testset "V&V: StringExpressionBC — user constant L scales output" begin
    bc = StringExpressionBC("x / L"; L = 2.0)
    v = evaluate(bc, 6.0, 0.0, 0.0, 0.0)
    @test v == 3.0
end

@testset "V&V: StringExpressionBC — cached compile (second call works)" begin
    bc = StringExpressionBC("x + y + z + t")
    v1 = evaluate(bc, 1.0, 2.0, 3.0, 4.0)
    v2 = evaluate(bc, 0.0, 0.0, 0.0, 0.0)
    @test v1 == 10.0
    @test v2 == 0.0
end

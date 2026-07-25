# test/v_and_v_fvdom_model.jl — FvDOMModel primitive V&V (v3.81)

using FiniteVolumeMethod
using FiniteVolumeMethod: AbstractRadiationModel
using LinearAlgebra: norm
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: FvDOMModel — 2D S2 field population" begin
    m = FvDOMModel(; a = 0.5, Dim = 2, order = :S2)
    @test length(m.directions) == 4
    @test length(m.weights) == 4
    # Sum of weights matches 2π in 2D.
    @test isapprox(sum(m.weights), 2 * pi; rtol = 1.0e-14)
    # Directions are unit vectors.
    for d in m.directions
        @test isapprox(norm(d), 1.0; rtol = 1.0e-14)
    end
end

@testset "V&V: FvDOMModel — 3D S2 field population" begin
    m = FvDOMModel(; a = 0.5, Dim = 3, order = :S2)
    @test length(m.directions) == 8
    @test length(m.weights) == 8
    @test isapprox(sum(m.weights), 4 * pi; rtol = 1.0e-14)
    for d in m.directions
        @test isapprox(norm(d), 1.0; rtol = 1.0e-14)
    end
end

@testset "V&V: FvDOMModel — 2D S4 field population" begin
    m = FvDOMModel(; a = 0.5, Dim = 2, order = :S4)
    @test length(m.directions) == 12
    @test length(m.weights) == 12
    for d in m.directions
        @test isapprox(norm(d), 1.0; rtol = 1.0e-6)
    end
end

@testset "V&V: FvDOMModel — default Dim = 2, order = :S2" begin
    m = FvDOMModel()
    @test length(m.directions) == 4   # default = 2D S2
end

@testset "V&V: FvDOMModel — scalar absorption a round-trip" begin
    m = FvDOMModel(; a = 0.7)
    @test m.a == 0.7
end

@testset "V&V: FvDOMModel — unknown order errors" begin
    @test_throws ErrorException FvDOMModel(; order = :unknown)
end

@testset "V&V: FvDOMModel — dispatch via AbstractRadiationModel" begin
    m = FvDOMModel()
    @test m isa FiniteVolumeMethod.AbstractRadiationModel
end

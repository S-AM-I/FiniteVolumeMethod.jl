# test/v_and_v_radiation_state.jl — RadiationState + P1Model V&V (v3.68)

using FiniteVolumeMethod
using FiniteVolumeMethod: AbstractRadiationModel, RadiationState
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: RadiationState — default G = 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    state = FiniteVolumeMethod.RadiationState(mesh)
    @test length(state.G.internal) == 64
    for v in state.G.internal
        @test v == 0.0
    end
end

@testset "V&V: RadiationState — custom G_init round-trip" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = FiniteVolumeMethod.RadiationState(mesh; G_init = 100.0)
    for v in state.G.internal
        @test v == 100.0
    end
end

@testset "V&V: P1Model — scalar a round-trip" begin
    rad = P1Model(; a = 0.3)
    @test rad.a == 0.3
end

@testset "V&V: P1Model — vector a round-trip" begin
    a_vec = [0.1, 0.2, 0.3, 0.4]
    rad = P1Model(; a = a_vec)
    @test rad.a == a_vec
end

@testset "V&V: P1Model — default a = 0.1" begin
    rad = P1Model()
    @test rad.a == 0.1
end

@testset "V&V: RadiationState — size matches mesh" begin
    for N in (4, 8, 16, 32)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        state = FiniteVolumeMethod.RadiationState(mesh)
        @test length(state.G.internal) == N * N
    end
end

@testset "V&V: P1Model — AbstractRadiationModel dispatch" begin
    rad = P1Model()
    @test rad isa FiniteVolumeMethod.AbstractRadiationModel
end

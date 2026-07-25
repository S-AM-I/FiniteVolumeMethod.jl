# test/v_and_v_vof_state.jl — VOFState + TwoPhaseProperties V&V (v3.69)

using FiniteVolumeMethod
using FiniteVolumeMethod: has_surface_tension
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: TwoPhaseProperties — kwargs round-trip" begin
    props = TwoPhaseProperties(;
        rho1 = 1000.0, rho2 = 1.2,
        mu1 = 1.0e-3, mu2 = 1.8e-5,
        sigma = 0.072,
    )
    @test props.rho1 == 1000.0
    @test props.rho2 == 1.2
    @test props.mu1 == 1.0e-3
    @test props.mu2 == 1.8e-5
    @test props.sigma == 0.072
end

@testset "V&V: TwoPhaseProperties — has_surface_tension detection" begin
    no_st = TwoPhaseProperties(; rho1 = 1.0, rho2 = 1.0, mu1 = 1.0e-3, mu2 = 1.0e-3, sigma = 0.0)
    with_st = TwoPhaseProperties(; rho1 = 1.0, rho2 = 1.0, mu1 = 1.0e-3, mu2 = 1.0e-3, sigma = 0.072)
    @test FiniteVolumeMethod.has_surface_tension(no_st) == false
    @test FiniteVolumeMethod.has_surface_tension(with_st) == true
end

@testset "V&V: VOFState — default α = 0 initialization" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = VOFState(mesh)
    @test length(state.alpha.internal) == 16
    @test length(state.rho) == 16
    @test length(state.mu) == 16
    for v in state.alpha.internal
        @test v == 0.0
    end
end

@testset "V&V: VOFState — custom alpha_init round-trip" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = VOFState(mesh; alpha_init = 0.6)
    for v in state.alpha.internal
        @test v == 0.6
    end
end

@testset "V&V: VOFState — function-valued alpha_init" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    props = TwoPhaseProperties(; rho1 = 1.0, rho2 = 1.0, mu1 = 1.0e-3, mu2 = 1.0e-3, sigma = 0.0)
    state = VOFState(mesh, x -> x[1] > 0.5 ? 1.0 : 0.0, props)

    for c in 1:length(mesh.cell_volumes)
        x = mesh.cell_centers[1, c]
        expected = x > 0.5 ? 1.0 : 0.0
        @test state.alpha.internal[c] == expected
    end
end

@testset "V&V: VOFState — size matches mesh" begin
    for N in (4, 8, 16)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        state = VOFState(mesh)
        @test length(state.alpha.internal) == N * N
        @test length(state.rho) == N * N
        @test length(state.mu) == N * N
    end
end

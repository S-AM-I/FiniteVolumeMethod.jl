# test/v_and_v_inc_state.jl — IncompressibleState V&V (v3.75)

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: IncompressibleState — zero init" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = IncompressibleState(mesh)
    nc = length(mesh.cell_volumes)
    @test length(state.U.internal) == nc
    @test length(state.p.internal) == nc
    for u in state.U.internal
        @test u[1] == 0.0 && u[2] == 0.0
    end
    for p in state.p.internal
        @test p == 0.0
    end
end

@testset "V&V: IncompressibleState — phi, A_P, H_U shapes" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = IncompressibleState(mesh)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    @test length(state.phi.values) == nf
    @test length(state.A_P) == nc
    @test length(state.H_U) == nc
end

@testset "V&V: IncompressibleState — sizing across N" begin
    for N in (4, 8, 16, 32)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        state = IncompressibleState(mesh)
        @test length(state.U.internal) == N * N
        @test length(state.p.internal) == N * N
    end
end

@testset "V&V: IncompressibleProblem — kwargs round-trip" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(0.1, 0.0)),
    )
    prob = IncompressibleProblem(
        mesh, bcs, SIMPLE(0.5, 0.2, 10, 1.0e-5);
        nu = 0.07, density = 2.5
    )
    @test prob.nu == 0.07
    @test prob.density == 2.5
    @test prob.mesh === mesh
end

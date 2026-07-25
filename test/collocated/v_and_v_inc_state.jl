# test/v_and_v_inc_state.jl — IncompressibleState V&V (v3.75)

using FiniteVolumeMethod
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

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

@testset "V&V: IncompressibleState — flat backing store (Stage 5f)" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = IncompressibleState(mesh)
    nc = length(mesh.cell_volumes)

    # Flat vector holds the velocity block then the pressure block.
    @test length(state.u) == nc * 2 + nc

    # The fields are VIEWS into `u`: writing a field mutates the backing vector.
    state.U.internal[1] = SVector(7.0, 9.0)
    @test state.u[1] == 7.0
    @test state.u[2] == 9.0
    state.p.internal[1] = 3.5
    @test state.u[nc * 2 + 1] == 3.5

    # ...and writing the backing vector shows through the field view.
    state.u[3] = -2.0
    @test state.U.internal[2][1] == -2.0

    # The state and its primary-field containers are concretely typed, so
    # solver-loop field access is type-stable (fixes the pre-5f instability).
    @test isconcretetype(typeof(state))
    @test isconcretetype(typeof(state.U.internal))
    @test isconcretetype(typeof(state.p.internal))

    # A deep copy is independent and identically typed (view-backed).
    copy_state = FiniteVolumeMethod.Collocated._copy_state(state, mesh)
    @test typeof(copy_state) === typeof(state)
    copy_state.U.internal[1] = SVector(0.0, 0.0)
    @test state.U.internal[1] == SVector(7.0, 9.0)  # original untouched
end

@testset "V&V: IncompressibleProblem — kwargs round-trip" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(0.1, 0.0)),
    )
    prob = SteadyIncompressibleProblem(
        mesh, bcs, SIMPLE(0.5, 0.2, 10, 1.0e-5);
        nu = 0.07, density = 2.5
    )
    @test prob.nu == 0.07
    @test prob.density == 2.5
    @test prob.mesh === mesh
end

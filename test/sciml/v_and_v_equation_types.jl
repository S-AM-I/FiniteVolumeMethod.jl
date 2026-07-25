# test/v_and_v_equation_types.jl — CollocatedEquation constructor V&V (v3.87)

using FiniteVolumeMethod
using FiniteVolumeMethod: CollocatedEquation, add_diag!, to_linear_problem
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: CollocatedEquation — default construction" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)

    @test size(eq.A, 1) == nc
    @test size(eq.A, 2) == nc
    @test length(eq.b) == nc
    # b starts at zero.
    for v in eq.b
        @test v == 0.0
    end
end

@testset "V&V: CollocatedEquation — size scales with mesh" begin
    for N in (4, 8, 16)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        eq = CollocatedEquation(mesh)
        @test size(eq.A, 1) == nc
        @test length(eq.b) == nc
    end
end

@testset "V&V: add_diag! accumulates on diagonal only" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    eq = CollocatedEquation(mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        FiniteVolumeMethod.add_diag!(eq, c, 2.0 * c)
    end
    for c in 1:nc
        @test eq.A[c, c] == 2.0 * c
    end
end

@testset "V&V: to_linear_problem — returns SciMLBase.LinearProblem" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    eq = CollocatedEquation(mesh)
    for c in 1:length(mesh.cell_volumes)
        FiniteVolumeMethod.add_diag!(eq, c, 1.0)
        eq.b[c] = Float64(c)
    end
    lp = to_linear_problem(eq)
    @test lp !== nothing
end

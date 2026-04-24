# test/v_and_v_petsc_stub.jl — PETSc weak-dep stub ergonomics.

using FiniteVolumeMethod
using Test

@testset "V&V: PETScLinearSolver — error without PETSc.jl loaded" begin
    @test_throws ErrorException PETScLinearSolver()
end

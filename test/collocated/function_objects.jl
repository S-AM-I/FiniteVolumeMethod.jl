# test/collocated/function_objects.jl — runtime probes, expression BCs and field statistics.

using FiniteVolumeMethod
using FiniteVolumeMethod: AbstractFVMBoundaryCondition, ExpressionBC, FieldStatistics, ForceProbe, PointProbe, evaluate_expression_bc
using Test
using StaticArrays: SVector

@testset "PointProbe accumulates samples" begin
    extract = (state, c) -> state.T[c]
    probe = PointProbe(:T_probe, SVector(0.5, 0.5), 1, extract)
    # Fake state with a temperature field
    state = (T = [300.0, 400.0],)
    FiniteVolumeMethod.run!(probe, state, 0.1, 1)
    FiniteVolumeMethod.run!(probe, state, 0.2, 2)
    @test length(probe.history) == 2
    @test probe.history[1] == (0.1, 300.0)
    @test probe.history[2] == (0.2, 300.0)
end

@testset "ForceProbe sums user-computed force" begin
    faces = [10, 20, 30]
    compute = (state, fs) -> SVector(1.0, 2.0)  # dummy 2D force
    fp = ForceProbe(:drag, faces, compute, Val(2), Float64)
    FiniteVolumeMethod.run!(fp, nothing, 0.0, 1)
    @test length(fp.history) == 1
    @test fp.history[1][2] == SVector(1.0, 2.0)
end

@testset "ExpressionBC evaluates closure at (x, t)" begin
    # Pulsating inlet: u_in(t) = sin(2π t)
    bc = ExpressionBC((x, t) -> SVector(sin(2π * t), 0.0), Val(2), Float64)
    @test bc isa AbstractFVMBoundaryCondition
    u_at_025 = evaluate_expression_bc(bc, SVector(0.0, 0.5), 0.25)
    @test u_at_025[1] ≈ 1.0 atol = 1.0e-12
    u_at_0 = evaluate_expression_bc(bc, SVector(0.0, 0.5), 0.0)
    @test u_at_0 ≈ SVector(0.0, 0.0)
end

@testset "FieldStatistics running average" begin
    stats = FieldStatistics(:T_mean, 3, Float64)
    FiniteVolumeMethod.update!(stats, [1.0, 2.0, 3.0])
    FiniteVolumeMethod.update!(stats, [3.0, 4.0, 5.0])
    @test stats.n_samples == 2
    @test stats.mean ≈ [2.0, 3.0, 4.0]
end

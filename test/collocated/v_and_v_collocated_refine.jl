# test/v_and_v_collocated_refine.jl — refinement + coarsening conservation V&V.

using FiniteVolumeMethod
using FiniteVolumeMethod: CoarseningPlan, RefinementPlan, apply_coarsening!, apply_refinement!, mark_for_refinement
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: refinement — single-cell split preserves Σ V" begin
    volumes = [1.0, 1.0, 1.0, 1.0]
    field = [1.0, 2.0, 3.0, 4.0]
    plan = RefinementPlan([2], 2)
    f_new, v_new = apply_refinement!(field, volumes, plan)
    @test sum(v_new) ≈ sum(volumes) rtol = 1.0e-14
end

@testset "V&V: refinement — Σ φ·V conserved" begin
    volumes = [2.0, 1.0, 1.0]
    field = [10.0, 5.0, 1.0]
    plan = RefinementPlan([1], 2)
    f_new, v_new = apply_refinement!(field, volumes, plan)
    @test sum(f_new .* v_new) ≈ sum(field .* volumes) rtol = 1.0e-14
end

@testset "V&V: refinement — 2^Dim children produced" begin
    volumes = [1.0, 1.0]
    field = [0.0, 0.0]
    plan2d = RefinementPlan([1], 2)   # children_per_cell = 4
    f2, v2 = apply_refinement!(field, volumes, plan2d)
    @test length(v2) == 2 + (4 - 1)
    plan3d = RefinementPlan([1], 3)   # children_per_cell = 8
    f3, v3 = apply_refinement!(field, volumes, plan3d)
    @test length(v3) == 2 + (8 - 1)
end

@testset "V&V: coarsening — Σ φ·V conserved" begin
    volumes = [0.5, 0.5, 0.5, 0.5, 1.0]
    field = [1.0, 2.0, 3.0, 4.0, 10.0]
    plan = CoarseningPlan([[1, 2, 3, 4]])
    f_new, v_new = apply_coarsening!(field, volumes, plan)
    @test sum(f_new .* v_new) ≈ sum(field .* volumes) rtol = 1.0e-14
    @test sum(v_new) ≈ sum(volumes) rtol = 1.0e-14
end

@testset "V&V: mark_for_refinement — threshold behaviour" begin
    indicator = [0.1, 0.5, 0.9, 0.05]
    @test mark_for_refinement(indicator, 0.2) == [2, 3]
    @test mark_for_refinement(indicator, 1.0) == Int[]
    @test mark_for_refinement(indicator, 0.0) == [1, 2, 3, 4]
end

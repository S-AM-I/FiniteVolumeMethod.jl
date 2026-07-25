# test/v_and_v_topo_changer.jl — Topology-change conservation V&V
#
# Invariants checked:
# 1. Single split: Σ V_new == Σ V_old  (rtol 1e-12)
# 2. Single split of a constant field: Σ φ·V conserved (rtol 1e-12)
# 3. Single merge: Σ V_new == Σ V_old  (rtol 1e-12)
# 4. Single merge of a linear field: Σ φ·V conserved (rtol 1e-12)

using Test

include(joinpath(@__DIR__, "..", "..", "src", "collocated", "dynamic_mesh", "topo_changer.jl"))

@testset "V&V topoChanger: single split preserves total volume" begin
    volumes = [1.0, 2.0, 3.0, 4.0]
    fields = [[10.0, 20.0, 30.0, 40.0]]

    split = CellSplit{Float64}(3, [5, 6, 7], [0.25, 0.25, 0.5])
    plan = TopoChanger{Float64}([split], CellMerge{Float64}[])

    new_V, new_fields = apply_topo_change!(volumes, fields, plan)

    @test sum(new_V) ≈ sum(volumes) rtol = 1.0e-12
    # Number of cells: 4 - 1 + 3 = 6
    @test length(new_V) == 6
    # Children have volume fractions 0.25, 0.25, 0.5 of V_parent = 3.0
    # Verify they appear somewhere in the new volume list.
    child_volumes_expected = sort([0.25 * 3.0, 0.25 * 3.0, 0.5 * 3.0])
    child_volumes_actual = sort([new_V[end - 2], new_V[end - 1], new_V[end]])
    @test child_volumes_actual ≈ child_volumes_expected rtol = 1.0e-12
end

@testset "V&V topoChanger: split preserves Σ φ·V for constant field" begin
    volumes = [1.0, 1.0, 1.0, 1.0]
    fields = [[5.0, 5.0, 5.0, 5.0]]

    total_old = sum(fields[1] .* volumes)

    split = CellSplit{Float64}(2, [5, 6], [0.4, 0.6])
    plan = TopoChanger{Float64}([split], CellMerge{Float64}[])

    _, new_fields = apply_topo_change!(volumes, fields, plan)
    new_V, _ = apply_topo_change!(volumes, fields, plan)

    total_new = sum(new_fields[1] .* new_V)

    @test total_new ≈ total_old rtol = 1.0e-12
end

@testset "V&V topoChanger: split of varying field preserves Σ φ·V" begin
    volumes = [1.0, 2.0, 3.0]
    fields = [[1.0, 2.0, 3.0]]
    total_old = sum(fields[1] .* volumes)

    split = CellSplit{Float64}(2, [4, 5, 6], [0.3, 0.3, 0.4])
    plan = TopoChanger{Float64}([split], CellMerge{Float64}[])

    new_V, new_fields = apply_topo_change!(volumes, fields, plan)

    total_new = sum(new_fields[1] .* new_V)
    @test total_new ≈ total_old rtol = 1.0e-12
    @test sum(new_V) ≈ sum(volumes) rtol = 1.0e-12
end

@testset "V&V topoChanger: merge preserves total volume" begin
    volumes = [1.0, 2.0, 3.0, 4.0]
    fields = [[10.0, 20.0, 30.0, 40.0]]

    merge_plan = CellMerge{Float64}([2, 4], 5)  # merge cells 2 and 4 → child index is positional
    plan = TopoChanger{Float64}(CellSplit{Float64}[], [merge_plan])

    new_V, new_fields = apply_topo_change!(volumes, fields, plan)

    # After merge: cells [1, 3] survive, plus one merged cell with V = 2 + 4 = 6
    @test sum(new_V) ≈ sum(volumes) rtol = 1.0e-12
    @test length(new_V) == 3
end

@testset "V&V topoChanger: merge preserves Σ φ·V (conservative mean)" begin
    volumes = [1.0, 2.0, 3.0, 4.0]
    fields = [[1.0, 2.0, 3.0, 4.0]]
    total_old = sum(fields[1] .* volumes)

    merge_plan = CellMerge{Float64}([2, 4], 5)
    plan = TopoChanger{Float64}(CellSplit{Float64}[], [merge_plan])

    new_V, new_fields = apply_topo_change!(volumes, fields, plan)
    total_new = sum(new_fields[1] .* new_V)

    @test total_new ≈ total_old rtol = 1.0e-12
end

@testset "V&V topoChanger: merge field == V-weighted mean of parents" begin
    volumes = [1.0, 3.0, 1.0]
    fields = [[0.0, 4.0, 0.0]]
    merge_plan = CellMerge{Float64}([1, 2], 4)  # merge cells 1 and 2
    plan = TopoChanger{Float64}(CellSplit{Float64}[], [merge_plan])

    new_V, new_fields = apply_topo_change!(volumes, fields, plan)

    # Expected merged field = (0·1 + 4·3) / (1+3) = 12/4 = 3
    # The merged cell appears last in the new layout.
    @test new_V[end] ≈ 4.0 rtol = 1.0e-14
    @test new_fields[1][end] ≈ 3.0 rtol = 1.0e-14
end

@testset "V&V topoChanger: total_volume / total_phi_V helpers" begin
    volumes = [1.0, 2.0, 3.0]
    phi = [10.0, 20.0, 30.0]
    @test total_volume(volumes) ≈ 6.0 rtol = 1.0e-14
    @test total_phi_V(phi, volumes) ≈ 10 + 40 + 90 rtol = 1.0e-14
    @test_throws ErrorException total_phi_V([1.0, 2.0], volumes)
end

@testset "V&V topoChanger: constructor validation" begin
    # Fractions must sum to 1
    @test_throws ErrorException CellSplit{Float64}(1, [2, 3], [0.5, 0.6])
    # Must have ≥ 2 children
    @test_throws ErrorException CellSplit{Float64}(1, [2], [1.0])
    # Matching lengths required
    @test_throws ErrorException CellSplit{Float64}(1, [2, 3, 4], [0.5, 0.5])
    # Negative fraction
    @test_throws ErrorException CellSplit{Float64}(1, [2, 3], [-0.1, 1.1])
    # Merge needs ≥ 2 parents
    @test_throws ErrorException CellMerge{Float64}([1], 2)
end

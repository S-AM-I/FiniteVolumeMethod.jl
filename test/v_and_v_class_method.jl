# test/v_and_v_class_method.jl — V&V for the sectional / class method
#
# Verifies:
# 1. Total volume Σ n_i V_i is conserved under aggregation (Hounslow
#    volume-splitting) to rtol 1e-12.
# 2. A pure-monodisperse initial state (all mass in one bin) stays
#    monodisperse under zero kernel.
# 3. Aggregation shifts mass from small bins toward larger bins
#    (net monotone shift in the first moment ⟨V⟩ = m_3 · π/6).
# 4. Breakage reverses: shifts from large bins toward small bins.
# 5. Zero kernel / zero selection ⇒ zero source.

using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "src", "population_balance", "types.jl"))
include(joinpath(@__DIR__, "..", "src", "population_balance", "class_method.jl"))

@testset "V&V: ClassMethod — construction" begin
    cm = ClassMethod(5, 1.0e-3, 1.0e-1; spacing = :geometric)
    @test cm.N_class == 5
    @test length(cm.L_edges) == 6
    @test length(cm.L_centers) == 5
    @test length(cm.V_centers) == 5
    @test issorted(cm.L_edges)
    @test all(cm.V_centers .> 0)

    cm2 = ClassMethod(4, 0.1, 0.5; spacing = :linear)
    @test isapprox(cm2.L_edges[end] - cm2.L_edges[1], 0.4; rtol = 1.0e-14)
end

@testset "V&V: ClassMethod — zero kernel ⇒ zero source" begin
    cm = ClassMethod(4, 0.1, 1.0; spacing = :geometric)
    n = [10.0, 5.0, 2.0, 1.0]
    dn = similar(n)
    aggregate_classes!(dn, n, cm, (_, _) -> 0.0)
    @test all(isapprox.(dn, 0.0; atol = 1.0e-14))
end

@testset "V&V: ClassMethod — monodisperse stays monodisperse under zero kernel" begin
    cm = ClassMethod(5, 0.1, 1.0; spacing = :geometric)
    n = zeros(5)
    n[2] = 7.0
    dn = similar(n)
    aggregate_classes!(dn, n, cm, (_, _) -> 0.0)
    @test all(isapprox.(dn, 0.0; atol = 1.0e-14))
end

@testset "V&V: ClassMethod — aggregation conserves total volume" begin
    # Wide geometric-volume grid so every V_new = V_i + V_j lands
    # strictly inside the grid (no overflow into the top bin).
    # Hounslow volume-splitting conserves the total volume exactly
    # per birth event, so Σ dn_i · V_i ≈ 0 to machine precision.
    N = 12
    L_edges = [(2.0^((k - 1) / 3)) for k in 1:(N + 1)]
    cm = ClassMethod{Float64}(N, L_edges)
    # Populate only the small-L end so aggregation output stays inside.
    n = zeros(N)
    n[1] = 5.0
    n[2] = 3.0
    n[3] = 2.0
    n[4] = 1.0
    dn = similar(n)
    aggregate_classes!(dn, n, cm, (_, _) -> 1.0)

    V_rate = sum(dn[i] * cm.V_centers[i] for i in 1:N)
    total_V = class_total_volume(n, cm)
    @test isapprox(V_rate, 0.0; atol = 1.0e-12 * total_V)
end

@testset "V&V: ClassMethod — aggregation shifts mass toward larger bins" begin
    cm = ClassMethod(6, 0.1, 1.0; spacing = :geometric)
    n = [10.0, 8.0, 5.0, 3.0, 1.0, 0.5]
    dn = similar(n)
    aggregate_classes!(dn, n, cm, (_, _) -> 1.0)

    # Number decreases (total count m_0 decreases under aggregation).
    @test sum(dn) < 0

    # The smallest bin must lose number (source of aggregation).
    @test dn[1] < 0

    # The mean size ⟨V⟩ must increase: d/dt (Σ n_i V_i / Σ n_i) > 0.
    # Since Σ n_i V_i is ≈ conserved but Σ n_i decreases, mean volume
    # strictly increases ⇒ d(mean V)/dt > 0.
    V_rate = sum(dn[i] * cm.V_centers[i] for i in 1:(cm.N_class))
    N_rate = sum(dn)
    N_total = sum(n)
    V_total = class_total_volume(n, cm)
    mean_V_rate = (V_rate * N_total - V_total * N_rate) / (N_total^2)
    @test mean_V_rate > 0
end

@testset "V&V: ClassMethod — breakage reverses aggregation direction" begin
    cm = ClassMethod(6, 0.1, 1.0; spacing = :geometric)
    # Initial condition concentrated in larger bins.
    n = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    dn = similar(n)

    # Rate proportional to L^2 (larger particles break faster).
    Kb(L) = L^2
    # Uniform binary split into the bin one below the parent (simple
    # model that produces a clear "shift-to-small" signal).
    function daughter(L_parent, L_child)
        # Parent of length L_parent breaks into 2 children, each
        # equally likely to land in the bin directly below the parent.
        # Find parent index, return 2 if child is parent_idx - 1 else 0.
        idx_parent = findfirst(x -> isapprox(x, L_parent), cm.L_centers)
        idx_child = findfirst(x -> isapprox(x, L_child), cm.L_centers)
        (idx_parent === nothing || idx_child === nothing) && return 0.0
        return idx_child == idx_parent - 1 ? 2.0 : 0.0
    end
    breakage_classes!(dn, n, cm, Kb, daughter)

    # Breakage increases total number (m_0): one parent → two children.
    @test sum(dn) > 0

    # Largest bin loses number (source of breakage).
    @test dn[end] < 0

    # Bin one below the largest gains number.
    @test dn[end - 1] > 0
end

@testset "V&V: ClassMethod — class_moments and class_total_volume" begin
    cm = ClassMethod(4, 0.1, 1.0; spacing = :geometric)
    n = [1.0, 2.0, 3.0, 4.0]
    @test class_moments(n, cm, 0) == sum(n)
    @test isapprox(
        class_moments(n, cm, 1),
        sum(n[i] * cm.L_centers[i] for i in 1:4); rtol = 1.0e-14
    )
    @test isapprox(
        class_total_volume(n, cm),
        sum(n[i] * cm.V_centers[i] for i in 1:4); rtol = 1.0e-14
    )
end

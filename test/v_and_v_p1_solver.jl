# test/v_and_v_p1_solver.jl — P1 radiation solver invariants V&V (v3.92)

using FiniteVolumeMethod
using LinearSolve
using Test

include("TestHelpers.jl")

const _cell_absorption = FiniteVolumeMethod._cell_absorption

@testset "V&V: STEFAN_BOLTZMANN constant value" begin
    # SI value 5.670374419e-8 W·m⁻²·K⁻⁴ (CODATA 2018).
    @test STEFAN_BOLTZMANN == 5.670374419e-8
    @test STEFAN_BOLTZMANN > 0.0
    @test STEFAN_BOLTZMANN < 1.0
end

@testset "V&V: _cell_absorption — scalar dispatch" begin
    # Scalar absorption returns the same value for every cell index.
    @test _cell_absorption(0.1, 1) == 0.1
    @test _cell_absorption(0.1, 42) == 0.1
    @test _cell_absorption(0.1, 10_000) == 0.1
    @test _cell_absorption(10.0, 1) == 10.0
    @test _cell_absorption(1.0e-5, 7) == 1.0e-5
end

@testset "V&V: _cell_absorption — vector dispatch" begin
    # Vector absorption indexes into the per-cell array.
    a = [0.1, 0.2, 0.3, 0.4, 0.5]
    for (idx, expected) in enumerate(a)
        @test _cell_absorption(a, idx) == expected
    end
end

@testset "V&V: P1 — isothermal cold medium with cold walls ⇒ G ≡ 0" begin
    # If T_field = 0 everywhere and the wall BC fixes G = 0, the only
    # source in -div(Gamma grad G) + a·G = 4·a·σ·T⁴ is zero. Solution
    # must be G ≡ 0 across the domain.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    rad_model = P1Model(; a = 1.0)
    T_field = CollocatedScalarField(:T, mesh; value = 0.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(0.0),
        :right => ParabolicDirichlet(0.0),
        :bottom => ParabolicDirichlet(0.0),
        :top => ParabolicDirichlet(0.0),
    )
    G = solve_p1_radiation(
        rad_model, T_field, mesh, bcs; linear_solver = LUFactorization(),
    )
    for c in 1:nc
        @test abs(G.internal[c]) < 1.0e-10
    end
end

@testset "V&V: P1 — isothermal hot medium with 4σT⁴ walls ⇒ G ≈ 4σT⁴" begin
    # Radiative equilibrium: if T is uniform T₀ and every wall is held at
    # G = 4σT₀⁴, the interior must settle to G ≡ 4σT₀⁴ (no net emission
    # or absorption). This is the strongest invariant check on P1.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    T0 = 500.0
    G_eq = 4.0 * STEFAN_BOLTZMANN * T0^4
    rad_model = P1Model(; a = 1.0)
    T_field = CollocatedScalarField(:T, mesh; value = T0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(G_eq),
        :right => ParabolicDirichlet(G_eq),
        :bottom => ParabolicDirichlet(G_eq),
        :top => ParabolicDirichlet(G_eq),
    )
    G = solve_p1_radiation(
        rad_model, T_field, mesh, bcs; linear_solver = LUFactorization(),
    )
    for c in 1:nc
        @test isapprox(G.internal[c], G_eq; rtol = 1.0e-2)
    end
end

@testset "V&V: P1 — G is non-negative under positive source" begin
    # Solver clamps G to zero via `max(sol.u[c], zero(T))` — verify the
    # invariant holds under a non-trivial T field (hot center).
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    rad_model = P1Model(; a = 1.0)
    T_field = CollocatedScalarField(:T, mesh; value = 0.0)
    # Hot blob in the center.
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.4 < x < 0.6 && 0.4 < y < 0.6
            T_field.internal[c] = 800.0
        end
    end
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(0.0),
        :right => ParabolicDirichlet(0.0),
        :bottom => ParabolicDirichlet(0.0),
        :top => ParabolicDirichlet(0.0),
    )
    G = solve_p1_radiation(
        rad_model, T_field, mesh, bcs; linear_solver = LUFactorization(),
    )
    for c in 1:nc
        @test G.internal[c] >= 0.0
    end
    # Interior G should be positive where the hot blob is.
    interior_positive = false
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.4 < x < 0.6 && 0.4 < y < 0.6
            if G.internal[c] > 0.0
                interior_positive = true
            end
        end
    end
    @test interior_positive
end

@testset "V&V: P1 — T⁴ scaling of G at fixed walls and absorption" begin
    # Doubling T everywhere (and the matched wall BC) should give 16× G
    # in the equilibrium interior (Stefan–Boltzmann law).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    T0 = 300.0
    T1 = 600.0   # factor 2 ⇒ 16× G
    G_eq0 = 4.0 * STEFAN_BOLTZMANN * T0^4
    G_eq1 = 4.0 * STEFAN_BOLTZMANN * T1^4
    rad_model = P1Model(; a = 1.0)
    bcs0 = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(G_eq0),
        :right => ParabolicDirichlet(G_eq0),
        :bottom => ParabolicDirichlet(G_eq0),
        :top => ParabolicDirichlet(G_eq0),
    )
    bcs1 = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(G_eq1),
        :right => ParabolicDirichlet(G_eq1),
        :bottom => ParabolicDirichlet(G_eq1),
        :top => ParabolicDirichlet(G_eq1),
    )
    Tf0 = CollocatedScalarField(:T, mesh; value = T0)
    Tf1 = CollocatedScalarField(:T, mesh; value = T1)
    G0 = solve_p1_radiation(rad_model, Tf0, mesh, bcs0; linear_solver = LUFactorization())
    G1 = solve_p1_radiation(rad_model, Tf1, mesh, bcs1; linear_solver = LUFactorization())
    # Interior cells (away from walls) should satisfy G1/G0 ≈ 16.
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.25 < x < 0.75 && 0.25 < y < 0.75
            @test isapprox(G1.internal[c] / G0.internal[c], 16.0; rtol = 5.0e-2)
        end
    end
end

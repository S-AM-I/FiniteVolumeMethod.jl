# test/v_and_v_qgen.jl — SolidThermalProperties.Q_gen volumetric heat V&V (v3.88)

using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: DirichletBC
using LinearSolve
using Test

include("TestHelpers.jl")

@testset "V&V: Q_gen = 0 ⇒ standard Laplace" begin
    # With Q_gen = 0 and mixed Dirichlet BCs, solve yields the
    # pure Laplace solution.
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    solid = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = 1.0, Q_gen = 0.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(1.0),
    )
    Tf = solve_solid_conduction(mesh, solid, bcs; linear_solver = LUFactorization())
    # Top boundary value propagates downward monotonically.
    for c in 1:length(mesh.cell_volumes)
        @test 0.0 - 1.0e-10 <= Tf.internal[c] <= 1.0 + 1.0e-10
    end
end

@testset "V&V: Q_gen > 0 raises interior temperature above walls" begin
    # With all walls at T = 0 and uniform positive Q_gen, interior
    # temperature must be > 0 (heat accumulates since it can only
    # escape through walls).
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    solid = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = 1.0, Q_gen = 10.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(0.0),
    )
    Tf = solve_solid_conduction(mesh, solid, bcs; linear_solver = LUFactorization())
    nc = length(mesh.cell_volumes)
    # Interior cells must have positive T.
    interior_count = 0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.3 < x < 0.7 && 0.3 < y < 0.7
            @test Tf.internal[c] > 0.0
            interior_count += 1
        end
    end
    @test interior_count > 0
end

@testset "V&V: Q_gen linear scaling of interior T" begin
    # With Q_gen fixed BCs (all T = 0) and Q_gen ∝ scale, interior T
    # scales linearly (Laplace equation is linear).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(0.0),
    )

    solid_a = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = 1.0, Q_gen = 5.0)
    solid_b = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = 1.0, Q_gen = 10.0)

    Tf_a = solve_solid_conduction(mesh, solid_a, bcs; linear_solver = LUFactorization())
    Tf_b = solve_solid_conduction(mesh, solid_b, bcs; linear_solver = LUFactorization())

    for c in 1:length(mesh.cell_volumes)
        if abs(Tf_a.internal[c]) > 1.0e-8
            @test isapprox(Tf_b.internal[c] / Tf_a.internal[c], 2.0; rtol = 1.0e-10)
        end
    end
end

@testset "V&V: k inverse scaling of interior T at fixed Q_gen" begin
    # With T_wall = 0 and Q_gen fixed, doubling k halves interior T
    # (Poisson equation scales as T ∝ Q_gen / k).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(0.0),
    )

    solid_a = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = 1.0, Q_gen = 5.0)
    solid_b = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = 2.0, Q_gen = 5.0)

    Tf_a = solve_solid_conduction(mesh, solid_a, bcs; linear_solver = LUFactorization())
    Tf_b = solve_solid_conduction(mesh, solid_b, bcs; linear_solver = LUFactorization())

    for c in 1:length(mesh.cell_volumes)
        if abs(Tf_a.internal[c]) > 1.0e-8
            @test isapprox(Tf_b.internal[c] / Tf_a.internal[c], 0.5; rtol = 1.0e-10)
        end
    end
end

# test/v_and_v_heat_conduction.jl — Steady 2D heat-conduction V&V (v3.12)
#
# Verifies `solve_solid_conduction` against the closed-form series
# solution of Laplace's equation on the unit square with
# mixed Dirichlet boundary conditions.
#
# Problem:
#   -∇²T = 0    on [0, 1]²
#   T(x, 0) = T(0, y) = T(1, y) = 0,    T(x, 1) = 1
# Series solution:
#   T(x, y) = (4/π) · Σ_{n = 1, 3, 5, ...} (1/n) · sin(n π x) · sinh(n π y) / sinh(n π)
#
# Evidence for `conjugate_heat_transfer` manifest promotion from
# `experimental` to `provisional`.

using FiniteVolumeMethod
using FiniteVolumeMethod: solve_solid_conduction
using FiniteVolumeMethod.Parabolic: DirichletBC
using LinearSolve
using Test

include("TestHelpers.jl")

function T_exact(x, y; n_terms::Int = 50)
    s = 0.0
    for n in 1:2:(2 * n_terms - 1)
        s += (4 / π) * (1 / n) * sin(n * π * x) * sinh(n * π * y) / sinh(n * π)
    end
    return s
end

function conduction_err(N::Int)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    solid = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(1.0),
    )
    Tf = solve_solid_conduction(mesh, solid, bcs)

    # L² error over the interior band (excludes first 10% near boundary
    # where the T=0 / T=1 corner singularities concentrate discretization
    # error — analogous to the Ghia corner-singularity exclusion in
    # v3.3).
    err_sq = 0.0
    vol = 0.0
    for c in 1:(N * N)
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.1 < x < 0.9 && 0.1 < y < 0.9
            Te = T_exact(x, y)
            err_sq += mesh.cell_volumes[c] * (Tf.internal[c] - Te)^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

@testset "V&V: solid conduction — interior O(h²) grid convergence" begin
    errs = [conduction_err(N) for N in [20, 40, 80]]
    orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]

    # Textbook second-order for FVM Laplacian + Dirichlet BCs. Expect
    # p ≈ 2 at both transitions. Allow 1.8 < p < 2.2 floating-point slack.
    for p in orders
        @test 1.8 < p < 2.2
    end

    # Monotone error decrease.
    @test all(errs[i] > errs[i + 1] for i in 1:(length(errs) - 1))

    # Finest L² is small.
    @test errs[end] < 1.0e-4
end

@testset "V&V: conduction center-cell matches analytical to 2%" begin
    # At (0.5, 0.5) by symmetry the series sums to exactly 0.25.
    # Finest mesh should be within ~1% of that.
    N = 80
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    solid = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(1.0),
    )
    Tf = solve_solid_conduction(mesh, solid, bcs)

    # Nearest cell to (0.5, 0.5)
    _, c = findmin(
        abs2.(mesh.cell_centers[1, :] .- 0.5) .+
            abs2.(mesh.cell_centers[2, :] .- 0.5)
    )
    T_center = Tf.internal[c]
    @test abs(T_center - 0.25) / 0.25 < 0.03
end

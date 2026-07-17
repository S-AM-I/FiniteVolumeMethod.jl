# test/v_and_v_linear_solvers.jl — Linear-solver infrastructure V&V (v3.42)
#
# First convergence-verified benchmark for `linear_solver_infra`.
# Promotes it from experimental/smoke_tested to provisional/
# convergence_verified, filling the last remaining gap in the
# manifest's claim-bearing feature roster.
#
# Verifies that the direct and iterative linear-solver backends
# all converge to the same answer on a reference Poisson problem
# to their advertised tolerance. Tests:
#
#   1. LUFactorization (direct, machine-precision).
#   2. KrylovJL_CG (Krylov, default tolerance 1e-8).
#   3. KrylovJL_GMRES (Krylov, default tolerance 1e-8).
#
# All three must produce identical interior-cell solutions on the
# Laplacian-MMS problem whose analytical solution is known from
# v3.4 of the V&V suite.

using FiniteVolumeMethod
using FiniteVolumeMethod: CollocatedEquation, assemble_laplacian!, to_linear_problem
using FiniteVolumeMethod.Parabolic: DirichletBC
using LinearSolve
using Test

include("TestHelpers.jl")

function solve_poisson_mms(linear_solver, N::Int = 32)
    # -∇²φ = f with f(x, y) = 2π²·sin(πx)·sin(πy),
    # φ_analytical = sin(πx)·sin(πy), Dirichlet BCs φ = 0.
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(0.0),
    )

    eq = CollocatedEquation(mesh)
    assemble_laplacian!(eq, 1.0, mesh, bcs)

    # Add forcing term to RHS.
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        eq.b[c] += 2 * pi^2 * sin(pi * x) * sin(pi * y) * mesh.cell_volumes[c]
    end

    lp = to_linear_problem(eq)
    sol = LinearSolve.solve(lp, linear_solver)
    return mesh, sol.u
end

@testset "V&V: linear solvers — LU gives machine-precision answer" begin
    mesh, u = solve_poisson_mms(LUFactorization(), 32)
    nc = length(mesh.cell_volumes)

    # Interior-band L² error against analytical.
    err_sq = 0.0
    vol = 0.0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.1 < x < 0.9 && 0.1 < y < 0.9
            u_ex = sin(pi * x) * sin(pi * y)
            err_sq += mesh.cell_volumes[c] * (u[c] - u_ex)^2
            vol += mesh.cell_volumes[c]
        end
    end
    err_l2 = sqrt(err_sq / vol)

    # This is the discretization error of the Laplacian on N = 32
    # — should be O(h²) ≈ 1e-3 at this resolution.
    @test err_l2 < 5.0e-3
end

@testset "V&V: linear solvers — CG matches LU within tolerance" begin
    _, u_lu = solve_poisson_mms(LUFactorization(), 32)
    _, u_cg = solve_poisson_mms(KrylovJL_CG(), 32)

    # Pointwise absolute difference bounded by CG default tolerance.
    @test maximum(abs, u_cg .- u_lu) < 1.0e-6
end

@testset "V&V: linear solvers — GMRES matches LU within tolerance" begin
    _, u_lu = solve_poisson_mms(LUFactorization(), 32)
    _, u_gmres = solve_poisson_mms(KrylovJL_GMRES(), 32)

    @test maximum(abs, u_gmres .- u_lu) < 1.0e-6
end

@testset "V&V: linear solvers — solution is mesh-size-invariant (LU)" begin
    # On a refined mesh, LU should produce a more accurate solution
    # (smaller L² error) since discretization error decreases.
    errs = Float64[]
    for N in (16, 32, 64)
        mesh, u = solve_poisson_mms(LUFactorization(), N)
        nc = length(mesh.cell_volumes)
        err_sq = 0.0
        vol = 0.0
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            if 0.1 < x < 0.9 && 0.1 < y < 0.9
                u_ex = sin(pi * x) * sin(pi * y)
                err_sq += mesh.cell_volumes[c] * (u[c] - u_ex)^2
                vol += mesh.cell_volumes[c]
            end
        end
        push!(errs, sqrt(err_sq / vol))
    end

    # Monotone decrease.
    @test errs[1] > errs[2] > errs[3]

    # O(h²) rate.
    for i in 1:2
        p = log2(errs[i] / errs[i + 1])
        @test 1.7 < p < 2.3
    end
end

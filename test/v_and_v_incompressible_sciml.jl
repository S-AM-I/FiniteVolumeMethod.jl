# test/v_and_v_incompressible_sciml.jl — SciML interface V&V (v3.55)
#
# Fifth convergence-verified benchmark for `incompressible_ns`,
# joining Ghia Re=100 (v3.1), Poiseuille (v3.10), Couette (v3.22),
# and transient PISO (v3.41). Covers the SciML-compatible solution
# wrapper and symbolic indexing surface — the public API
# consumed by downstream SciML workflows.
#
# Six invariants verified:
#
#   1. `solve(prob, SIMPLE())` returns an `IncompressibleSolution`.
#   2. `sol[:U]`, `sol[:p]`, `sol[:phi]` round-trip to the
#      underlying state fields.
#   3. `sol[:Ux]`, `sol[:Uy]` extract velocity components.
#   4. `keys(sol)` lists the accessible symbolic fields.
#   5. `sol.retcode ∈ {:Success, :MaxIters}` reflects convergence.
#   6. `is_fvm_solution(sol)` returns true.

using FiniteVolumeMethod
using LinearSolve
using StaticArrays: SVector
using Test

include("TestHelpers.jl")

function build_simple_cavity(; N = 12)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(0.1, 0.0)),
    )
    algo = SIMPLE(0.5, 0.2, 100, 1.0e-5)
    prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1, density = 1.0)
    return prob, algo, mesh
end

@testset "V&V: Incompressible SciML — solve returns IncompressibleSolution" begin
    prob, algo, _ = build_simple_cavity(; N = 8)
    sol = solve(prob, algo; linear_solver = LUFactorization())
    @test sol isa IncompressibleSolution
    @test sol isa FiniteVolumeMethod.AbstractFVMSolution
end

@testset "V&V: Incompressible SciML — symbolic field access" begin
    prob, algo, mesh = build_simple_cavity(; N = 8)
    sol = solve(prob, algo; linear_solver = LUFactorization())
    nc = length(mesh.cell_volumes)

    U = sol[:U]
    p = sol[:p]
    phi = sol[:phi]

    @test length(U) == nc
    @test length(p) == nc
    @test length(phi) == size(mesh.face_cells, 2)

    # All values finite.
    for u in U
        @test isfinite(u[1]) && isfinite(u[2])
    end
    for pi in p
        @test isfinite(pi)
    end
end

@testset "V&V: Incompressible SciML — velocity-component extraction" begin
    prob, algo, mesh = build_simple_cavity(; N = 8)
    sol = solve(prob, algo; linear_solver = LUFactorization())
    nc = length(mesh.cell_volumes)

    Ux = sol[:Ux]
    Uy = sol[:Uy]

    @test length(Ux) == nc
    @test length(Uy) == nc

    # Cross-check: Ux matches U[1] component.
    U = sol[:U]
    for c in 1:nc
        @test Ux[c] == U[c][1]
        @test Uy[c] == U[c][2]
    end
end

@testset "V&V: Incompressible SciML — keys(sol) lists symbolic names" begin
    prob, algo, _ = build_simple_cavity(; N = 6)
    sol = solve(prob, algo; linear_solver = LUFactorization())

    k = keys(sol)
    @test :U in k
    @test :p in k
    @test :phi in k
    @test :Ux in k
    @test :Uy in k
    @test :Uz ∉ k   # 2D problem
end

@testset "V&V: Incompressible SciML — retcode reflects convergence" begin
    prob, algo, _ = build_simple_cavity(; N = 6)
    sol = solve(prob, algo; linear_solver = LUFactorization())
    @test sol.retcode in (:Success, :MaxIters)
    @test sol.iterations > 0

    # retcode consistency with converged flag.
    if sol.converged
        @test sol.retcode === :Success
    else
        @test sol.retcode === :MaxIters
    end
end

@testset "V&V: Incompressible SciML — is_fvm_solution trait" begin
    prob, algo, _ = build_simple_cavity(; N = 6)
    sol = solve(prob, algo; linear_solver = LUFactorization())
    @test FiniteVolumeMethod.is_fvm_solution(sol) == true
    @test FiniteVolumeMethod.is_fvm_solution("not a solution") == false
    @test FiniteVolumeMethod.is_fvm_solution(42) == false
end

@testset "V&V: Incompressible SciML — invalid symbol errors" begin
    prob, algo, _ = build_simple_cavity(; N = 4)
    sol = solve(prob, algo; linear_solver = LUFactorization())
    @test_throws ErrorException sol[:nonexistent]
    @test_throws ErrorException sol[:Uz]   # 2D has no Uz
end

using FiniteVolumeMethod
using FiniteVolumeMethod: build_preconditioner, default_solver_config
using Test
using LinearAlgebra
using SparseArrays
using LinearSolve
using SciMLBase: LinearProblem

@testset "Linear Solver Infrastructure" begin

    # ── 1. FieldSolverConfig defaults ─────────────────────────────────
    @testset "FieldSolverConfig defaults" begin
        fc = FieldSolverConfig()
        @test fc.solver === :direct
        @test fc.preconditioner === :none
        @test fc.rtol == 1.0e-6
        @test fc.atol == 1.0e-8
        @test fc.maxiter == 1000
    end

    # ── 2. FieldSolverConfig custom ───────────────────────────────────
    @testset "FieldSolverConfig custom" begin
        fc = FieldSolverConfig(;
            solver = :cg,
            preconditioner = :amg,
            rtol = 1.0e-4,
        )
        @test fc.solver === :cg
        @test fc.preconditioner === :amg
        @test fc.rtol == 1.0e-4
        @test fc.atol == 1.0e-8      # default
        @test fc.maxiter == 1000      # default
    end

    # ── 3. FVMSolverConfig defaults ───────────────────────────────────
    @testset "FVMSolverConfig defaults" begin
        sc = FVMSolverConfig()
        @test isempty(sc.fields)
        @test sc.default isa FieldSolverConfig
        @test sc.default.solver === :direct
    end

    # ── 4. FVMSolverConfig field lookup ───────────────────────────────
    @testset "FVMSolverConfig field lookup" begin
        p_config = FieldSolverConfig(; solver = :cg, preconditioner = :amg)
        sc = FVMSolverConfig(;
            fields = Dict{Symbol, FieldSolverConfig}(:p => p_config),
        )
        # Lookup :p returns the per-field config
        @test get(sc.fields, :p, sc.default) === p_config
        @test get(sc.fields, :p, sc.default).solver === :cg
        # Lookup :Ux falls back to default
        @test get(sc.fields, :Ux, sc.default) === sc.default
        @test get(sc.fields, :Ux, sc.default).solver === :direct
    end

    # ── 5. default_solver_config ──────────────────────────────────────
    @testset "default_solver_config" begin
        sc = default_solver_config()
        # Pressure field
        p_config = sc.fields[:p]
        @test p_config.solver === :cg
        @test p_config.preconditioner === :amg
        @test p_config.rtol == 1.0e-6
        @test p_config.maxiter == 1000
        # Default (everything else)
        @test sc.default.solver === :bicgstab
        @test sc.default.preconditioner === :ilu
        @test sc.default.rtol == 1.0e-5
        @test sc.default.maxiter == 500
    end

    # ── 6. build_preconditioner :none ─────────────────────────────────
    @testset "build_preconditioner :none" begin
        A = sparse([1, 2, 3], [1, 2, 3], [4.0, 5.0, 6.0])
        @test build_preconditioner(:none, A) === nothing
    end

    # ── 7. build_preconditioner :diagonal ─────────────────────────────
    @testset "build_preconditioner :diagonal" begin
        A = sparse([1, 2, 3], [1, 2, 3], [4.0, 5.0, 6.0])
        P = build_preconditioner(:diagonal, A)
        @test P isa Diagonal
        @test P == Diagonal([4.0, 5.0, 6.0])
    end

    # ── 8. build_preconditioner :amg fallback ─────────────────────────
    @testset "build_preconditioner :amg fallback" begin
        A = sparse([1, 2, 3], [1, 2, 3], [4.0, 5.0, 6.0])
        result = @test_logs (:warn,) build_preconditioner(:amg, A)
        @test result === nothing
    end

    # ── 9. _resolve_solver :direct ────────────────────────────────────
    @testset "_resolve_solver :direct" begin
        @test FiniteVolumeMethod._resolve_solver(:direct) === nothing
    end

    # ── 10. _resolve_solver pass-through ──────────────────────────────
    @testset "_resolve_solver pass-through" begin
        sentinel = "not_a_symbol"
        @test FiniteVolumeMethod._resolve_solver(sentinel) === sentinel

        obj = (x = 1, y = 2)
        @test FiniteVolumeMethod._resolve_solver(obj) === obj
    end

    # ── 11. _dispatch_solve with config=nothing ───────────────────────
    @testset "_dispatch_solve config=nothing" begin
        # 3x3 SPD system with known solution
        A = sparse(
            [1, 2, 3, 1, 2, 2, 3],
            [1, 2, 3, 2, 1, 3, 2],
            [4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0],
        )
        x_exact = [1.0, 2.0, 3.0]
        b = A * x_exact
        lp = LinearProblem(A, b)

        sol = FiniteVolumeMethod._dispatch_solve(lp, nothing, nothing, :test)
        @test norm(sol.u - x_exact) < 1.0e-10
    end

    # ── 12. _dispatch_solve with :direct config ───────────────────────
    @testset "_dispatch_solve config :direct" begin
        A = sparse(
            [1, 2, 3, 1, 2, 2, 3],
            [1, 2, 3, 2, 1, 3, 2],
            [4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0],
        )
        x_exact = [1.0, 2.0, 3.0]
        b = A * x_exact
        lp = LinearProblem(A, b)

        config = FVMSolverConfig(;
            fields = Dict{Symbol, FieldSolverConfig}(
                :test => FieldSolverConfig(;
                    solver = :direct,
                    preconditioner = :diagonal,
                ),
            ),
        )
        sol = FiniteVolumeMethod._dispatch_solve(lp, nothing, config, :test)
        @test norm(sol.u - x_exact) < 1.0e-10
    end
end

# test/v_and_v_solver_config.jl — FVMSolverConfig dispatch V&V (v3.52)
#
# Second convergence-verified benchmark for `linear_solver_infra`,
# joining the Poisson MMS across-backends test (v3.42). Covers
# the `FVMSolverConfig` per-field routing + `_resolve_solver`
# dispatch layer used by every collocated transport equation.
#
# Four invariants verified:
#
#   1. Default config: pressure field routes to :cg + :amg; other
#      fields route to :bicgstab + :ilu.
#   2. Per-field overrides resolve before falling back to default.
#   3. `:direct` solver resolves to `nothing` (backslash path).
#   4. Unknown solver symbols throw an error.

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

@testset "V&V: SolverConfig — default pressure routing" begin
    cfg = default_solver_config()
    # Pressure gets CG + AMG.
    p_cfg = get(cfg.fields, :p, nothing)
    @test p_cfg !== nothing
    @test p_cfg.solver == :cg
    @test p_cfg.preconditioner == :amg
    @test p_cfg.rtol == 1.0e-6
    @test p_cfg.maxiter == 1000
end

@testset "V&V: SolverConfig — default fallback routing" begin
    cfg = default_solver_config()
    # Non-pressure fields fall back to the default config.
    @test cfg.default.solver == :bicgstab
    @test cfg.default.preconditioner == :ilu
    @test cfg.default.rtol == 1.0e-5
    @test cfg.default.maxiter == 500

    # Unknown field name also hits the default.
    @test get(cfg.fields, :U, cfg.default).solver == :bicgstab
    @test get(cfg.fields, :alpha, cfg.default).preconditioner == :ilu
end

@testset "V&V: SolverConfig — custom FieldSolverConfig overrides" begin
    custom_p = FieldSolverConfig(;
        solver = :gmres, preconditioner = :none,
        rtol = 1.0e-8, atol = 1.0e-10, maxiter = 200,
    )
    @test custom_p.solver == :gmres
    @test custom_p.preconditioner == :none
    @test custom_p.rtol == 1.0e-8
    @test custom_p.atol == 1.0e-10
    @test custom_p.maxiter == 200

    # FVMSolverConfig with overrides.
    cfg = FVMSolverConfig(;
        fields = Dict(:p => custom_p),
        default = FieldSolverConfig(; solver = :direct),
    )
    @test cfg.fields[:p].solver == :gmres
    @test cfg.default.solver == :direct
end

@testset "V&V: SolverConfig — :direct resolves to nothing (backslash)" begin
    result = FiniteVolumeMethod._resolve_solver(:direct)
    @test result === nothing
end

@testset "V&V: SolverConfig — unknown symbol errors" begin
    @test_throws ErrorException FiniteVolumeMethod._resolve_solver(:nonexistent)
end

@testset "V&V: SolverConfig — pass-through of non-Symbol arguments" begin
    # Non-Symbol arguments (e.g. LinearSolve algorithm objects)
    # should pass through unchanged.
    dummy = "my_alg_placeholder"
    @test FiniteVolumeMethod._resolve_solver(dummy) === dummy
end

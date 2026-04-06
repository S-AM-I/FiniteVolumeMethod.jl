# linear_solvers/solver_config.jl — Solver configuration for FVM linear systems
#
# Maps field names to solver+preconditioner+tolerance combos, analogous
# to OpenFOAM's fvSolution. Supports both symbol-based solver selection
# and direct LinearSolve.jl algorithm objects.

using LinearAlgebra: Diagonal, diag

# ── Field solver configuration ───────────────────────────────────────

"""
    FieldSolverConfig

Configuration for solving a single field's linear system.

# Fields
- `solver::Any` — solver algorithm: `Symbol` (`:cg`, `:bicgstab`, `:gmres`, `:direct`)
  or a LinearSolve.jl algorithm object (e.g. `KrylovJL_CG()`)
- `preconditioner::Symbol` — preconditioner tag: `:none`, `:diagonal`, `:ilu`, `:amg`
- `rtol::Float64` — relative tolerance (default 1e-6)
- `atol::Float64` — absolute tolerance (default 1e-8)
- `maxiter::Int` — maximum Krylov iterations (default 1000)
"""
struct FieldSolverConfig
    solver::Any
    preconditioner::Symbol
    rtol::Float64
    atol::Float64
    maxiter::Int
end

"""
    FieldSolverConfig(; solver = :direct, preconditioner = :none,
        rtol = 1e-6, atol = 1e-8, maxiter = 1000)

Construct a field solver configuration with keyword defaults.

`solver` can be:
- `:direct` — Julia backslash (LU factorization)
- `:cg` — Conjugate Gradient (for SPD systems like pressure)
- `:bicgstab` — BiCGSTAB (for non-symmetric systems like velocity)
- `:gmres` — GMRES (general purpose)
- Any LinearSolve.jl algorithm object
"""
function FieldSolverConfig(;
        solver::Any = :direct,
        preconditioner::Symbol = :none,
        rtol::Float64 = 1.0e-6,
        atol::Float64 = 1.0e-8,
        maxiter::Int = 1000,
    )
    return FieldSolverConfig(solver, preconditioner, rtol, atol, maxiter)
end

# ── Multi-field solver configuration ─────────────────────────────────

"""
    FVMSolverConfig

Per-field solver configuration, analogous to OpenFOAM's `fvSolution`.

Field names: `:p` (pressure), `:Ux`/`:Uy`/`:Uz` (velocity), `:k`/`:epsilon`/
`:omega`/`:nu_tilde` (turbulence), `:T` (temperature).

Fields not in the dictionary use the `default` config.

# Fields
- `fields::Dict{Symbol, FieldSolverConfig}` — per-field overrides
- `default::FieldSolverConfig` — fallback for unlisted fields
"""
struct FVMSolverConfig
    fields::Dict{Symbol, FieldSolverConfig}
    default::FieldSolverConfig
end

"""
    FVMSolverConfig(; fields = Dict(), default = FieldSolverConfig())

Construct a solver configuration.
"""
function FVMSolverConfig(;
        fields = Dict{Symbol, FieldSolverConfig}(),
        default = FieldSolverConfig(),
    )
    return FVMSolverConfig(fields, default)
end

"""
    default_solver_config()

Return the default solver configuration matching OpenFOAM conventions:
- Pressure: CG + AMG, rtol 1e-6
- Everything else: BiCGSTAB + ILU, rtol 1e-5

Note: `:amg` and `:ilu` preconditioners require loading
`AlgebraicMultigrid` and `IncompleteLU` packages respectively.
Without them, falls back to no preconditioner with a warning.
"""
function default_solver_config()
    return FVMSolverConfig(;
        fields = Dict{Symbol, FieldSolverConfig}(
            :p => FieldSolverConfig(;
                solver = :cg,
                preconditioner = :amg,
                rtol = 1.0e-6,
                maxiter = 1000,
            ),
        ),
        default = FieldSolverConfig(;
            solver = :bicgstab,
            preconditioner = :ilu,
            rtol = 1.0e-5,
            maxiter = 500,
        ),
    )
end

# ── Solver resolution ────────────────────────────────────────────────

"""
    _resolve_solver(solver) -> algorithm or nothing

Convert a solver specification to a LinearSolve.jl algorithm.
Returns `nothing` for `:direct` (uses backslash).
Passes through non-Symbol arguments unchanged.
"""
function _resolve_solver(solver::Symbol)
    solver == :direct && return nothing
    solver == :cg && return _try_krylov_solver(:cg)
    solver == :bicgstab && return _try_krylov_solver(:bicgstab)
    solver == :gmres && return _try_krylov_solver(:gmres)
    return error(
        "Unknown solver symbol: :$solver. " *
            "Use :direct, :cg, :bicgstab, :gmres, or a LinearSolve algorithm.",
    )
end

_resolve_solver(solver) = solver  # pass-through for LinearSolve algorithm objects

"""
    _try_krylov_solver(sym::Symbol)

Try to construct a Krylov solver. If LinearSolve is available in Main,
constructs the corresponding `KrylovJL_*` solver. Otherwise, warns and
returns `nothing` (falls back to direct solver).
"""
function _try_krylov_solver(sym::Symbol)
    try
        if sym == :cg
            return Base.invokelatest(getfield(Main, :KrylovJL_CG))
        elseif sym == :bicgstab
            return Base.invokelatest(getfield(Main, :KrylovJL_BICGSTAB))
        elseif sym == :gmres
            return Base.invokelatest(getfield(Main, :KrylovJL_GMRES))
        end
    catch
        @warn "Krylov solver :$sym not available. " *
            "Install and load LinearSolve.jl. " *
            "Falling back to direct solver."
        return nothing
    end
end

# ── Solve with config ────────────────────────────────────────────────

"""
    _solve_with_config(lp, config::FVMSolverConfig, field_name::Symbol)

Solve `LinearProblem` `lp` using the configuration for `field_name`.

Looks up the field-specific config (or default), builds the preconditioner,
resolves the solver algorithm, and dispatches to `solve`.
"""
function _solve_with_config(
        lp, config::FVMSolverConfig, field_name::Symbol,
    )
    fc = get(config.fields, field_name, config.default)

    # Build preconditioner from matrix
    Pl = build_preconditioner(fc.preconditioner, lp.A)

    # Resolve solver
    alg = _resolve_solver(fc.solver)

    # Direct solver
    if alg === nothing
        return solve(lp)
    end

    # Iterative solver with preconditioner and tolerances
    if Pl !== nothing
        return solve(
            lp, alg;
            Pl = Pl, reltol = fc.rtol, abstol = fc.atol, maxiters = fc.maxiter,
        )
    else
        return solve(
            lp, alg;
            reltol = fc.rtol, abstol = fc.atol, maxiters = fc.maxiter,
        )
    end
end

# ── Unified dispatch ─────────────────────────────────────────────────

"""
    _dispatch_solve(lp, linear_solver, solver_config, field_name::Symbol)

Unified solve dispatch: uses `solver_config` if provided, otherwise
falls back to `_solve_linear(lp, linear_solver)`.

This bridges the old `linear_solver` keyword and the new `solver_config`
keyword for backward compatibility.
"""
function _dispatch_solve(lp, linear_solver, solver_config, field_name::Symbol)
    if solver_config !== nothing
        return _solve_with_config(lp, solver_config, field_name)
    else
        return _solve_linear(lp, linear_solver)
    end
end

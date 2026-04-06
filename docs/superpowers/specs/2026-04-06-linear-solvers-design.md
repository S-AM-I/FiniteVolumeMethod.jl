# Phase 5: Linear Solver Infrastructure

**Date**: 2026-04-06
**Status**: Design
**Depends on**: Phase 1 (Incompressible NS — provides `_solve_linear` dispatch point)

## Goal

Provide configurable iterative linear solvers with AMG and ILU preconditioners for the incompressible solver, replacing the default backslash (LU) that doesn't scale beyond ~1000 cells. Configuration maps field names to solver+preconditioner combos, matching OpenFOAM's `fvSolution` pattern.

## Architecture

### Two Layers

1. **Core solver config** (no new deps) — `FVMSolverConfig` type that maps field names to `(solver, preconditioner_tag, tolerances)`. A new `_solve_linear` overload dispatches per-field.

2. **Package extensions** (weak deps) — `FVMAMGExt` wraps AlgebraicMultigrid.jl, `FVMILUExt` wraps IncompleteLU.jl. Users `using AlgebraicMultigrid` to activate. Extensions register preconditioner constructors via a dispatch function.

### File Layout

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/linear_solvers/solver_config.jl` | FVMSolverConfig, FieldSolverConfig, defaults, `_solve_with_config` | ~120 |
| `src/linear_solvers/preconditioners.jl` | `build_preconditioner` dispatch + fallback stubs | ~50 |
| `ext/FVMAMGExt/FVMAMGExt.jl` | AlgebraicMultigrid.jl extension | ~30 |
| `ext/FVMILUExt/FVMILUExt.jl` | IncompleteLU.jl extension | ~30 |

Core files wired into Layer 2 (after incompressible, before turbulence — so turbulence and thermal solvers can use it). Extensions in `ext/` following existing pattern.

## Type Design

### FieldSolverConfig

```julia
struct FieldSolverConfig
    solver::Any             # LinearSolve.jl algorithm (e.g., KrylovJL_CG()) or nothing
    preconditioner::Symbol  # :none, :ilu, :amg, :diagonal
    rtol::Float64           # relative tolerance (default 1e-6)
    atol::Float64           # absolute tolerance (default 1e-8)
    maxiter::Int            # max Krylov iterations (default 1000)
end
```

Using `Symbol` for preconditioner tag rather than the actual preconditioner object because:
- The preconditioner must be constructed fresh for each matrix (different sparsity/values each iteration)
- Extensions register constructors, not instances
- Symbols are serializable and readable in configs

Convenience constructor:
```julia
FieldSolverConfig(; solver = nothing, preconditioner = :none,
    rtol = 1e-6, atol = 1e-8, maxiter = 1000)
```

### FVMSolverConfig

```julia
struct FVMSolverConfig
    fields::Dict{Symbol, FieldSolverConfig}
    default::FieldSolverConfig
end
```

Field-specific configs override the default. Field names match the solver context: `:p` for pressure, `:Ux`/`:Uy`/`:Uz` for velocity components, `:k`/`:epsilon`/`:omega`/`:nu_tilde` for turbulence, `:T` for temperature.

Convenience constructor:
```julia
FVMSolverConfig(; fields = Dict{Symbol, FieldSolverConfig}(),
    default = FieldSolverConfig())
```

### Default Configurations

```julia
function default_solver_config()
    return FVMSolverConfig(;
        fields = Dict{Symbol, FieldSolverConfig}(
            :p => FieldSolverConfig(;
                solver = KrylovJL_CG(),
                preconditioner = :amg,
                rtol = 1e-6,
                maxiter = 1000,
            ),
        ),
        default = FieldSolverConfig(;
            solver = KrylovJL_BICGSTAB(),
            preconditioner = :ilu,
            rtol = 1e-5,
            maxiter = 500,
        ),
    )
end
```

This matches OpenFOAM defaults: CG+AMG for pressure (SPD system), BiCGSTAB+ILU for everything else.

Note: `KrylovJL_CG` and `KrylovJL_BICGSTAB` come from LinearSolve.jl. Since LinearSolve.jl is not a direct dependency, these are used only when users import LinearSolve. For the core package, `default_solver_config()` requires `using LinearSolve` first. An alternative: store solver types as symbols too (`:cg`, `:bicgstab`) and resolve at solve time. **Decision: use symbols for both solver and preconditioner** to avoid the LinearSolve dependency in the core. Resolution happens in `_solve_with_config`.

Revised:
```julia
struct FieldSolverConfig
    solver::Symbol          # :cg, :bicgstab, :gmres, :direct (backslash)
    preconditioner::Symbol  # :none, :ilu, :amg, :diagonal
    rtol::Float64
    atol::Float64
    maxiter::Int
end
```

### Solver Resolution

```julia
function _resolve_solver(sym::Symbol)
    sym == :cg && return KrylovJL_CG()
    sym == :bicgstab && return KrylovJL_BICGSTAB()
    sym == :gmres && return KrylovJL_GMRES()
    sym == :direct && return nothing  # backslash
    error("Unknown solver: $sym")
end
```

`KrylovJL_CG`, `KrylovJL_BICGSTAB`, `KrylovJL_GMRES` are from SciMLBase (re-exported from LinearSolve). Actually — checking: SciMLBase exports `LinearProblem` but the Krylov solver types come from LinearSolve.jl itself. We need to handle this gracefully.

**Final decision: use Any for solver, accept both symbols and LinearSolve algorithm objects.** When a symbol is passed, we resolve it at call time. When a LinearSolve algorithm is passed directly, we use it as-is. This avoids depending on LinearSolve in the core while allowing power users to pass specific algorithms.

```julia
struct FieldSolverConfig
    solver::Any             # Symbol (:cg, :bicgstab, :gmres, :direct) or LinearSolve algorithm
    preconditioner::Symbol  # :none, :ilu, :amg, :diagonal
    rtol::Float64
    atol::Float64
    maxiter::Int
end
```

## Preconditioner Dispatch

### Core Dispatch Function

```julia
function build_preconditioner(tag::Symbol, A::SparseMatrixCSC)
    tag == :none && return nothing
    tag == :diagonal && return Diagonal(diag(A))
    # :ilu and :amg handled by extensions — if not loaded, fall back
    return _extension_preconditioner(Val(tag), A)
end
```

### Extension Registration

The base package defines:
```julia
function _extension_preconditioner(::Val{T}, A) where {T}
    @warn "Preconditioner :$T not available. Load the required package:\n" *
          "  :amg → `using AlgebraicMultigrid`\n" *
          "  :ilu → `using IncompleteLU`\n" *
          "Falling back to no preconditioner."
    return nothing
end
```

Extensions override this for their specific tag:
```julia
# In FVMAMGExt:
function FiniteVolumeMethod._extension_preconditioner(::Val{:amg}, A::SparseMatrixCSC)
    ml = AlgebraicMultigrid.ruge_stuben(A)
    return AlgebraicMultigrid.aspreconditioner(ml)
end

# In FVMILUExt:
function FiniteVolumeMethod._extension_preconditioner(::Val{:ilu}, A::SparseMatrixCSC)
    return IncompleteLU.ilu(A; τ = 0.1)
end
```

### Diagonal Preconditioner (built-in)

The `:diagonal` (Jacobi) preconditioner needs no external package:
```julia
tag == :diagonal && return Diagonal(diag(A))
```

This is a reasonable fallback when neither AMG nor ILU is available.

## Solver Dispatch

### `_solve_with_config`

```julia
function _solve_with_config(
    lp::LinearProblem, config::FVMSolverConfig, field_name::Symbol,
)
    fc = get(config.fields, field_name, config.default)
    
    # Build preconditioner
    Pl = build_preconditioner(fc.preconditioner, lp.A)
    
    # Resolve solver
    alg = _resolve_solver(fc.solver)
    
    # Solve
    if alg === nothing
        return solve(lp)  # direct/backslash
    else
        kwargs = Dict{Symbol, Any}()
        Pl !== nothing && (kwargs[:Pl] = Pl)
        fc.rtol > 0 && (kwargs[:reltol] = fc.rtol)
        fc.atol > 0 && (kwargs[:abstol] = fc.atol)
        fc.maxiter > 0 && (kwargs[:maxiters] = fc.maxiter)
        return solve(lp, alg; kwargs...)
    end
end
```

### Updated `_solve_linear`

The existing `_solve_linear(lp, linear_solver)` stays unchanged for backward compatibility. A new method handles the config:

```julia
function _solve_linear(lp, config::FVMSolverConfig, field_name::Symbol)
    return _solve_with_config(lp, config, field_name)
end
```

### Solver Wrapper Integration

All solver wrappers (`solve_simple`, `solve_simple_turbulent`, `solve_simple_thermal`, etc.) gain an optional `solver_config` keyword. When provided, `_solve_linear` calls use the config with the appropriate field name. When `nothing`, uses the existing `linear_solver` path.

The integration is done in the solver wrappers by wrapping the existing `_solve_linear` calls:

```julia
# In solve_simple:
function _dispatch_solve(lp, linear_solver, solver_config, field_name)
    if solver_config !== nothing
        return _solve_with_config(lp, solver_config, field_name)
    else
        return _solve_linear(lp, linear_solver)
    end
end
```

This is backward-compatible — existing code that passes `linear_solver` still works.

## Project.toml Changes

Add to `[weakdeps]`:
```toml
AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c"
IncompleteLU = "40713840-3770-5561-ab4c-a76e7d0d7895"
```

Add to `[extensions]`:
```toml
FVMAMGExt = "AlgebraicMultigrid"
FVMILUExt = "IncompleteLU"
```

Add to `[compat]`:
```toml
AlgebraicMultigrid = "0.6"
IncompleteLU = "0.2"
```

## Export List

```julia
# Config types
export FVMSolverConfig, FieldSolverConfig
export default_solver_config

# Preconditioner dispatch
export build_preconditioner
```

## Validation

- **Config construction**: Verify default_solver_config() creates valid config with correct field mappings.
- **Direct solver fallback**: When no extensions loaded, verify solve still works (falls back to backslash or no preconditioner).
- **Diagonal preconditioner**: Verify `:diagonal` produces correct `Diagonal(diag(A))`.
- **SIMPLE with config**: Run solve_simple on the 8x4 test mesh with a solver config using `:direct` solver and `:diagonal` preconditioner. Verify convergence behavior is similar to backslash.
- **Extension smoke test**: If AlgebraicMultigrid is in test deps, verify `_extension_preconditioner(Val(:amg), A)` returns a working preconditioner.

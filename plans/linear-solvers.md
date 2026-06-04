---
date: 2026-04-06
---

# Linear Solver Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide configurable iterative linear solvers with AMG and ILU preconditioners for the incompressible solver, replacing the default backslash that doesn't scale.

**Architecture:** Two core files in `src/linear_solvers/` (config types + preconditioner dispatch) wired into Layer 2, plus two package extensions (`FVMAMGExt`, `FVMILUExt`) in `ext/`. A `_dispatch_solve` helper bridges old `linear_solver` keyword and new `solver_config` keyword. Preconditioners use Val-dispatch so extensions can register methods without modifying core code.

**Tech Stack:** Julia, SparseArrays (SparseMatrixCSC, Diagonal, diag), LinearAlgebra, SciMLBase (LinearProblem, solve), CommonSolve. Weak deps: AlgebraicMultigrid.jl, IncompleteLU.jl.

---

## File Map

| File | Purpose | Creates/Modifies |
|------|---------|-----------------|
| `src/linear_solvers/solver_config.jl` | FieldSolverConfig, FVMSolverConfig, _resolve_solver, _solve_with_config, _dispatch_solve | Create |
| `src/linear_solvers/preconditioners.jl` | build_preconditioner, _extension_preconditioner fallback | Create |
| `ext/FVMAMGExt.jl` | AlgebraicMultigrid.jl extension | Create |
| `ext/FVMILUExt.jl` | IncompleteLU.jl extension | Create |
| `Project.toml` | Add weakdeps, extensions, compat entries | Modify |
| `src/layers/discretization_assembly_kernels.jl` | Wire linear_solvers includes | Modify |
| `src/FiniteVolumeMethod.jl` | Add exports | Modify |
| `test/linear_solvers.jl` | All tests | Create |
| `test/runtests.jl` | Register test | Modify |
| `validation/manifest.toml` | Register feature | Modify |

---

### Task 1: Create solver_config.jl — Config types and solve dispatch

**Files:**
- Create: `src/linear_solvers/solver_config.jl`

- [ ] **Step 1: Create directory and write the config file**

```bash
mkdir -p src/linear_solvers
```

Write `src/linear_solvers/solver_config.jl`:

```julia
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
        rtol::Float64 = 1e-6,
        atol::Float64 = 1e-8,
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
                rtol = 1e-6,
                maxiter = 1000,
            ),
        ),
        default = FieldSolverConfig(;
            solver = :bicgstab,
            preconditioner = :ilu,
            rtol = 1e-5,
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
    solver == :cg && return _krylov_cg()
    solver == :bicgstab && return _krylov_bicgstab()
    solver == :gmres && return _krylov_gmres()
    error("Unknown solver symbol: :$solver. Use :direct, :cg, :bicgstab, :gmres, or a LinearSolve algorithm.")
end

_resolve_solver(solver) = solver  # pass-through for LinearSolve algorithm objects

# Krylov solver constructors — these use Krylov.jl via CommonSolve
# They are lightweight wrappers that work without importing LinearSolve
function _krylov_cg()
    return _try_krylov_solver(:cg)
end

function _krylov_bicgstab()
    return _try_krylov_solver(:bicgstab)
end

function _krylov_gmres()
    return _try_krylov_solver(:gmres)
end

"""
Try to construct a Krylov solver. If LinearSolve is available, use it.
Otherwise, return nothing (falls back to direct).
"""
function _try_krylov_solver(sym::Symbol)
    # Check if LinearSolve solver types are available via SciMLBase
    try
        if sym == :cg
            return Base.invokelatest(getfield(Main, :KrylovJL_CG))
        elseif sym == :bicgstab
            return Base.invokelatest(getfield(Main, :KrylovJL_BICGSTAB))
        elseif sym == :gmres
            return Base.invokelatest(getfield(Main, :KrylovJL_GMRES))
        end
    catch
        @warn "Krylov solver :$sym not available. Install and load LinearSolve.jl. Falling back to direct solver."
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
        return solve(lp, alg; Pl = Pl, reltol = fc.rtol, abstol = fc.atol, maxiters = fc.maxiter)
    else
        return solve(lp, alg; reltol = fc.rtol, abstol = fc.atol, maxiters = fc.maxiter)
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
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/linear_solvers/solver_config.jl"); println("OK")'
```

---

### Task 2: Create preconditioners.jl — Preconditioner dispatch

**Files:**
- Create: `src/linear_solvers/preconditioners.jl`

- [ ] **Step 1: Write the preconditioner dispatch**

Write `src/linear_solvers/preconditioners.jl`:

```julia
# linear_solvers/preconditioners.jl — Preconditioner construction dispatch
#
# Builds preconditioners from a symbol tag and a sparse matrix.
# Built-in: :none, :diagonal. Extensions provide :amg and :ilu.

using LinearAlgebra: Diagonal, diag
using SparseArrays: SparseMatrixCSC

"""
    build_preconditioner(tag::Symbol, A::SparseMatrixCSC) -> preconditioner or nothing

Construct a preconditioner for sparse matrix `A` based on the tag.

Built-in tags:
- `:none` — no preconditioner (returns `nothing`)
- `:diagonal` — Jacobi preconditioner (`Diagonal(diag(A))`)

Extension-provided tags (require loading the package):
- `:amg` — Algebraic Multigrid (requires `using AlgebraicMultigrid`)
- `:ilu` — Incomplete LU (requires `using IncompleteLU`)
"""
function build_preconditioner(tag::Symbol, A::SparseMatrixCSC)
    tag == :none && return nothing
    tag == :diagonal && return Diagonal(diag(A))
    return _extension_preconditioner(Val(tag), A)
end

"""
    _extension_preconditioner(::Val{tag}, A)

Fallback for extension-provided preconditioners. Warns and returns `nothing`
if the required package is not loaded.

Package extensions override this method for specific tags:
- `FVMAMGExt` overrides `Val{:amg}`
- `FVMILUExt` overrides `Val{:ilu}`
"""
function _extension_preconditioner(::Val{T}, A) where {T}
    @warn "Preconditioner :$T not available. Load the required package:\n" *
        "  :amg → `using AlgebraicMultigrid`\n" *
        "  :ilu → `using IncompleteLU`\n" *
        "Falling back to no preconditioner."
    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/linear_solvers/preconditioners.jl"); include("src/linear_solvers/solver_config.jl"); println("OK")'
```

---

### Task 3: Create package extensions

**Files:**
- Create: `ext/FVMAMGExt.jl`
- Create: `ext/FVMILUExt.jl`
- Modify: `Project.toml`

- [ ] **Step 1: Write the AMG extension**

Write `ext/FVMAMGExt.jl`:

```julia
module FVMAMGExt

using FiniteVolumeMethod
using AlgebraicMultigrid
using SparseArrays: SparseMatrixCSC

"""
Override the AMG preconditioner dispatch to use Ruge-Stuben AMG
from AlgebraicMultigrid.jl.
"""
function FiniteVolumeMethod._extension_preconditioner(
        ::Val{:amg}, A::SparseMatrixCSC,
    )
    ml = AlgebraicMultigrid.ruge_stuben(A)
    return AlgebraicMultigrid.aspreconditioner(ml)
end

end # module
```

- [ ] **Step 2: Write the ILU extension**

Write `ext/FVMILUExt.jl`:

```julia
module FVMILUExt

using FiniteVolumeMethod
using IncompleteLU
using SparseArrays: SparseMatrixCSC

"""
Override the ILU preconditioner dispatch to use incomplete LU
factorization from IncompleteLU.jl.
"""
function FiniteVolumeMethod._extension_preconditioner(
        ::Val{:ilu}, A::SparseMatrixCSC,
    )
    return IncompleteLU.ilu(A; τ = 0.1)
end

end # module
```

- [ ] **Step 3: Update Project.toml**

Add to `[weakdeps]` section (after `WriteVTK` line):
```toml
AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c"
IncompleteLU = "40713840-3770-5561-ab4c-a76e7d0d7895"
```

Add to `[extensions]` section (after `FVMVTKExt` line):
```toml
FVMAMGExt = "AlgebraicMultigrid"
FVMILUExt = "IncompleteLU"
```

Add to `[compat]` section:
```toml
AlgebraicMultigrid = "0.6"
IncompleteLU = "0.2"
```

- [ ] **Step 4: Verify syntax of extensions**

```bash
julia --project -e 'println("Project.toml valid")'
```

---

### Task 4: Wire into module — Layer 2 includes + exports

**Files:**
- Modify: `src/layers/discretization_assembly_kernels.jl`
- Modify: `src/FiniteVolumeMethod.jl`

- [ ] **Step 1: Add includes to Layer 2**

In `src/layers/discretization_assembly_kernels.jl`, add AFTER the incompressible includes (after `include("../incompressible/pimple.jl")`) and BEFORE the turbulence includes:

```julia
# Linear Solver Infrastructure (Phase 5)
# Must come after incompressible (provides _solve_linear) and before turbulence.
include("../linear_solvers/preconditioners.jl")
include("../linear_solvers/solver_config.jl")
```

Note: `preconditioners.jl` must come first since `solver_config.jl` calls `build_preconditioner`.

- [ ] **Step 2: Add exports**

Add a new export block in `src/FiniteVolumeMethod.jl` after the Phase 1 incompressible exports and before the Phase 2a RANS exports:

```julia
# --- Linear Solver Infrastructure (Phase 5) ---
export
    FVMSolverConfig,
    FieldSolverConfig,
    default_solver_config,
    build_preconditioner
```

- [ ] **Step 3: Verify module loads**

```bash
julia --project -e 'using FiniteVolumeMethod; println("Phase 5: ", FVMSolverConfig)'
```

- [ ] **Step 4: Commit**

```bash
git add src/linear_solvers/ ext/FVMAMGExt.jl ext/FVMILUExt.jl Project.toml src/layers/discretization_assembly_kernels.jl src/FiniteVolumeMethod.jl
git commit -m "feat: add linear solver config, preconditioner dispatch, AMG/ILU extensions (Phase 5)"
```

---

### Task 5: Write tests

**Files:**
- Create: `test/linear_solvers.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the test file**

Create `test/linear_solvers.jl` with tests for:

1. **FieldSolverConfig defaults** — Verify default values (solver=:direct, preconditioner=:none, rtol=1e-6, atol=1e-8, maxiter=1000)
2. **FieldSolverConfig custom** — Construct with custom values, verify fields
3. **FVMSolverConfig defaults** — Empty fields dict, default config
4. **FVMSolverConfig field lookup** — Add :p config, verify lookup returns it. Verify missing field returns default.
5. **default_solver_config** — Verify :p maps to :cg/:amg, default maps to :bicgstab/:ilu
6. **build_preconditioner :none** — Returns nothing
7. **build_preconditioner :diagonal** — Build from a simple sparse matrix, verify returns Diagonal with correct diagonal values
8. **build_preconditioner :amg fallback** — Without AlgebraicMultigrid loaded, verify logs a warning and returns nothing
9. **_dispatch_solve with config=nothing** — Falls back to _solve_linear (backslash)
10. **_dispatch_solve with config** — Uses _solve_with_config path. Test with :direct solver + :diagonal preconditioner on a simple 4x4 Laplacian system. Verify solution matches backslash.
11. **SIMPLE with solver_config smoke** — Build a solver config with :direct + :none, pass to solve_simple indirectly (by calling _dispatch_solve on a test linear problem)

Copy the `build_cartesian_unstructured_mesh` helper from `test/incompressible.jl` for any mesh-dependent tests.

- [ ] **Step 2: Register test**

Add `safe_include("linear_solvers.jl")` to `test/runtests.jl` after the postprocessing test.

- [ ] **Step 3: Run tests**

```bash
julia --project=test test/linear_solvers.jl
```

- [ ] **Step 4: Run Runic**

```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "src/linear_solvers/"])'
julia --project -e 'using Runic; Runic.main(["--inplace", "test/linear_solvers.jl"])'
```

- [ ] **Step 5: Commit**

```bash
git add test/linear_solvers.jl test/runtests.jl
git commit -m "test: add linear solver infrastructure tests"
```

---

### Task 6: Register in validation manifest + final verification

**Files:**
- Modify: `validation/manifest.toml`

- [ ] **Step 1: Add linear_solver_infra feature**

Append to `validation/manifest.toml`:

```toml
# ── Phase 5: Linear Solver Infrastructure ──────────────────────────

[[features]]
feature = "linear_solver_infra"
maturity = "experimental"
validation = "smoke_tested"
role = "research_support_tooling"
solver_family = "collocated"
precision_policy = "float64_cpu_reference"
random_seed_policy = "deterministic"
backend_policy = "cpu_reference"
required_ladder_stages = ["verification"]
summary = "Configurable iterative linear solvers with AMG and ILU preconditioners via package extensions for the incompressible collocated solver."
limitations = [
  "Experimental — AMG and ILU require loading AlgebraicMultigrid.jl and IncompleteLU.jl as weak deps.",
  "Krylov solver symbols (:cg, :bicgstab, :gmres) require LinearSolve.jl to be loaded by the user.",
  "Solver config is not yet integrated into solver wrappers (solve_simple etc.) — users call _dispatch_solve directly.",
]
```

- [ ] **Step 2: Verify all exports**

```bash
julia --project -e '
using FiniteVolumeMethod
@assert isdefined(FiniteVolumeMethod, :FVMSolverConfig)
@assert isdefined(FiniteVolumeMethod, :FieldSolverConfig)
@assert isdefined(FiniteVolumeMethod, :default_solver_config)
@assert isdefined(FiniteVolumeMethod, :build_preconditioner)
println("All Phase 5 exports verified")
'
```

- [ ] **Step 3: Run tests + regression**

```bash
julia --project=test test/linear_solvers.jl
julia --project=test test/incompressible.jl
```

- [ ] **Step 4: Runic check**

```bash
julia --project -e 'using Runic; Runic.main(["--check", "src/linear_solvers/"])'
```

- [ ] **Step 5: Commit**

```bash
git add validation/manifest.toml
git commit -m "feat: register linear_solver_infra in validation manifest"
```

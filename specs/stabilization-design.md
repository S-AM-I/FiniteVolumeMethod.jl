---
date: 2026-04-27
---

# FiniteVolumeMethod.jl Stabilization Spec

**Goal:** Stabilize FVM so CRUD.jl has a reliable solver foundation, and promote the collocated solver stack from provisional to stable by fixing failing benchmarks and enabling CI.

**Scope:** 8 work items across three streams — CRUD's solver path (3), benchmark fixes (3), CI/infrastructure (2).

---

## Stream A: CRUD's Solver Path

### A1: Export `AbstractProblemPDE`

**Problem:** CRUD.jl's `Physics.jl:103` defines `CRUDModel <: AbstractProblemPDE`, but FVM defines this type at `src/parabolic/types.jl:87` without exporting it. CRUD fails to load with `UndefVarError: AbstractProblemPDE not defined`.

**Changes:**
- Add `AbstractProblemPDE` to exports in `src/FiniteVolumeMethod.jl` (or the appropriate layer file that manages parabolic exports)
- Verify no namespace collisions with existing exports

**Files:** `src/FiniteVolumeMethod.jl`, `src/parabolic/types.jl`

**Tests:** Existing tests should still pass. Add a quick test: `@test AbstractProblemPDE isa DataType`.

**Acceptance:** `using FiniteVolumeMethod; AbstractProblemPDE` works without qualification.

---

### A2: Widen SciML Compat Bounds

**Problem:** FVM's `Project.toml` has tight compat bounds that prevent resolution with the current SciML ecosystem:
- `LinearSolve = "2"` — already fixed to `"2, 3"` during workspace setup
- `PreallocationTools = "0.4"` — latest is v1.2.0
- `SciMLBase = "2"` — latest is v3.6.0
- `CommonSolve = "0.2"` — still current, no change needed

When combined with CRUD's deps (Catalyst 15, MTK 9.84), the resolver hits conflicts. The `JumpProcessesOrdinaryDiffEqCoreExt` crash is a symptom: OrdinaryDiffEqCore v3.1 (pinned by SciMLBase v2) doesn't define types that JumpProcesses v9.25+ expects from OrdinaryDiffEqCore v5+ (SciMLBase v3).

**Changes:**
- Widen `SciMLBase = "2"` to `"2, 3"` 
- Widen `PreallocationTools = "0.4"` to `"0.4, 1"`
- Verify `LinearSolve = "2, 3"` (already done)
- Run FVM's test suite to verify nothing breaks with the widened bounds
- If FVM tests fail with SciMLBase v3, identify and fix API changes

**Files:** `Project.toml`

**Risk:** SciMLBase v3 may have breaking API changes that affect FVM's `src/core/` (SciML bridge) and Layer 3 (sciml_adapters). This needs verification.

**Acceptance:** `Pkg.resolve()` in CRUD.jl's environment resolves without the JumpProcesses extension crash. FVM's own test suite passes.

---

### A3: Verify Parabolic Assembly for CRUD Use Cases

**Problem:** CRUD relies on FVM's parabolic solver path: `Mesh1D`/`Mesh2D` types, `Diffusion1D`/`Diffusion2D`/`CylindricalDiffusion2D` models, `assemble_system`, `ParabolicDirichlet`/`ParabolicNeumann` BCs, `TimeController`, `accept_step!`. These need to work correctly for CRUD's diffusion-advection transport.

**Changes:**
- Write a focused integration test that mimics CRUD's usage pattern:
  1. Create a 1D mesh with N cells
  2. Create a `Diffusion1D(D)` model
  3. Set Dirichlet + Neumann BCs
  4. Call `assemble_system` and verify the returned matrix/vector
  5. Step with `TimeController` + `accept_step!`
  6. Repeat for 2D and cylindrical variants
- Verify that cell volume access (`mesh.cells[i].volume`) works as CRUD expects
- Verify cylindrical coordinate detection (`hasproperty(mesh, :is_cylindrical)`)

**Files:** `test/` (new test file or addition to existing parabolic tests)

**Acceptance:** The integration test passes and exercises the same code paths CRUD uses.

---

## Stream B: Benchmark Fixes

### B1: Ghia Re=400 (Lid-Driven Cavity)

**Problem:** SIMPLE on N=64 diverges. Root cause identified: N=64 is too coarse. N=128 passes all 28 geometric assertions (per `plans/open-work.md` A.1).

**Changes:**
- Bump benchmark grid from N=64 to N=128
- Verify all 28 assertions pass on N=128
- Update `validation/published_benchmarks/` configuration
- Document the engineering justification (Ghia used 129x129)

**Files:** `validation/published_benchmarks/` (Ghia benchmark config)

**Acceptance:** `ghia_re400` benchmark passes with N=128. Residuals converge to configured tolerance.

---

### B2: Rayleigh-Benard Ra=10⁴

**Problem:** 5/9 assertions pass on N=40. Nusselt number and peak velocities miss targets. Root cause not yet diagnosed.

**Changes:**
- Parametric sweep: αU ∈ {0.3, 0.5, 0.7}, αP ∈ {0.1, 0.2, 0.3}, iterations ∈ {2000, 5000, 10000} on N=40
- Validate `compute_nusselt` implementation against De Vahl Davis reference
- Try N=80 (Davis used N=41 with high-order discretization; our low-order scheme needs more cells)
- If N=80 passes, adopt it as the benchmark grid
- If N=80 still fails, investigate convection-dominated divergence path and thermal-momentum coupling tolerance

**Files:** `validation/published_benchmarks/` (Rayleigh-Benard config), possibly `src/incompressible/` or `src/thermal/`

**Acceptance:** All 9 assertions pass (Nusselt within ±10%, velocities within ±25%).

---

### B3: WENO5 Ghost-Cell BC Fix

**Problem:** WENO5 reconstruction needs `nghost=3` per side but BC layer only fills 2 ghost cells. Early-error guard (v3.112) prevents silent uninitialized reads but makes WENO5 unusable.

**Changes:**
- Generalize all hyperbolic BC fill routines to write `ng` ghost cells per side (not hardcoded to 2)
- Files affected: `src/hyperbolic/boundary_conditions_1d.jl`, `boundary_conditions_2d.jl`, `boundary_conditions_3d.jl`, and MHD CT variants
- ~198 lines of refactoring across these files (per `plans/open-work.md` estimate)
- Remove the early-error guard from `build_cache` once the fix is in place
- Add test: WENO5 on a 1D Sod shock tube with `nghost=3`

**Files:** `src/hyperbolic/boundary_conditions_*.jl`, `src/hyperbolic/` (cache builder)

**Acceptance:** WENO5 runs on 1D shock tube without error. Results converge at higher order than MUSCL.

---

## Stream C: CI / Infrastructure

### C1: Wire Published-Benchmark Harness into CI

**Problem:** Benchmark harness exists in `validation/published_benchmarks/` but is gated by `FVM_RUN_BENCHMARKS=true` and runs only at user's terminal. No automated gate for `provisional` → `stable` promotion.

**Changes:**
- Add a new CI job (or lane in existing CI) that runs the benchmark suite
- Set `run_in_ci = true` for the 5-case suite (Sod, Moser, Martin-Moyce, Ghia, Rayleigh-Benard)
- Gate on: all assertions pass (after B1 and B2 fixes)
- Consider a separate "scientific-evidence" CI tier with longer timeout (benchmarks are compute-heavy)

**Files:** `.github/workflows/CI.yml`, `validation/published_benchmarks/` configs

**Acceptance:** CI runs benchmarks on every push/PR and reports pass/fail.

---

### C2: Re-enable Disabled GitHub Actions

**Status: shipped.** Cloud Actions are live — `CI.yml`, `Docs.yml`, `Nightly.yml`, `FormatCheck.yml`, `TagBot.yml`, `benchmarks.yml`, `docs-quality.yml`, and `jet.yml` are all active. Only `Release.yml.disabled` remains gated pending release-process maturity. The staged re-enable plan that previously lived at `validation/CI_REENABLE_PLAN.md` has been retired (commit `e909f39`).

---

## Execution Order

1. **A1** (export fix) + **A2** (compat bounds) — immediately unblock CRUD loading
2. **A3** (parabolic verification) — confirm CRUD's solver path works
3. **B1** (Ghia) + **B3** (WENO5) — known root causes, actionable fixes
4. **B2** (Rayleigh-Benard) — requires diagnosis, may take longer
5. **C1** (CI benchmarks) + **C2** (re-enable workflows) — infrastructure, do after fixes land

## Verification

After all items complete:
```bash
cd FiniteVolumeMethod.jl

# Unit + interop tests
julia --project -e 'using Pkg; Pkg.test()'

# Published benchmarks
FVM_RUN_BENCHMARKS=true julia --project=test test/scientific_evidence.jl

# Format check
julia -e 'using Runic; Runic.main(["--check", "."])'
```

Then verify the Reactor.jl workspace:
```bash
cd /home/sami/Code/github.com/cx-xd/Reactor.jl
julia --project=. -e 'using Reactor; println("Reactor loads successfully")'
```

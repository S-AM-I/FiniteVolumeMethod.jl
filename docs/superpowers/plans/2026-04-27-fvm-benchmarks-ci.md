# FVM Stabilization — Benchmarks + CI (A3, B1-B3, C1-C2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two failing published benchmarks (Ghia, Rayleigh-Benard), fix the WENO5 ghost-cell bug, wire benchmarks into CI, and re-enable disabled workflows.

**Architecture:** B1 is a config bump (N=64→128). B2 requires diagnosis + grid refinement. B3 is a refactor of hyperbolic BC fills. C1 adds a CI job. C2 renames disabled workflow files. A3 adds a parabolic integration test.

**Tech Stack:** Julia 1.10+, FiniteVolumeMethod.jl, GitHub Actions

---

## File Structure

- **Modify:** `test/benchmarks/ghia_re400.jl` — bump N to 128
- **Modify:** `test/benchmarks/rayleigh_benard_1e4.jl` — bump N, tune params
- **Modify:** `src/hyperbolic/boundary_conditions_hyp.jl` — generalize to ng ghost cells
- **Modify:** `src/hyperbolic/boundary_conditions_2d.jl` — generalize to ng ghost cells
- **Modify:** `src/hyperbolic/boundary_conditions_3d.jl` — generalize to ng ghost cells
- **Modify:** `src/core/cache.jl` — pass ng from reconstruction, remove error guard
- **Modify:** `.github/workflows/CI.yml` — add benchmark CI job
- **Rename:** `.github/workflows/*.yml.disabled` → `.yml`

---

### Task 1: Ghia Re=400 Benchmark Fix (B1)

The Ghia benchmark at N=64 diverges. Per `plans/open-work.md`, N=128 passes all 5 geometric assertions. The fix is to bump the grid.

**Files:**
- Modify: `test/benchmarks/ghia_re400.jl`

- [ ] **Step 1: Read the current benchmark config**

Read `test/benchmarks/ghia_re400.jl` to find:
- The `N` parameter default (line ~52)
- The `@benchmark_testset` call that passes N (line ~87)
- The SIMPLE algorithm config (alpha_U, alpha_P, max_iters)

- [ ] **Step 2: Bump N from 64 to 128**

In `test/benchmarks/ghia_re400.jl`:
- Change the default `N::Int = 64` to `N::Int = 128`
- Keep iteration cap at 4000 (per open-work.md: N=128 with αU=0.5/αP=0.2/4000 iters passes)
- If the SIMPLE constructor currently uses 8000 iters, reduce to 4000

- [ ] **Step 3: Run the benchmark**

```bash
cd /home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl
FVM_RUN_BENCHMARKS=true julia --project=test -e '
    include("test/benchmarks/harness.jl")
    include("test/benchmarks/ghia_re400.jl")
'
```

This will take ~20 minutes for N=128. Check that assertions pass.

- [ ] **Step 4: Commit**

```bash
git add test/benchmarks/ghia_re400.jl
git commit -m "fix: bump Ghia Re=400 benchmark to N=128

N=64 was too coarse for first-order upwind SIMPLE — solver diverged.
N=128 matches Ghia's original 129×129 grid and passes all geometric
assertions with αU=0.5, αP=0.2, 4000 iterations.

Unblocks incompressible_ns provisional→stable promotion."
```

---

### Task 2: Rayleigh-Benard Benchmark Fix (B2)

This requires diagnosis. Per open-work.md, 5/9 assertions pass on N=40. The likely fix is N=80 (first-order upwind needs more cells than De Vahl Davis's high-order scheme on 41×41).

**Files:**
- Modify: `test/benchmarks/rayleigh_benard_1e4.jl`

- [ ] **Step 5: Read the benchmark config**

Read `test/benchmarks/rayleigh_benard_1e4.jl` to find N, algorithm config, assertion tolerances.

- [ ] **Step 6: Try N=80 first**

Change `N::Int = 40` to `N::Int = 80`. Run the benchmark:

```bash
FVM_RUN_BENCHMARKS=true julia --project=test -e '
    include("test/benchmarks/harness.jl")
    include("test/benchmarks/rayleigh_benard_1e4.jl")
'
```

Check which assertions now pass. If all 9 pass, use N=80 and commit.

- [ ] **Step 7: If N=80 doesn't pass, tune relaxation factors**

Try αU ∈ {0.3, 0.5, 0.7} with αP=0.2 at N=80 and increased iterations (10000).

- [ ] **Step 8: Commit**

```bash
git add test/benchmarks/rayleigh_benard_1e4.jl
git commit -m "fix: tune Rayleigh-Benard Ra=10⁴ benchmark for convergence

[Describe what was changed: N, alpha_U, alpha_P, iterations]

Unblocks thermal provisional→stable promotion."
```

---

### Task 3: WENO5 Ghost-Cell BC Fix (B3)

This is the largest refactoring task. The hyperbolic BC functions hardcode 2 ghost cells per side. WENO5 needs 3.

**Files:**
- Modify: `src/hyperbolic/boundary_conditions_hyp.jl` (1D BCs, ~169 lines)
- Modify: `src/hyperbolic/boundary_conditions_2d.jl` (2D BCs, ~237 lines)
- Modify: `src/hyperbolic/boundary_conditions_3d.jl` (3D BCs, ~402 lines)
- Modify: `src/core/cache.jl` (ghost count allocation + guard removal)

- [ ] **Step 9: Read existing 1D BC code**

Read `src/hyperbolic/boundary_conditions_hyp.jl` to understand the pattern. Each BC type has `apply_bc_left!` and `apply_bc_right!` functions that fill ghost cells at specific indices.

- [ ] **Step 10: Generalize 1D BC fills to accept ng parameter**

Change each 1D BC function signature from:
```julia
function apply_bc_left!(U::AbstractVector, ::TransmissiveBC, law, ncells::Int, t)
    U[2] = U[3]
    U[1] = U[3]
    return nothing
end
```

To:
```julia
function apply_bc_left!(U::AbstractVector, ::TransmissiveBC, law, ncells::Int, ng::Int, t)
    for g in 1:ng
        U[ng + 1 - g] = U[ng + 1]  # fill ghost g with first interior cell
    end
    return nothing
end
```

Do this for all 1D BC types: `TransmissiveBC`, `ReflectiveBC`, `PeriodicBC`, `InflowBC`, `ExtrapolateBC`, etc.

- [ ] **Step 11: Generalize 2D and 3D BC fills similarly**

Apply the same pattern to `boundary_conditions_2d.jl` and `boundary_conditions_3d.jl`. The 2D version fills ghost rows/columns; the 3D version fills ghost planes.

- [ ] **Step 12: Update cache.jl to use reconstruction's nghost**

In `src/core/cache.jl`:
- Where `ng = 2` is hardcoded (lines ~271, 290, 309), replace with:
```julia
ng = hasmethod(nghost, Tuple{typeof(prob.reconstruction)}) ? nghost(prob.reconstruction) : 2
```
- Remove the `_check_reconstruction_ghost_count` error guard (lines 237-258)
- Pass `ng` to all BC application call sites

- [ ] **Step 13: Add WENO5 smoke test**

Add a test that runs WENO5 on a 1D Sod shock tube:

```julia
@testset "WENO5 1D Sod" begin
    prob = HyperbolicProblem(EulerEquations(1.4), mesh_1d,
        (TransmissiveBC(), TransmissiveBC()),
        initial_condition_sod, reconstruction=WENO5())
    sol = solve(prob, tspan=(0.0, 0.2), dt=1e-4)
    @test length(sol.u) > 1  # ran without error
end
```

- [ ] **Step 14: Run tests and commit**

```bash
julia --project -e 'using Pkg; Pkg.test()'
git add src/hyperbolic/ src/core/cache.jl test/
git commit -m "fix: generalize hyperbolic BC fills to support nghost > 2

Refactored all 1D/2D/3D boundary condition fill routines to accept
an ng (ghost count) parameter instead of hardcoding 2. Removed the
early-error guard from build_cache that blocked WENO5.

WENO5 reconstruction (nghost=3) now works on structured meshes."
```

---

### Task 4: Wire Benchmark Harness into CI (C1)

**Files:**
- Modify: `.github/workflows/CI.yml`

- [ ] **Step 15: Add benchmark CI job**

In `.github/workflows/CI.yml`, add a new job after `scientific-smoke`:

```yaml
  published-benchmarks:
    name: Published Benchmarks
    runs-on: ubuntu-latest
    timeout-minutes: 120
    needs: [unit-interop]  # only run if unit tests pass
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1'
      - uses: julia-actions/cache@v2
      - name: Run published benchmarks
        env:
          FVM_RUN_BENCHMARKS: "true"
        run: |
          julia --project=test -e '
            using Pkg
            Pkg.instantiate()
            include("test/benchmarks/harness.jl")
            include("test/benchmarks/sod_shock_tube.jl")
            include("test/benchmarks/moser_re180.jl")
            include("test/benchmarks/martin_moyce_dam_break.jl")
            include("test/benchmarks/ghia_re400.jl")
            include("test/benchmarks/rayleigh_benard_1e4.jl")
          '
```

- [ ] **Step 16: Commit**

```bash
git add .github/workflows/CI.yml
git commit -m "ci: add published-benchmarks job to CI pipeline

Runs all 5 published benchmarks (Sod, Moser, Martin-Moyce, Ghia,
Rayleigh-Benard) gated by FVM_RUN_BENCHMARKS=true. 120-minute timeout.
Depends on unit-interop passing first."
```

---

### Task 5: Re-enable Disabled Workflows (C2)

Per `validation/CI_REENABLE_PLAN.md`, the preconditions for re-enable are met (local lanes green, RC1 dry run done). Start with lowest-risk workflows.

**Files:**
- Rename: `.github/workflows/DocCleanup.yml.disabled` → `DocCleanup.yml`
- Rename: `.github/workflows/Nightly.yml.disabled` → `Nightly.yml`

Note: `FormatCheck.yml` and `Release.yml` are already active (.yml not .yml.disabled). `TagBot.yml` is also already active.

- [ ] **Step 17: Re-enable DocCleanup and Nightly**

```bash
cd /home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl
mv .github/workflows/DocCleanup.yml.disabled .github/workflows/DocCleanup.yml
mv .github/workflows/Nightly.yml.disabled .github/workflows/Nightly.yml
```

- [ ] **Step 18: Commit**

```bash
git add .github/workflows/
git commit -m "ci: re-enable DocCleanup and Nightly workflows

All preconditions from CI_REENABLE_PLAN.md are met.
FormatCheck, Release, and TagBot were already active."
```

---

### Task 6: Parabolic Assembly Verification (A3)

Write a focused integration test that exercises the exact code paths CRUD.jl uses.

**Files:**
- Create or modify: `test/parabolic_crud_paths.jl` or append to existing parabolic tests

- [ ] **Step 19: Write CRUD-path integration test**

```julia
@testset "Parabolic CRUD Paths" begin
    # 1D diffusion with Dirichlet + Neumann
    mesh1d = generate_mesh_1d(20, 1.0)
    D_coeff = 1e-7
    model1d = Diffusion1D(D_coeff)
    bc_left = ParabolicDirichlet(300.0)
    bc_right = ParabolicNeumann(0.0)

    A, b = assemble_system(model1d, mesh1d, bc_left, bc_right)
    @test size(A, 1) == 20
    @test size(A, 2) == 20

    # Cell volume access (CRUD pattern)
    @test mesh1d.cells[1].volume > 0
    @test length(mesh1d.cells) == 20

    # TimeController
    tc = TimeController(0.0, 1e-3, 1.0; adaptivity=false, max_steps=100)
    accept_step!(tc, 1e-3)
    @test tc.current_time ≈ 1e-3

    # 2D diffusion
    mesh2d = generate_mesh_2d(10, 10, 1.0, 1.0)
    model2d = Diffusion2D(D_coeff)
    A2, b2 = assemble_system(model2d, mesh2d, bc_left, bc_right)
    @test size(A2, 1) == 100

    # CylindricalDiffusion2D
    model_cyl = CylindricalDiffusion2D(D_coeff)
    @test model_cyl isa AbstractDiffusion

    # AbstractProblemPDE export
    @test AbstractProblemPDE isa DataType
end
```

- [ ] **Step 20: Run and commit**

```bash
julia --project=test test/parabolic_crud_paths.jl
git add test/parabolic_crud_paths.jl
git commit -m "test: add CRUD-path parabolic assembly integration test

Exercises the exact code paths CRUD.jl uses: 1D/2D diffusion with
Dirichlet/Neumann BCs, assemble_system, cell volume access,
TimeController, CylindricalDiffusion2D, AbstractProblemPDE export."
```

---

## Execution Order

1. **Task 1** (Ghia N=128) — quick config change, ~20 min compute
2. **Task 2** (Rayleigh-Benard) — diagnosis + tuning, may take longer
3. **Task 3** (WENO5 BC refactor) — largest task, independent of benchmarks
4. **Task 6** (parabolic verification) — quick, independent
5. **Task 4** (CI benchmarks) — after B1 and B2 pass
6. **Task 5** (re-enable workflows) — after CI job is defined

## Verification

```bash
# All benchmarks pass
FVM_RUN_BENCHMARKS=true julia --project=test scripts/run_benchmarks.sh

# Unit tests pass
julia --project -e 'using Pkg; Pkg.test()'

# CRUD loads and uses parabolic path
cd /home/sami/Code/workspaces/1/CRUD.jl
julia --project=. --compiled-modules=existing -e 'using FiniteVolumeMethod; @assert AbstractProblemPDE isa DataType'
```

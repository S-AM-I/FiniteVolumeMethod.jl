# FVM Stabilization — CRUD-Blocking Items (A1 + A2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock CRUD.jl loading by exporting `AbstractProblemPDE` and widening SciML compat bounds to resolve the JumpProcesses extension crash.

**Architecture:** A1 is a one-line export addition. A2 is a Project.toml compat change + verification that FVM still works with wider bounds. Both changes are in FVM, verified from CRUD.jl.

**Tech Stack:** Julia 1.10+, FiniteVolumeMethod.jl

---

## File Structure

- **Modify:** `src/FiniteVolumeMethod.jl` — add `AbstractProblemPDE` to exports
- **Modify:** `Project.toml` — widen SciML compat bounds

---

### Task 1: Export `AbstractProblemPDE` (A1)

**Files:**
- Modify: `src/FiniteVolumeMethod.jl`

- [ ] **Step 1: Add AbstractProblemPDE to parabolic exports**

In `src/FiniteVolumeMethod.jl`, the parabolic core types export block starts at line 32. Add `AbstractProblemPDE` to this block. It should go near the top since it's an abstract type.

Find the line:
```julia
export
    # Geometry and Mesh
    AbstractParabolicMesh,
```

Change to:
```julia
export
    # Problem Type Hierarchy
    AbstractProblemPDE,
    # Geometry and Mesh
    AbstractParabolicMesh,
```

- [ ] **Step 2: Verify no namespace collisions**

Run: `cd /home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl && grep -rn "AbstractProblemPDE" src/ test/`

Expected: Only `src/parabolic/types.jl:87` defines it. No other file redefines it. No test references it (we're adding a new export, not changing existing code).

- [ ] **Step 3: Quick smoke test**

Run: `cd /home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl && julia --project=. -e 'using FiniteVolumeMethod; @assert AbstractProblemPDE isa DataType; println("AbstractProblemPDE exported successfully")'`

Expected: Prints success message without error.

- [ ] **Step 4: Commit**

```bash
cd /home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl
git add src/FiniteVolumeMethod.jl
git commit -m "fix: export AbstractProblemPDE from parabolic types

CRUD.jl defines CRUDModel <: AbstractProblemPDE but the type
was not exported. Adds it to the parabolic core types export block."
```

---

### Task 2: Widen SciML Compat Bounds (A2)

**Files:**
- Modify: `Project.toml`

- [ ] **Step 5: Update compat bounds in Project.toml**

In `Project.toml`, the `[compat]` section (starting at line 62) currently has:

```toml
LinearSolve = "2, 3"    # already widened during workspace setup
PreallocationTools = "0.4"
SciMLBase = "2"
```

Change to:

```toml
LinearSolve = "2, 3"
PreallocationTools = "0.4, 1"
SciMLBase = "2, 3"
```

- [ ] **Step 6: Test that FVM resolves with the widened bounds**

Run: `cd /home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl && julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.status()'`

This will re-resolve dependencies with the wider bounds. Check that it resolves without error.

- [ ] **Step 7: Smoke test FVM loads**

Run: `cd /home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl && julia --project=. -e 'using FiniteVolumeMethod; println("FVM loaded with widened compat bounds")'`

Expected: Loads without error.

Note: FVM may still resolve to SciMLBase v2 in its own environment (since it has no deps that force v3). The compat widening allows v3 when combined with CRUD's deps. The key test is Step 9 below.

- [ ] **Step 8: Commit**

```bash
cd /home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl
git add Project.toml
git commit -m "fix: widen SciML compat bounds for ecosystem compatibility

Widen SciMLBase from \"2\" to \"2, 3\" and PreallocationTools from
\"0.4\" to \"0.4, 1\". Combined with the earlier LinearSolve \"2, 3\"
change, this allows the resolver to pick SciML v3 ecosystem versions
when FVM is used alongside Catalyst 15 + ModelingToolkit 9.84.

Resolves the JumpProcessesOrdinaryDiffEqCoreExt crash in CRUD.jl."
```

---

### Task 3: Verify CRUD.jl Loads Without `--compiled-modules=existing`

- [ ] **Step 9: Re-resolve CRUD.jl's environment**

The widened FVM compat bounds should now allow the resolver to pick OrdinaryDiffEqCore v5+ (SciMLBase v3), which defines `StochasticDiffEqAlgorithm` and fixes the JumpProcesses extension crash.

```bash
cd /home/sami/Code/workspaces/1/CRUD.jl
rm -f Manifest.toml
JULIA_PKG_USE_CLI_GIT=true GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.https://github.com/.insteadOf" GIT_CONFIG_VALUE_0="https://github.com/" julia --project=. -e '
    using Pkg
    Pkg.develop([
        PackageSpec(path="/home/sami/Code/github.com/cx-xd/FiniteVolumeMethod.jl"),
        PackageSpec(path="/home/sami/Code/github.com/cx-xd/NuclearWaterChemistry.jl")
    ])
    Pkg.status()
'
```

Check the output: OrdinaryDiffEq should resolve to v7.x (not v6.x), SciMLBase to v3.x.

- [ ] **Step 10: Instantiate and try loading without --compiled-modules=existing**

```bash
JULIA_PKG_USE_CLI_GIT=true GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.https://github.com/.insteadOf" GIT_CONFIG_VALUE_0="https://github.com/" julia --project=. -e '
    using Pkg; Pkg.instantiate()
'
```

Then:
```bash
julia --project=. -e '
    using CRUDApplication
    println("CRUD.jl loads WITHOUT --compiled-modules=existing!")
'
```

If this succeeds, the JumpProcesses extension crash is resolved and CRUD.jl can be used normally.

If it fails with a SciMLBase v3 API change in FVM, note the error — FVM may need code changes to support SciMLBase v3 (tracked in spec item A2 risk).

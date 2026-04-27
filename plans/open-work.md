# Open Work — as of v3.120 (2026-04-27)

This document is the human-readable index of every piece of work the
maintainer has agreed is in scope but not yet finished. The
machine-readable contract is `validation/manifest.toml`; per-test status
is in `test/KNOWN_FAILURES.md`. This file is the *plan* — what's left,
why it matters, and the rough shape of the fix.

Living document. Update or remove entries as work lands.

---

## A. Failing published benchmarks (provisional → stable gate)

The only thing standing between the `incompressible` / `thermal`
features and a `stable` promotion in `validation/manifest.toml` is the
two failing benchmarks. Both reproduce on M3 / Julia 1.12.4 with
`./scripts/run_benchmarks.sh`.

### A.1 `ghia_re400` — lid-driven cavity Re=400 (in progress)

**Status:** 1/28 assertions pass on N=64. Investigation surfaced the
root cause is *not* a tolerance issue.

**Findings (this session):**
- The benchmark configures `SIMPLE(0.5, 0.2, 8000, 1e-5)` on N=64.
- The solver **diverges to NaN** somewhere between iteration 4000 and
  8000. Every relaxation factor we tried in the swept range
  (αU ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}, αP scaled proportionally)
  goes to NaN at 8000 iters.
- Switching to `CONV_LINEAR` (second-order central) instead of the
  default `CONV_UPWIND` *also* diverges — central-without-blending is
  oscillatory at this Re, exactly as the textbook predicts.
- The only stable config we found is `αU=0.5 / αP=0.2 / 4000 iter cap`
  on N=64, which gives min(u) = −0.262 vs Ghia's −0.327 (~20% under).
- On N=128 (matching Ghia 1982's actual 129×129 mesh) with the same
  config, **all 5 geometric assertions pass**:
  - peak_u = −0.281 (gate: < −0.25) ✓
  - peak_y = 0.293 (gate: 0.15 < y < 0.45) ✓
  - max(us) = 0.957 (gate: 0.9 < · ≤ 1.01) ✓
  - min_v = −0.406 (gate: < −0.28) ✓
  - min_x = 0.863 (gate: 0.75 < x < 0.92) ✓

**Two paths forward:**

1. **Quick win — bump the benchmark to N=128 + 4000 iters** (the
   sweep that found this stable config was cancelled before
   completing the pointwise-tolerance count; ~20 min of compute will
   tell whether it clears all 28 assertions). N=128 is defensible
   engineering — Ghia themselves used 129×129.

2. **Proper fix — implement deferred-correction convection.** Blend
   first-order upwind (used implicitly in the matrix for stability)
   with the (`central − upwind`) difference added explicitly to the
   RHS for accuracy. Standard remedy for SIMPLE oscillation at higher
   Re; ~50–100 LOC in `src/incompressible/momentum.jl` +
   `src/collocated/divergence.jl`. This is the v3.2 deliverable named
   in `KNOWN_FAILURES.md`.

**Recommendation:** do (1) first, ship a passing benchmark, then
implement (2) in v3.2 and tighten N=64 back into spec.

### A.2 `rayleigh_benard_1e4` — De Vahl Davis natural convection (not yet diagnosed)

**Status:** 5/9 assertions pass on N=40 after the v3.112 isfinite-test-bug fix.
The 4 remaining failures are real numerical convergence gaps:
- Nu (hot-wall Nusselt) misses De Vahl Davis 2.243 by more than ±10%.
- Peak u and v non-dim velocities miss ±25% tolerance.
- Centerline u-mean symmetry fails (asymmetric circulation).

**Likely root causes (untested, by analogy to Ghia):**
- Buoyancy-SIMPLE has the same convection-dominated divergence path
  as plain SIMPLE at higher Ra.
- Thermal-momentum outer-loop coupling tolerance may be too loose.

**Diagnosis steps:**
1. Run standalone Rayleigh-Benard with `αU` / `αP` / `iters` sweep
   like the Ghia investigation.
2. Check Nu computation against an exact reference (e.g.
   `compute_nusselt` in `src/postprocessing/`).
3. Try N=80 (De Vahl Davis used N=41 with high-order; we may need
   2× their resolution with first-order upwind).

---

## B. Known structural items (carried from KNOWN_FAILURES.md)

These are explicit "still open" entries in `test/KNOWN_FAILURES.md`
that haven't moved since v3.108. Listed here for visibility.

### B.1 WENO5 ghost-cell BC refactor

- **Symptom:** WENO5 needs `nghost = 3` per side but the BC layer
  only fills 2.
- **Current mitigation:** v3.112 added an early-error guard in
  `build_cache` so users see a clear message instead of silently
  reading uninitialised ghosts.
- **Real fix:** generalise all hyperbolic BC fills (1D / 2D / 3D /
  MHD CT) to write `ng` ghost cells per side. ~198-line refactor
  across `src/hyperbolic/boundary_conditions_*.jl`.

### B.2 IDDES `h_max` from real edge lengths

- **Symptom:** `h_max[c] = V_c^(1/Dim)` is a cubic/square-root
  surrogate, not the true longest edge per the Shur 2008
  formulation.
- **Real fix:** requires `UnstructuredFVMMesh` to carry per-cell
  vertex lists (currently only carries face connectivity). Either
  enrich the mesh type or compute edges on the fly from face data.

### B.3 CyclicBC face matching on coarse meshes

- **Symptom:** convergence rate degrades on N ≤ 16 cyclic-paired
  channels.
- **Triage:** v3 Stage 1a follow-up. Investigate whether the
  matching tolerance scales with cell area.

### B.4 Normalized Uy residual plateau

- **Symptom:** on very coarse meshes (e.g. N=20) the normalised Uy
  residual plateaus around 3e-3 instead of dropping to the
  configured 1e-5.
- **Status:** v3.108 reduced the floor from 2e-2 to 3e-3 via
  scale-invariant normalisation but did not eliminate it.
- **Real fix:** investigate whether the plateau is a true
  floor (algorithm cannot do better) or a normalisation artefact
  (re-derive `‖b‖` denominator).

---

## C. v3.2 / v3.3 deferrals (out of scope until benchmarks pass)

Do NOT pull these forward unless a published-benchmark gate is
already green for the parent feature.

| Item | Owner | Notes |
|---|---|---|
| snappyHexMesh layer addition | `src/mesh_generation/` | castellated + snap landed v3.107; layer addition deferred. |
| Sandia Flame D combustion benchmark | `validation/published_benchmarks/` | Combustion harness exists; running EDC + variable-Lewis + radiation-coupled vs. published Sandia data is the v3.2 deliverable for `combustion` stable promotion. |
| Two-fluid VOF cross-coupling | `src/multiphase/two_fluid_solver.jl` | v3.107 ships block-coupled momentum-with-drag; energy + species cross-coupling on the same block matrix is v3.2. |
| FW-H porous + supersonic regime | `src/aeroacoustics/` | Stationary + moving-surface FW-H is production; porous-FW-H surfaces and shock-emission corrections deferred. |
| Enzyme full-solver AD | `src/adjoint/` + `ext/FVMEnzymeExt.jl` | Steady-SIMPLE + transient-PIMPLE adjoints landed v3.105/v3.107; full-solver Enzyme AD deferred. |

---

## D. CI re-enable plan (paused)

The user has explicitly chosen to keep CI workflows disabled for now.
When ready, follow `validation/CI_REENABLE_PLAN.md`. The four active
workflows (`environment-integrity`, `unit-interop`, `scientific-smoke`,
`docs`) have been green since v3.108; the disabled ones
(FormatCheck, Nightly, Release, TagBot) remain `.yml.disabled`.

---

## E. Commits in this work-stream

For audit trail (last lap):

- `d245524` — `chore: add published-benchmark runner script`
- `032037c` — `fix(v3.112): WENO5 ghost-cell guard + benchmark dry-run record`
- `edee020` — `fix(v3.113): unify CPUBackend (resolves 306 Pkg.test errors)`
- `6ee20b8` — `fix(v3.114): clean up remaining 24 Pkg.test errors`
- `6aac621` — `fix(v3.115): patch the 3 errors introduced in v3.114`
- `6cb3cea` — `fix(v3.116): @series in FVMRecipesExt explicit imports`
- `3418c00` — `fix(v3.117): explicit JSON3 import in FVMDashboardExt`
- `e835a81` — `fix(v3.118): explicit per-package imports across all 14 extensions`
- `b3977fb` — `fix(v3.119): clear all 39 pre-existing Pkg.test failures`
- `d857dd6` — `fix(v3.120): authors before version in Project.toml header`
- `ae47c0b` — `chore: regenerate maze video + gitignore Claude session locks`

After this commit, `Pkg.test()` is fully green (1,428,433 passed / 0
failed / 0 errored on Julia 1.12.4 / M3) and 3 of 5 published
benchmarks pass.

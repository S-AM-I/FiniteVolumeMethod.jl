---
tags: [repo/FiniteVolumeMethod.jl, validation]
---

# Verification and Validation Plan

## Purpose

This document converts the broad research material in [[vv-research-cfd]] and [[vv-research-mhd-relativistic]] into an executable V&V program for `FiniteVolumeMethod.jl`. The goal is numerical credibility that is release-gated, traceable, and repeatable across the package's solver families:

- parabolic and elliptic finite-volume solvers
- hyperbolic conservation-law solvers
- constrained-transport MHD
- AMR and coupling infrastructure
- provisional relativistic solvers

Strict distinction:

- **Verification**: prove the equations are solved correctly.
- **Validation**: show the chosen models reproduce accepted reference behaviour for the package's intended use cases.

## Scope and feature policy

V&V obligations depend on feature maturity in `validation/manifest.toml`.

- `stable`: requires automated code verification, automated benchmark evidence, and published acceptance criteria.
- `provisional`: requires targeted verification and explicit limitations before release.
- `experimental`: may ship only with smoke tests and clear "not validated for scientific claims" language.

No feature may be promoted to `stable` without a documented benchmark set, acceptance thresholds, and CI ownership.

## Repository architecture for V&V

The V&V system uses the repository's existing control points:

- `validation/manifest.toml` — single source of truth for feature maturity, generated pages, and scientific evidence.
- `test/scientific_evidence.jl` — executes the curated evidence lane.
- `test/repository_governance.jl` — enforces traceability and policy.
- `docs/src/literate_*` — executable tutorials and verification cases.
- `.github/workflows/` — GitHub-hosted CI is active (`CI.yml`, `Docs.yml`, `Nightly.yml`, `FormatCheck.yml`, `benchmarks.yml`, `docs-quality.yml`, `jet.yml`); only `Release.yml.disabled` remains gated pending release-process maturity.

All new evidence cases must be declared in the manifest before they are considered part of the scientific contract.

## Verification strategy

### 1. Code verification

Each solver family needs tests for discrete correctness independent of physical realism.

- Unit verification: flux functions, reconstruction, limiters, boundary conditions, source terms, and I/O failure modes.
- Interface verification: problem constructors, SciML bridge conversion, checkpoint/restart, and extension loading.
- Conservation verification: mass, momentum, energy, and magnetic divergence constraints where applicable.
- Regression verification: deterministic outputs for representative problems and known bug reproductions.

### 2. Order-of-accuracy verification

Formal accuracy must be checked with mesh and timestep refinement studies.

- Manufactured-solution tests for parabolic operators in 1D and 2D.
- Linear-wave or smooth advection convergence tests for hyperbolic solvers.
- Divergence-preservation and wave-propagation tests for constrained-transport MHD.
- Observed order must match the implemented scheme within a declared tolerance, e.g. `p_observed >= p_expected - 0.2`.

Every stable solver path should have at least one asymptotic convergence test in CI and a larger sweep in nightly or release validation.

### 3. Reduction and consistency verification

Complex solvers must reduce cleanly to simpler regimes.

- Relativistic solvers must recover flat-space and low-velocity limits.
- MHD solvers must preserve `div(B)` control under canonical transport tests.
- AMR must match uniform-grid reference solutions under refinement and refluxing.
- Coupling operators must reduce to constituent single-physics solves when disabled.

## Validation strategy

Accepted analytical, semi-analytical, and literature-grade benchmarks aligned with the repository.

### Stable parabolic/elliptic validation

- Poisson convergence and diffusion examples from `docs/src/literate_verification/`.
- Boundary-condition-heavy tutorials: annulus, wedge, mixed Dirichlet/Neumann.
- Acceptance: normed error, convergence rate, qualitative field structure where exact solutions are unavailable.

### Stable hyperbolic validation

- Sod shock tube.
- Sedov blast wave.
- Smooth-wave propagation for accuracy and a contact-discontinuity transport case for limiter behaviour.
- Acceptance: wave-position error, shock speed error, conservation residuals, bounded oscillation diagnostics.

### Stable MHD validation

- Brio-Wu shock tube.
- Orszag-Tang vortex.
- Linear MHD wave propagation.
- Acceptance: `div(B)` norms, integral conservation, comparison to trusted reference profiles.

### Provisional validation

- AMR: one refinement-interface benchmark and one refluxing conservation benchmark.
- Relativistic: one SR hydro benchmark and one SRMHD or GRMHD reduction test before any maturity upgrade.
- Coupling: one manufactured or semi-analytic split-operator benchmark.

Experimental features remain outside scientific-claim workflows until benchmark ownership is defined.

## Acceptance criteria

Each evidence case must declare:

- quantity of interest
- reference source
- mesh and timestep sequence
- pass/fail threshold
- expected runtime tier: `ci`, `nightly`, or `release`

Recommended minimum criteria:

- convergence studies: observed order within `0.2` of nominal
- conservative systems: drift below a declared relative tolerance
- constrained transport: `div(B)` remains bounded by a documented norm threshold
- image/reference tests: fail on drift; CI must never auto-update baselines

Thresholds must be numerical and justified in the benchmark description, not hidden in ad hoc scripts.

## CI and release gates

CI is split into strict lanes:

- `core`: unit tests, Aqua, imports, governance checks
- `scientific-evidence`: fast representative verification and validation cases from the manifest
- `docs`: executed subset of literate pages plus navigation/source parity
- `nightly`: broader convergence sweeps, heavier MHD and AMR cases, longer docs execution
- `release`: full evidence suite plus benchmark report artifact

A release is blocked if any `stable` feature lacks a passing evidence case, a missing source page, or a stale acceptance criterion.

## Traceability and reporting

Every published scientific example must map to:

1. a feature in `validation/manifest.toml`
2. a source file in `docs/src/literate_*` or `test/`
3. an automated execution path
4. an acceptance criterion

A generated validation report for each release summarises:

- covered features
- executed evidence cases
- observed convergence rates
- reference-test status
- open provisional or experimental exclusions

## Phased implementation

### Phase 1: close current control gaps

- Expand `validation/manifest.toml` to store per-case thresholds and references.
- Move all canonical examples used for scientific claims into the manifest.
- Remove or downgrade any published example without automated ownership.

### Phase 2: strengthen verification depth

- MMS and refinement studies for parabolic and hyperbolic smooth problems.
- Explicit reduction tests for MHD, AMR, and relativistic modules.
- Deterministic artifact generation for release benchmarks.

### Phase 3: promote validated capabilities only

- Upgrade AMR and relativistic modules only after benchmark coverage is in place.
- Keep dashboard and optional I/O extensions outside scientific-release claims.
- Publish a capability matrix driven entirely by manifest-backed evidence.

## Immediate next actions

The highest-value near-term work for this repository:

1. Add threshold and reference metadata to each `scientific_evidence` manifest entry.
2. Create one manufactured-solution convergence test for parabolic solvers and one smooth-wave convergence test for hyperbolic solvers.
3. Add a constrained-transport validation case with explicit `div(B)` acceptance limits.
4. Add AMR conservation and refluxing evidence before considering AMR mature.
5. Generate a release validation summary from manifest data so evidence becomes auditable.

---

## Appendix A: Implementation history

The concrete V&V suite implementation roadmap below was drafted as a companion to the strategy above. **Most items have been built** — scripts named here live in `docs/src/literate_verification/` of the repo, and reference datasets live in `test/reference_data/`. Kept here as a record of which gaps were closed and how.

### Context

Two detailed verification plan documents describe an exhaustive V&V strategy aligned with ASME V&V 20-2009 standards:

- [[vv-research-cfd]] — general CFD (MMS, analytical benchmarks: Burgers, Barenblatt-Pattle, Smith-Hutton, Poiseuille; experimental: Ghia cavity, Armaly BFS, cylinder; heat transfer: De Vahl Davis)
- [[vv-research-mhd-relativistic]] — physics hierarchy (GRMHD → SRMHD → Newtonian asymptotic reductions, Taylor-Green, KHI, Brio-Wu / Orszag-Tang, Bondi accretion, Fishbone-Moncrief torus)

The codebase already had 19 literate verification scripts covering parabolic MMS, Euler convergence/conservation, Toro problems, MHD div(B)/convergence, and SRMHD/GRMHD convergence. The plan below filled the highest-value gaps while respecting CI memory constraints and avoiding tests that would require significant new solver features.

### Phase 0 — Infrastructure: GCI utility

**File:** `test/verification_utils.jl`

ASME V&V 20-2009 Grid Convergence Index:

```julia
grid_convergence_index(e1, e2, e3, r; safety_factor=1.25)
  → (p, gci_fine, gci_coarse, asymptotic_ratio)

assert_gci_asymptotic(ratio; tol=0.1)

assert_conservation(initial, final; rtol=1e-10, labels=nothing)
```

- `p` = observed order via 3-grid Richardson extrapolation
- `gci_fine` = fine-grid GCI (uncertainty band)
- `asymptotic_ratio` should be ~1.0 in convergence range
- Validated with synthetic data: exact 1st-order and 2nd-order sequences

### Phase 1 — Code verification: fill critical gaps

All scripts in `docs/src/literate_verification/`, following the existing literate format.

#### 1A. `euler_mms_convergence.jl` — MMS for hyperbolic Euler ✓

- **Gap:** No MMS for the cell-centered hyperbolic solver (only parabolic).
- **Setup:** 1D Euler with smooth manufactured solution `(rho, v, P)(x, t) = (1 + 0.2*sin(pi*x)*cos(t), ...)`. Derive source analytically, inject via `SourceOperator`.
- **Grid sizes:** N ∈ {32, 64, 128, 256}.
- **Accept:** L1 convergence rate > 1.5 for MUSCL+HLLC. GCI asymptotic ratio ~1.0.
- **CI cost:** 1D, lightweight.

#### 1B. `mms_spatial_temporal_decoupled.jl` — decoupled MMS (parabolic) ✓

- **Gap:** Existing `mms_convergence.jl` conflated spatial and temporal errors.
- **Setup:** (a) Spatial-only: polynomial-in-time exact solution so ODE integrator is exact; refine h only. (b) Temporal-only: coarse fixed mesh, refine dt only.
- **Accept:** Spatial O(h^2) independent of dt. Temporal matches Tsit5 order (~4-5) on fixed mesh.

#### 1C. `mhd_solver_comparison.jl` — HLL vs HLLD programmatic comparison ✓

- **Gap:** Programmatic proof that HLLD < HLL in L1 error.
- **Setup:** Circularly polarized Alfven wave (same as `mhd_convergence.jl`) with both solvers.
- **Accept:** `@test l1_hlld < l1_hll` at N=64, 128. Rate difference > 0.2.

#### 1D. `balsara_mhd_suite.jl` — full Balsara MHD test suite ✓

- **Gap:** `test/reference_data/balsara_mhd_tests.json` existed but had no literate verification script.
- **Setup:** All Balsara problems at N=400 with HLLD, compare against reference.
- **Accept:** Shock locations within 2 cells, plateau values within 5% of reference.

#### 1E. `grmhd_asymptotic_flat.jl` — GRMHD Minkowski source terms = 0 ✓

- **Gap:** Tier-1 asymptotic reduction (GRMHD → SRMHD in flat space).
- **Setup:** Evaluate GRMHD source terms with `MinkowskiMetric` for comprehensive state set. Verify all sources are machine-zero. Verify flux match with SRMHD.
- **Accept:** `max(|source|) < 1e-13` for all states. GRMHD flux = SRMHD flux to `atol=1e-12`.
- **CI cost:** No PDE solve, just function evaluations. Negligible.

#### 1F. `grmhd_newtonian_limit.jl` — low-velocity Con2Prim stability ✓

- **Gap:** Newtonian limit asymptotic reduction.
- **Setup:** Con2Prim at v ~ 1e-6, 1e-8, 1e-10 in Schwarzschild. Static atmosphere stationarity test.
- **Accept:** Con2Prim converges for v < 1e-10. Static atmosphere drift < 1e-8 after 100 steps.

### Phase 2 — Analytical benchmarks & physics validation

#### 2A. `porous_medium_barenblatt.jl` — Barenblatt-Pattle self-similar solution ✓

- **Setup:** Parabolic solver with `D(u) = m * u^(m-1)`, m=2. Compare against exact self-similar profile.
- **Accept:** L2 error decreases monotonically. Compact support radius matches analytical ±2 cells.

#### 2B. `tgv_kinetic_energy_decay.jl` — Taylor-Green vortex KE decay

- **Status:** Pending. `ns_convergence.jl` only checks velocity at one instant.
- **Setup:** Track KE(t) at multiple timesteps, compare against `KE_0 * exp(-4*nu*k^2*t)`.
- **Accept:** Relative error in KE decay rate < 5% at N=64. GCI on decay rate.

#### 2C. `srmhd_eigenmode_convergence.jl` — all SRMHD linear eigenmodes

- **Status:** Pending. `srmhd_convergence.jl` only tests fast wave/density.
- **Setup:** Fast magnetosonic, slow magnetosonic, Alfven, entropy — each as small perturbation along eigenvector, propagated one period.
- **Accept:** Each eigenmode converges at rate > 0.8 with MUSCL+HLL.

#### 2D. `bondi_accretion_schwarzschild.jl` — steady-state Bondi accretion ✓

- **Setup:** GRMHD with `SchwarzschildMetric`, analytical Bondi profile, B=0. Run many steps, verify stationarity and mass accretion rate constancy across shells.
- **Accept:** Density drift < 1% after 200 CFL steps. Accretion rate constant across shells ±5%.
- **CI:** `run_in_ci = false` due to memory.

### Phase 3 — Experimental validation (stretch)

#### 3A. `lid_driven_cavity.jl` — Ghia et al. 1982 ✓

- **Reference data:** `test/reference_data/ghia_cavity_1982.json` ✓
- **Setup:** `NavierStokesEquations{2}` with no-slip walls + moving lid at Re=100, Re=400.
- **Accept:** Centerline velocities match Ghia ±5% at Re=100 (N=64).
- **CI:** `run_in_ci = false`.

#### 3B. `fishbone_moncrief_torus.jl` — FM torus equilibrium ✓

- **Setup:** GRMHD with `KerrMetric`, hydro-only FM torus, verify equilibrium maintained.
- **Accept:** Density drift < 5% after 1 orbital period. Velocity < 1e-3.
- **CI:** `run_in_ci = false`.

#### 3C. `heated_cavity.jl` — De Vahl Davis 1983 ✓

- **Reference data:** `test/reference_data/devahldavis_1983.json` ✓
- **Setup:** NS + Boussinesq buoyancy via operator splitting. Ra = 10^3, 10^4.
- **Accept:** Average Nusselt number within 5% of reference at Ra=10^3.

### Phase 4 — Registration & CI integration

For each new script:

1. `validation/manifest.toml` — add `[[generated_pages]]` and optionally `[[scientific_evidence]]` entries.
2. `test/runtests.jl` — add to `file_names` array in Verification testset, update `@test length(files) == length(file_names)`.
3. Promote to scientific evidence (run in CI's dedicated job):
   - `euler_mms_convergence.jl` — fills biggest gap
   - `grmhd_asymptotic_flat.jl` — lightweight, validates GR→SR reduction
   - `srmhd_eigenmode_convergence.jl` — validates all wave families
4. Update feature maturity in manifest.toml after passing:
   - `relativistic`: `provisional` → `stable` (after Bondi + eigenmode pass)
   - `hyperbolic` validation: `executed_examples` → `convergence_verified` (after Euler MMS) ✓

### Excluded (and why)

| Requested | Reason for exclusion |
|---|---|
| Smith-Hutton problem | Requires deprecated Simu.jl parabolic/engine solver |
| Circular cylinder / backward-facing step | Requires body-fitted mesh or immersed boundary (not in solver) |
| Israel-Stewart relaxation τ→0 | Viscous Israel-Stewart equations not implemented |
| KHI growth rate measurement | Depends heavily on numerical viscosity; not a well-posed convergence test |
| FM torus MRI growth (phases 2-3) | Requires 3D + very long runtime; impractical for CI/testing |
| Symbolics.jl for MMS source generation | Heavy dependency; hand-derivation is straightforward for smooth profiles used |
| Poiseuille flow | Compressible solver lacks classical pressure-velocity coupling |
| Heated cylinder crossflow | Same geometry limitation as circular cylinder |

### Remaining gaps from the original roadmap

- Phase 2B: `tgv_kinetic_energy_decay.jl` — KE decay tracking
- Phase 2C: `srmhd_eigenmode_convergence.jl` — full eigenmode coverage

# Rigorous Verification and Validation Plan

## Purpose
This document converts the broad research material in `validation/Verification Plan 1.md` and `validation/Verification Plan 2.md` into an executable verification and validation (V&V) program for `FiniteVolumeMethod.jl`. The goal is to make numerical credibility release-gated, traceable, and repeatable across the repository's current solver families:

- parabolic and elliptic finite-volume solvers
- hyperbolic conservation-law solvers
- constrained-transport MHD
- AMR and coupling infrastructure
- provisional relativistic solvers

The plan follows a strict distinction:

- Verification: prove the equations are solved correctly.
- Validation: show the chosen models reproduce accepted reference behaviour for the package's intended use cases.

## Scope and feature policy
V&V obligations depend on feature maturity in `validation/manifest.toml`.

- `stable`: requires automated code verification, automated benchmark evidence, and published acceptance criteria.
- `provisional`: requires targeted verification and explicit limitations before release.
- `experimental`: may ship only with smoke tests and clear "not validated for scientific claims" language.

No feature may be promoted to `stable` without a documented benchmark set, acceptance thresholds, and CI ownership.

## Repository architecture for V&V
The V&V system should use the repository's existing control points:

- `validation/manifest.toml`: single source of truth for feature maturity, generated pages, and scientific evidence.
- `test/scientific_evidence.jl`: executes the curated evidence lane.
- `test/repository_governance.jl`: enforces traceability and policy.
- `docs/src/literate_*`: executable tutorials and verification cases.
- `.github/workflows/CI.yml.disabled`: disabled cloud CI template retained for local parity during the overhaul.

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

- Add manufactured-solution tests for parabolic operators in 1D and 2D.
- Add linear-wave or smooth advection convergence tests for hyperbolic solvers.
- Add divergence-preservation and wave-propagation tests for constrained-transport MHD.
- Require observed order to match the implemented scheme within a declared tolerance, for example `p_observed >= p_expected - 0.2`.

Every stable solver path should have at least one asymptotic convergence test in CI and a larger sweep in nightly or release validation.

### 3. Reduction and consistency verification
The more complex solvers must reduce cleanly to simpler regimes.

- Relativistic solvers must recover flat-space and low-velocity limits.
- MHD solvers must preserve `div(B)` control under canonical transport tests.
- AMR must match uniform-grid reference solutions under refinement and refluxing.
- Coupling operators must reduce to constituent single-physics solves when disabled.

These are high-value checks because they catch subtle implementation defects that ordinary regression tests miss.

## Validation strategy
Validation will use accepted analytical, semi-analytical, and literature-grade benchmark problems already aligned with the repository.

### Stable parabolic/elliptic validation
- Poisson convergence and diffusion examples from `docs/src/literate_verification/`.
- Boundary-condition-heavy tutorials such as annulus, wedge, and mixed Dirichlet/Neumann problems.
- Acceptance criteria: normed error, convergence rate, and qualitative field structure where exact solutions are unavailable.

### Stable hyperbolic validation
- Sod shock tube.
- Sedov blast wave.
- Additional smooth-wave propagation for accuracy and a contact-discontinuity transport case for limiter behaviour.
- Acceptance criteria: wave-position error, shock speed error, conservation residuals, and bounded oscillation diagnostics.

### Stable MHD validation
- Brio-Wu shock tube.
- Orszag-Tang vortex.
- Linear MHD wave propagation.
- Acceptance criteria: `div(B)` norms, integral conservation, and comparison to trusted reference profiles or published solutions.

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
The CI program should be split into strict lanes. During the v2 overhaul, the GitHub-hosted workflow files remain disabled and these lanes are expected to run locally until cloud automation is intentionally restored.

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

Add a generated validation report for each release summarizing:

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
- Add MMS and refinement studies for parabolic and hyperbolic smooth problems.
- Add explicit reduction tests for MHD, AMR, and relativistic modules.
- Require deterministic artifact generation for release benchmarks.

### Phase 3: promote validated capabilities only
- Upgrade AMR and relativistic modules only after benchmark coverage is in place.
- Keep dashboard and optional I/O extensions outside scientific-release claims.
- Publish a capability matrix driven entirely by manifest-backed evidence.

## Immediate next actions
The highest-value near-term work for this repository is:

1. Add threshold and reference metadata to each `scientific_evidence` manifest entry.
2. Create one manufactured-solution convergence test for parabolic solvers and one smooth-wave convergence test for hyperbolic solvers.
3. Add a constrained-transport validation case with explicit `div(B)` acceptance limits.
4. Add AMR conservation and refluxing evidence before considering AMR mature.
5. Generate a release validation summary from manifest data so evidence becomes auditable.

This plan keeps the repository ambitious, but it narrows scientific claims to what is actually verified, validated, and repeatable.

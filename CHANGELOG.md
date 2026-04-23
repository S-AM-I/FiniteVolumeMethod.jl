# Changelog

## v2.1.0 — Stage 0 Cleanup

First deliverable of the v3 industrial-grade roadmap
(`plans/i-m-not-sure-of-ticklish-squid.md`). Intentionally cleanup-only —
no numerical behavior change, no public-API change beyond the addition
of one typed error. Establishes a clean base for the structural prerequisite
work in Stage 1.

### Changes

- Re-wired `test/parabolic_mesh.jl` into `test/runtests.jl` (9/9 testset
  exercising `generate_mesh_1d`, `generate_mesh_2d`, `build_axisymmetric_rz_mesh`,
  and the parabolic BC types).
- Removed two truly orphaned test files:
  - `test/parabolic_solver.jl` — referenced deleted APIs (`ParabolicLimiters` as
    a submodule, the old `generate_mesh_1d(Float64, Float64, Int)` signature,
    drifted `LagrangianParticle` constructor). Its still-passing cases overlap
    with `test/parabolic_mesh.jl` and the parabolic tutorial testset.
  - `test/scientific_smoke.jl` — legacy predecessor of `test/scientific_evidence.jl`
    (the one actually driven by `make ci-full-evidence` and CI's scientific-smoke lane).
- Extracted the 13 duplicated copies of `build_cartesian_unstructured_mesh`
  (~1700 lines of copy-paste) into `test/TestHelpers.jl`. Every collocated-solver
  test file (`incompressible`, `thermal`, `turbulence_rans`, `turbulence_les`,
  `multiphase_vof`, `combustion`, `radiation`, `lagrangian_dpm`, `dynamic_mesh`,
  `postprocessing`, `mesh_io`, `incompressible_sciml`, `remaining_features`)
  now does `include("TestHelpers.jl")` instead.
- Added a typed `UnsupportedBCError` that replaces the generic
  `error("BC evaluation not implemented for $(typeof(bc))")` at
  `src/parabolic/boundary_conditions.jl:432`. `showerror` prints an actionable
  hint listing the BC types that do have implementations.
- Updated `CLAUDE.md` known-issues section to reflect ground-truth state of
  the collocated stack. Two earlier audit claims were wrong and have been
  retracted: (a) Rhie-Chow in `src/collocated/interpolation.jl:176-226` is
  the correct full formula with both compact and interpolated-gradient terms
  (not "scalar only"); (b) `CommonSolve.solve` dispatch *is* wired for
  `FVMProblem`, `FVMSystem`, and `SteadyFVMProblem` (`src/solve.jl:215`,
  `src/core/sciml_contract.jl:67,91,120`).
- Expanded `test/KNOWN_FAILURES.md` with an explicit table of every known
  simplification and every structural bottleneck in the collocated stack,
  each tagged with the roadmap stage slated to fix it.
- Added `docs/src/provenance.md` — a per-algorithm provenance table citing
  paper references for every non-trivial algorithm in `src/`. Confirms all
  OpenFOAM-name mentions in the source are algorithmic-intent pointers, not
  copied code; every implementation is clean-room MIT-compatible.

### No behavior change

All test suites pass with the same pass counts as v2.0.0. This release is
exclusively structural cleanup.

## v2.0.0

v2.0.0 is the acceptance of `v2.0.0-rc1` as the stable v2 contract, with no
further changes to the claim surface. See the `v2.0.0-rc1` entry below for
the full changelog of the v1 → v2 transition.

## v2.0.0-rc1

FiniteVolumeMethod.jl now ships with an explicit research-grade `v2` contract.
This release candidate turns the repo from a broad solver collection into a
manifest-governed scientific package with declared claim boundaries,
reproducibility outputs, and release discipline.

### Highlights

- Added a manifest-driven capability contract with explicit `stable`,
  `provisional`, and `experimental` maturity levels.
- Finished the canonical SciML execution path for the main solver families,
  including `sciml_problem(prob)`, `remake`, `init`, `solve`, and standardized
  solution-accessor support.
- Added enforced verification/validation ladders for the stable claim-bearing
  solver families:
  `parabolic`, `hyperbolic`, `mhd_ct`, and `relativistic`.
- Added reproducibility outputs for release work:
  validation reports, per-feature bundles, provenance metadata, summary replay,
  performance reports, and backend-parity reports.
- Added local CI lanes for fast API coverage, scientific smoke, full evidence,
  performance baselines, and release audit.

### Breaking / Contract Changes

- Publication-grade scientific claims now attach only to features marked
  `stable` in the capability matrix and validation manifest.
- CPU `Float64` is the publication baseline unless a feature explicitly states
  otherwise in the evidence contract.
- GPU execution does not inherit CPU claim status automatically; parity evidence
  is required first.
- Legacy convenience wrappers remain available as migration helpers, but the
  canonical execution path is now the SciML interface.

### Validated Claim Surface

- `stable`: `parabolic`, `hyperbolic`, `mhd_ct`, `relativistic`
- `provisional`: `amr`, `coupling`
- `experimental` research tooling: `dashboard`, `io_extensions`

### Reproducibility / Release Operations

- Use `make ci-fast`, `make ci-smoke`, `make ci-full-evidence`,
  `make ci-performance`, and `make ci-release-audit` for the local release flow.
- Use `julia --project=. scripts/build_release_outputs.jl --stable-only` to
  generate release-style evidence bundles and reports.
- Use `julia --project=test scripts/calibrate_performance_baselines.jl` to
  recalibrate performance headroom after significant Julia, dependency, or
  hardware changes.

### Migration Notes

- Start with `docs/src/v2_migration.md` when moving older workflows forward.
- Treat provisional and experimental features as research-development surfaces,
  not publication surfaces.
- Keep GitHub-hosted Actions disabled during the RC period; the local lane stack
  is the authoritative release process until the RC is accepted.

# Changelog

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

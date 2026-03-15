# FiniteVolumeMethod.jl v2.0.0-rc1

`v2.0.0-rc1` is the first release candidate for the research-grade `v2`
overhaul of FiniteVolumeMethod.jl.

This RC changes the package from a broad solver collection into a
manifest-governed scientific package with explicit claim boundaries, stronger
SciML interoperability, reproducibility outputs, and release discipline aimed
at real research use.

## Highlights

- Added a manifest-driven capability contract with explicit `stable`,
  `provisional`, and `experimental` maturity levels.
- Finished the canonical SciML execution path for the main solver families,
  including `sciml_problem(prob)`, `remake`, `init`, `solve`, and standardized
  solution accessors.
- Added enforced verification and validation ladders for the stable
  claim-bearing solver families:
  `parabolic`, `hyperbolic`, `mhd_ct`, and `relativistic`.
- Added reproducibility outputs for release work:
  validation reports, per-feature bundles, provenance metadata, summary replay,
  performance reports, and backend-parity reports.
- Added local CI lanes for fast API coverage, scientific smoke, full evidence,
  performance baselines, and release audit.

## Validated Claim Surface

- `stable`: `parabolic`, `hyperbolic`, `mhd_ct`, `relativistic`
- `provisional`: `amr`, `coupling`
- `experimental` research tooling: `dashboard`, `io_extensions`

Publication-grade scientific claims in `v2` attach only to features marked
`stable` in the capability matrix and validation manifest.

## Breaking and Contract Changes

- CPU `Float64` is the publication baseline unless a feature explicitly states
  otherwise in the evidence contract.
- GPU execution does not inherit CPU claim status automatically; parity
  evidence is required first.
- Legacy convenience wrappers remain available as migration helpers, but the
  canonical execution path is now the SciML interface.

## Release Validation Status

The full local release-candidate lane stack passed during RC review:

- `fast-api-interop`
- `scientific-smoke`
- `performance`
- `full-evidence`
- `release-audit`

The release-audit output included:

- 12 executed evidence summaries
- 4 stable feature bundles
- `provenance.toml`
- `replay_report.toml`
- `performance_report.toml`
- `backend_parity_report.toml`
- `reports/validation_report.md`

See:
- [CHANGELOG.md](https://github.com/cx-xd/FiniteVolumeMethod.jl/blob/v2.0.0-rc1/CHANGELOG.md)
- [validation/V2_RC1_STATUS.md](https://github.com/cx-xd/FiniteVolumeMethod.jl/blob/v2.0.0-rc1/validation/V2_RC1_STATUS.md)
- [validation/RELEASE_CANDIDATE_DRY_RUN.md](https://github.com/cx-xd/FiniteVolumeMethod.jl/blob/v2.0.0-rc1/validation/RELEASE_CANDIDATE_DRY_RUN.md)
- [docs/src/v2_migration.md](https://github.com/cx-xd/FiniteVolumeMethod.jl/blob/v2.0.0-rc1/docs/src/v2_migration.md)

## Notes for RC Review

- GitHub-hosted Actions remain intentionally disabled during the RC review
  window. The local lane stack is the authoritative release process for this
  RC.
- The current research claim boundary is intentionally CPU-first and narrow for
  backend parity.
- Feedback should focus on the validated stable surface, release artifacts, and
  the `v2` migration path.

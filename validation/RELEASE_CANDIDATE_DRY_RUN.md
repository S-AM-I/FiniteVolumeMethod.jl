# Release Candidate Dry Run

This note records the full local release-candidate dry run completed on
March 15, 2026.

## Commands Run

The dry run exercised the full local lane stack through
`scripts/run_ci_lane.jl`:

- `fast-api-interop`
- `scientific-smoke`
- `performance`
- `full-evidence`
- `release-audit --output-root=/tmp/fvm-release-candidate`

The corresponding public operator entrypoints remain:

- `make ci-fast`
- `make ci-smoke`
- `make ci-performance`
- `make ci-full-evidence`
- `make ci-release-audit`

## Results

- `fast-api-interop`: pass
- `scientific-smoke`: pass, `43/43`
- `performance`: pass, `7/7`
- `full-evidence`: pass, `181/181`
- `release-audit`: pass, `129/129`

The release-audit output tree contained:

- 12 executed evidence summaries
- 4 stable feature bundles: `hyperbolic`, `mhd_ct`, `parabolic`,
  `relativistic`
- `provenance.toml`
- `replay_report.toml`
- `performance_report.toml`
- `backend_parity_report.toml`

## Notes

- No release blockers were found in this dry run.
- The bound `reference_datasets` artifact was recorded in release provenance.
- `backend_parity_report.toml` reported the CUDA parity case as `not_run` on
  the dry-run machine because CUDA was not functional there. Under the current
  claim boundary, that is acceptable and does not block a CPU-reference release.

## Follow-up

- Use `scripts/calibrate_performance_baselines.jl` after significant Julia,
  dependency, or hardware changes before tightening performance thresholds.
- Keep GitHub-hosted Actions disabled until the re-enable criteria in
  `validation/CI_REENABLE_PLAN.md` are satisfied.

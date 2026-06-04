# Research Release Checklist

Use this checklist before cutting a release that will be presented as
research-grade.

- All stable claim-bearing features are present in `validation/manifest.toml`.
- Each stable claim-bearing feature has automated scientific evidence.
- Each stable claim-bearing feature has documented limitations.
- Each stable claim-bearing feature has at least one maintained generated page.
- Exclusions and demotions are declared explicitly in `validation/manifest.toml`.
- `validation/generate_report.jl` produces the release validation report from a
  clean environment.
- `scripts/verification_validation_report.jl` regenerates executed evidence
  summaries for the release report.
- `scripts/build_reproduction_bundles.jl` refreshes per-feature reproduction
  bundles for archival and review.
- `scripts/build_release_outputs.jl --stable-only` produces the release-style
  output tree, including the top-level index, bundles, summaries, report,
  `provenance.toml`, and `replay_report.toml`.
- Release provenance records the bound `reference_datasets` artifact used for
  the long-lived benchmark corpus.
- Selected stable evidence summaries are replay-checked successfully during
  release-output generation, or the replay step is explicitly disabled and
  justified for that release candidate.
- `make ci-release-audit` (or the equivalent `scripts/run_ci_lane.jl
  release-audit`) passes locally before the release is considered ready.
- `make ci-release-audit` includes the stable-family performance baseline checks
  from `validation/performance_baselines.toml` with no large regressions.
- `make ci-release-audit` writes `performance_report.toml` and
  `backend_parity_report.toml`; any hard performance regression fails the gate,
  while CUDA parity may remain `not_run` on machines without a functional GPU.
- `julia --project=test scripts/calibrate_performance_baselines.jl` has been
  rerun after major Julia,
  dependency, or hardware changes if performance thresholds were expected to
  move.
- `test/release_audit.jl` passes and reports no missing stable-feature evidence,
  limitations, maintained pages, or release-output artifacts.
- `test/environment_integrity.jl` and `docs/environment_integrity.jl` pass.
- Documentation and README language match the current capability matrix and
  evidence status.
- The most recent dry-run results are recorded in CHANGELOG.md and the release tag's notes.

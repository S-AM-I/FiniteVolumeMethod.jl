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
- Selected stable evidence summaries are replay-checked successfully during
  release-output generation, or the replay step is explicitly disabled and
  justified for that release candidate.
- `test/environment_integrity.jl` and `docs/environment_integrity.jl` pass.
- Documentation and README language match the current capability matrix and
  evidence status.

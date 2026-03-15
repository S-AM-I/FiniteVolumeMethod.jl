# Release Candidate Dry Run

This note records the full local release-candidate dry run completed on
March 15, 2026.

## Environment

The dry run used a clean detached worktree at `/tmp/fvm-rc-check` together with
the Julia binary
`/Users/sami/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`
and
`JULIA_DEPOT_PATH=/tmp/fvm-julia-depot:/Users/sami/.julia`.

A truly empty depot could not be exercised inside this sandbox because registry
downloads are blocked by network policy. The clean-worktree plus writable overlay
depot setup was used to keep the RC pass isolated without introducing false
package-resolution failures unrelated to the repository.

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
- A subsequent full docs build from the clean worktree surfaced RC docs
  regressions that were fixed immediately:
  duplicate `@docs` entries for the new SciML interface helpers, a Makie
  `triplot!` attribute rename in `docs/src/math.md`, and AMR verification pages
  that needed a build-safe path to `amr_common.jl`.

## Follow-up

- Use `scripts/calibrate_performance_baselines.jl` after significant Julia,
  dependency, or hardware changes before tightening performance thresholds.
- Keep GitHub-hosted Actions disabled until the re-enable criteria in
  `validation/CI_REENABLE_PLAN.md` are satisfied.
- Confirm the patched full docs build as the last RC sign-off step before
  calling the branch ready for `v2.0.0-rc1`.

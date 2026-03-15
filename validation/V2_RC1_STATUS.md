# v2.0.0-rc1 Status

This note records the current release-candidate status for the research-grade
`v2` overhaul as of March 15, 2026.

## Environment Used

- Repository checkout: clean detached worktree at `/tmp/fvm-rc-check`
- Julia binary:
  `/Users/sami/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`
- Depot configuration:
  `JULIA_DEPOT_PATH=/tmp/fvm-julia-depot:/Users/sami/.julia`

## Important Limitation

A truly fresh depot could not be exercised inside this sandbox because registry
downloads are blocked by network restrictions. The RC pass therefore used a
clean worktree plus a writable overlay depot layered over the local user depot.
That still validates the repo from an isolated checkout while avoiding false
package-resolution failures unrelated to the codebase itself.

## Lane Status

The full local lane stack passed from the clean worktree:

- `fast-api-interop`: pass
- `scientific-smoke`: pass, `43/43`
- `performance`: pass, `7/7`
- `full-evidence`: pass, `181/181`
- `release-audit`: pass, `129/129`

The release-audit output tree included:

- 12 executed evidence summaries
- 4 stable feature bundles:
  `hyperbolic`, `mhd_ct`, `parabolic`, `relativistic`
- `provenance.toml`
- `replay_report.toml`
- `performance_report.toml`
- `backend_parity_report.toml`
- `reports/validation_report.md`

## Documentation Status

The first clean-worktree full docs build surfaced two release-candidate docs
blockers:

- duplicate `@docs` entries for `sciml_problem`, `solution_accessor`, and
  `solution_snapshot`
- Makie API drift in `docs/src/math.md` from `show_all_points` /
  `point_color` to `show_points` / `markercolor`

The patched rerun then surfaced one more docs-build issue in the new AMR
verification pages:

- the generated pages executed from `docs/build/verification`, but they used a
  bare `include("amr_common.jl")` instead of a path that also resolves from the
  generated `../literate_verification/` location

All three docs issues have now been fixed in the main checkout.

A full docs build from the docs environment now completes successfully with a
build-only CI-mode invocation:

- `julia --project=docs -e 'ENV["CI"] = "true"; ENV["FVM_DOCS_EXECUTION"] = "none"; include("docs/make.jl")'`

That build regenerated the tracked `docs/src` pages and produced `docs/build/index.html`.
The remaining docs messages were warnings only:

- Makie deprecation warning for `arrows`
- several pre-existing “Unexpected Julia interpolation in the Markdown” warnings
  in generated tutorial pages
- HTML/search-index size warnings for the large interface and search outputs

## Current Blocker State

- Local validation, performance, release-audit, and docs build: no blocking failures found
- GitHub-hosted Actions: intentionally remain disabled through RC1 review

## Recommendation

Treat the branch as release-candidate ready for `v2.0.0-rc1`, subject only to
the normal human release review of the generated bundles, validation report, and
regenerated docs sources.

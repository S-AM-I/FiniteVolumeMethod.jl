# v2 Migration

FiniteVolumeMethod.jl now exposes an explicit research-grade `v2` contract. If
you have used earlier revisions of the repository, the main change is that
solver APIs, scientific claims, and release operations are now all tied to the
same validation manifest.

## What Changed

- Scientific claims are now manifest-driven. The authoritative contract lives in
  the capability matrix and the validation manifest, not in scattered tutorials
  or README prose.
- Only `stable` claim-bearing solver families may support publication-grade
  claims. Today that set is `parabolic`, `hyperbolic`, `mhd_ct`, and
  `relativistic`.
- `amr` and `coupling` remain `provisional`. They have automated evidence, but
  they are still limited to the narrower cases recorded in the manifest.
- Dashboard and archival I/O paths are treated as research-support tooling, not
  solver validation.

## Execution Contract

- The canonical execution path is the SciML interface:
  construct a problem, derive the SciML problem with `sciml_problem(prob)`, and
  use `remake`, `init`, and `solve` through that path.
- Legacy convenience helpers such as older `solve_*` entrypoints remain a
  migration layer only. New research workflows should use the canonical SciML
  contract and the documented solution accessors.
- CPU `Float64` runs are the publication baseline. Other precision modes or
  backends need their own parity evidence before they should inherit the same
  claim level.

## Backend Claim Boundary

- The repo currently ships an executable CUDA parity check only for the
  supported 2D hyperbolic extension path.
- A release audit may therefore report backend parity as `not_run` on machines
  without a functional CUDA setup. That is expected and does not extend the
  public claim boundary.
- Until broader parity coverage exists, GPU execution should be treated as
  experimental outside the specific audited CUDA path.

## Release Operations

- Use the local lane runner rather than cloud CI while the overhaul remains in
  progress:
  `make ci-fast`, `make ci-smoke`, `make ci-full-evidence`,
  `make ci-performance`, and `make ci-release-audit`.
- `make ci-release-audit` is the release-blocking path. It builds stable release
  outputs, checks replay summaries, runs performance baselines, and records
  provenance plus backend-parity status.
- `julia --project=. scripts/build_release_outputs.jl --stable-only` produces
  the archival release tree used by the release audit.
- `julia --project=test scripts/calibrate_performance_baselines.jl` reruns the
  stable performance suite repeatedly so warning and fail thresholds can be
  reviewed after hardware, Julia, or dependency changes.

## User Checklist

- Check the capability matrix before treating a solver family as publication
  grade.
- Prefer the canonical SciML execution path for new code.
- Treat provisional and experimental features as development surfaces, not
  publication surfaces.
- Use the release audit and release outputs when preparing archived scientific
  results.

# GitHub Actions Re-enable Plan

Cloud GitHub Actions remain intentionally disabled while the `v2` overhaul is
still settling. The local lane runner is the canonical release process until the
conditions below are met.

## Preconditions

- The local lane stack is routinely green from a clean checkout:
  `ci-fast`, `ci-smoke`, `ci-full-evidence`, `ci-performance`, and
  `ci-release-audit`.
- The first `v2` release candidate has completed a full dry run with no open
  release blockers.
- The release-output tree, provenance, replay report, performance report, and
  backend-parity report have all been reviewed as archive-grade artifacts.
- Docs deployment is ready to stay gated on green scientific and release lanes.

## Proposed Cloud Mapping

- Pull requests:
  - format check
  - fast API/interop lane
  - scientific smoke lane
- Nightly or manual:
  - full evidence lane
  - docs build without deployment unless required lanes are green
- Release candidates and tagged releases:
  - release-audit lane
  - stable release-output packaging
- Optional dedicated runners:
  - performance lane
  - CUDA parity lane, only on machines with maintained GPU support

## Guardrails

- Do not re-enable docs deployment until it depends on green required lanes.
- Keep any GPU-specific cloud lanes opt-in until backend parity coverage extends
  beyond the current supported CUDA hyperbolic path.
- Preserve the current local entrypoints even after cloud workflows return, so
  the repo continues to support offline and cost-controlled release practice.

## Recommendation

Keep `*.yml.disabled` workflow files disabled until after the first accepted
research-grade `v2` release candidate. When re-enabling, restore only the
minimum set of workflows needed for pull-request feedback and release gating,
then add heavier lanes incrementally.

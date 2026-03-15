# Research Governance

FiniteVolumeMethod.jl distinguishes between solver capabilities that may support
publication-grade scientific claims and features that exist primarily to support
research workflow, visualization, or data handling.

## Capability Roles

- `claim_bearing_solver`: a solver family that may eventually support
  publication-grade claims once it is both `stable` and backed by automated
  evidence.
- `research_support_tooling`: reproducibility or workflow infrastructure such as
  reporting, dashboards, checkpointing, or export utilities. These features are
  important to research practice but do not themselves constitute solver
  validation.
- `experimental_sandbox`: opt-in work that is intentionally outside the stable
  research contract.

## Claim Policy

- `stable` claim-bearing solver features are the only ones that may support
  publication-grade scientific claims.
- `provisional` claim-bearing solver features may be used for internal research
  and method development, but their evidence ladder is incomplete.
- `experimental` features are engineering-only unless promoted later.
- Features with declared required ladder stages must satisfy every stage in the
  validation manifest before their research contract is considered complete.

## Support Policy

- Release-supported Julia targets are the current stable Julia release and the
  current Julia LTS release.
- CPU `Float64` runs are the scientific reference baseline unless a feature's
  evidence explicitly states otherwise.
- GPU execution is treated as an extension path that must demonstrate parity
  against the CPU reference baseline before it can inherit the same claim level.

## Release Discipline

Stable claim-bearing solver features must have:

- a declared capability entry in the validation manifest
- automated scientific evidence
- all declared required evidence-ladder stages present in the manifest
- documented limitations
- at least one maintained generated tutorial or example page

Demoted, excluded, or manually reviewed cases remain in the repository only when
their status is declared explicitly in the validation manifest and generated
report.

## Evidence Recording

- Evidence entries may declare a canonical entrypoint, ladder stage, runtime
  tier, and summary requirement.
- The scientific evidence runner executes those entrypoints in isolation and
  writes machine-readable summaries for local and release-grade reporting.

## Reproduction Outputs

- `julia --project=. scripts/verification_validation_report.jl` regenerates the
  validation report together with executed evidence summaries in
  `validation/reports/`.
- `julia --project=. scripts/build_reproduction_bundles.jl` creates per-feature
  reproduction bundles in `validation/reproduction_bundles/`.
- `julia --project=. scripts/build_release_outputs.jl --stable-only` builds a
  release-style output tree with executed summaries, bundle indexes, per-feature
  bundles, and the validation report in `validation/release_outputs/`.
- Release outputs also record `provenance.toml` and `replay_report.toml` so the
  archived tree carries the git/Julia context and the selected summary-replay
  check that was used to validate the packaged evidence.
- Release provenance also records the bound `reference_datasets` artifact so the
  long-lived benchmark corpus is versioned alongside the release metadata.
- These bundles copy the exact machine-readable summaries and referenced figure
  artifacts needed to archive or review the current claim-bearing evidence set.

## Local CI Lanes

- `make ci-fast` runs the fast API/interop lane for environment integrity,
  SciML contract coverage, semidiscrete adapters, and repository governance.
- `make ci-smoke` runs one verification-grade evidence case per stable
  claim-bearing solver family.
- `make ci-full-evidence` runs the complete scientific evidence catalog.
- `make ci-performance` runs the stable-family performance baselines for
  `hyperbolic`, `parabolic`, `mhd_ct`, and `relativistic`.
- `make ci-release-audit` runs the release-audit lane, including stable release
  output generation plus the executable release gate in
  `test/release_audit.jl`, using the same provenance, replay, reference-dataset
  artifact, stable-family performance, and optional CUDA backend-parity checks
  as the release-packaging workflow.

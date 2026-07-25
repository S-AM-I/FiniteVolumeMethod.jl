# Capability Matrix

The table below is generated directly from `validation/manifest.toml`, which is
the authoritative contract. It uses that file's vocabulary, reproduced here.

**Maturity** — how much validation evidence stands behind a capability:

- `stable` — verified (convergence studies, manufactured solutions) *and*
  validated against published benchmarks. Suitable for publication-grade
  results.
- `provisional` — automated verification exists, but V&V coverage is
  incomplete. Suitable for internal research and method development.
- `experimental` — available to develop against, but not covered by the
  package's validation claims. Not suitable for scientific claims.

Most of the package is `experimental`: of the capabilities listed below, only
four are `stable`. In particular the entire collocated stack is currently
`experimental` — its evidence items are real tests, but they are not yet
machine-linked evidence entries, so the governance ladder is not satisfied at
a higher maturity.

**Role** — what kind of thing the capability is, independent of its maturity:

- `claim_bearing_solver` — a solver whose results are intended to support
  scientific claims once its maturity allows.
- `experimental_sandbox` — research scaffolding; see the
  [Experimental](experimental/overview.md) section for honest per-module scope.
- `research_support_tooling` — I/O, dashboards, checkpointing and similar. These
  support the research workflow and are not part of the solver V&V.

The **V&V Coverage** column lists the verification stages that must be present
in the validation manifest for a feature to be considered fully verified at its
claimed maturity.

```@eval
using FiniteVolumeMethod
include(joinpath(dirname(pathof(FiniteVolumeMethod)), "..", "validation", "manifest.jl"))
using .RepoValidationManifest
using Markdown

manifest = RepoValidationManifest.load_manifest(joinpath(dirname(pathof(FiniteVolumeMethod)), "..", "validation", "manifest.toml"))
rows = RepoValidationManifest.capability_rows(manifest)
header = "| Capability | Category | Maturity | Validation Status | V&V Method | Solver Family | V&V Coverage | Notes | Limitations |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
body = join(
    [
        "| `$(row.feature)` | `$(row.role)` | `$(row.maturity)` | `$(row.claim_policy)` | `$(row.validation)` | `$(row.solver_family)` | $(isempty(row.required_ladder_stages) ? "n/a" : row.required_ladder_stages) | $(row.summary) | $(row.limitations) |"
            for row in rows
    ],
    "\n",
)
Markdown.parse(header * body)
```

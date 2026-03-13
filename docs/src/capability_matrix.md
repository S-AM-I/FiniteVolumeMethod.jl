# Capability Matrix

The table below is the public capability contract for this repository.

- `stable` claim-bearing solver features are the only ones that may support publication-grade scientific claims.
- `provisional` claim-bearing solver features are suitable for internal research and method development, but not strong external claims.
- `research_support_tooling` features improve reproducibility and workflow quality, but they do not constitute solver validation on their own.
- When a feature declares a required evidence ladder, every listed stage must be present in the validation manifest before that feature is treated as fully covered by the research contract.

```@eval
using FiniteVolumeMethod
include(joinpath(dirname(pathof(FiniteVolumeMethod)), "..", "validation", "manifest.jl"))
using .RepoValidationManifest
using Markdown

manifest = RepoValidationManifest.load_manifest(joinpath(dirname(pathof(FiniteVolumeMethod)), "..", "validation", "manifest.toml"))
rows = RepoValidationManifest.capability_rows(manifest)
header = "| Capability | Role | Maturity | Claim Policy | Validation | Solver Family | Required Ladder | Notes | Limitations |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
body = join(
    [
        "| `$(row.feature)` | `$(row.role)` | `$(row.maturity)` | `$(row.claim_policy)` | `$(row.validation)` | `$(row.solver_family)` | $(isempty(row.required_ladder_stages) ? "n/a" : row.required_ladder_stages) | $(row.summary) | $(row.limitations) |"
            for row in rows
    ],
    "\n",
)
Markdown.parse(header * body)
```

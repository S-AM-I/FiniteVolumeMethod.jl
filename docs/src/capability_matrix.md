# Capability Matrix

The table below summarises the verification and validation status of each
solver capability in the package.

- **Stable** solvers have been verified (convergence studies, manufactured
  solutions) and validated against published benchmarks. Suitable for
  publication-grade results.
- **Provisional** solvers have automated verification but incomplete V&V
  coverage. Suitable for internal research and method development.
- **Tooling** features (I/O, dashboards, checkpointing) support the research
  workflow but are not part of the solver V&V.
- The "V&V Coverage" column lists the verification stages that must be present
  in the validation manifest for a feature to be considered fully verified.

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

# Capability Matrix

The table below is the public capability contract for this repository. `stable`
features are release-grade and must remain covered by automated scientific
evidence. `provisional` features are supported but may change as the
architecture matures. `experimental` features are opt-in and are not covered by
the same compatibility guarantees.

```@eval
using FiniteVolumeMethod
include(joinpath(dirname(pathof(FiniteVolumeMethod)), "..", "validation", "manifest.jl"))
using .RepoValidationManifest
using Markdown

manifest = RepoValidationManifest.load_manifest(joinpath(dirname(pathof(FiniteVolumeMethod)), "..", "validation", "manifest.toml"))
rows = RepoValidationManifest.capability_rows(manifest)
header = "| Capability | Maturity | Validation | Notes |\n| --- | --- | --- | --- |\n"
body = join(
    [
        "| `$(row.feature)` | `$(row.maturity)` | `$(row.validation)` | $(row.summary) |"
            for row in rows
    ],
    "\n",
)
Markdown.parse(header * body)
```

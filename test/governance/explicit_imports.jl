using ExplicitImports
using FiniteVolumeMethod
using Test

# RepoValidationManifest is loaded via a dynamic `include` in
# `src/capabilities.jl` and cannot be statically analysed by
# ExplicitImports. The dynamic include is intentional (the manifest is
# a runtime artefact loaded once on package init) so we whitelist the
# submodule rather than restructure capabilities.jl.
const _UNANALYZABLE = (
    FiniteVolumeMethod,
    FiniteVolumeMethod.RepoValidationManifest,
)
# `FiniteVolumeMethod` and the embedded `RepoValidationManifest`
# submodule are flagged unanalyzable because of the dynamic include
# of validation/manifest.jl in capabilities.jl. Whitelist both.
@test check_no_implicit_imports(
    FiniteVolumeMethod;
    allow_unanalyzable = _UNANALYZABLE,
) === nothing
@test check_no_stale_explicit_imports(
    FiniteVolumeMethod;
    allow_unanalyzable = _UNANALYZABLE,
) === nothing

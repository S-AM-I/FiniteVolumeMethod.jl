using ExplicitImports
using FiniteVolumeMethod
using Test

# RepoValidationManifest is loaded via a dynamic `include` in
# `src/capabilities.jl` and cannot be statically analysed by
# ExplicitImports. The dynamic include is intentional (the manifest is
# a runtime artefact loaded once on package init) so we whitelist the
# submodule rather than restructure capabilities.jl.
@test check_no_implicit_imports(FiniteVolumeMethod) === nothing
@test check_no_stale_explicit_imports(
    FiniteVolumeMethod;
    allow_unanalyzable = (FiniteVolumeMethod.RepoValidationManifest,),
) === nothing

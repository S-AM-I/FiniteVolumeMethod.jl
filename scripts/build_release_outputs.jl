using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function parse_args(args)
    output_root = joinpath(REPO_ROOT, "validation", "release_outputs")
    rerun_evidence = true
    features = Symbol[]
    stable_only = false
    for arg in args
        if startswith(arg, "--feature=")
            push!(features, Symbol(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--output-root=")
            output_root = normpath(split(arg, "=", limit = 2)[2])
        elseif arg == "--reuse-summaries"
            rerun_evidence = false
        elseif arg == "--stable-only"
            stable_only = true
        else
            error("Unknown argument: $arg")
        end
    end
    return (; output_root, rerun_evidence, features, stable_only)
end

options = parse_args(ARGS)
if options.rerun_evidence
    Pkg.activate(joinpath(REPO_ROOT, "test"))
    Pkg.develop(PackageSpec(path = REPO_ROOT))
    Pkg.instantiate()
end

include(joinpath(REPO_ROOT, "validation", "manifest.jl"))
include(joinpath(REPO_ROOT, "validation", "reproducibility.jl"))
include(joinpath(REPO_ROOT, "validation", "release_packaging.jl"))
using .RepoReproducibility
using .RepoReleasePackaging
using .RepoValidationManifest

manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))
bundle_features = if !isempty(options.features)
    options.features
elseif options.stable_only
    RepoReproducibility.stable_claim_bundle_features(manifest)
else
    RepoReproducibility.bundle_features(manifest)
end

outputs = RepoReleasePackaging.build_release_outputs(
    manifest;
    repo_root = REPO_ROOT,
    output_root = options.output_root,
    bundle_features,
    rerun_evidence = options.rerun_evidence,
)

println("Release outputs written to $(outputs.output_root)")
println(" - report: $(outputs.report_path)")
println(" - bundle index: $(outputs.bundle_index_path)")
println(" - release index: $(outputs.release_index_path)")

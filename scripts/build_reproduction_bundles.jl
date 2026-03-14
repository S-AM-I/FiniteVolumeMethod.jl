using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function parse_args(args)
    output_root = joinpath(REPO_ROOT, "validation", "reproduction_bundles")
    summary_dir = nothing
    rerun_evidence = true
    features = Symbol[]
    for arg in args
        if startswith(arg, "--feature=")
            push!(features, Symbol(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--output-root=")
            output_root = normpath(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--summary-dir=")
            summary_dir = normpath(split(arg, "=", limit = 2)[2])
        elseif arg == "--reuse-summaries"
            rerun_evidence = false
        else
            error("Unknown argument: $arg")
        end
    end
    return (; output_root, summary_dir, rerun_evidence, features)
end

options = parse_args(ARGS)
if options.rerun_evidence
    Pkg.activate(joinpath(REPO_ROOT, "test"))
    Pkg.develop(PackageSpec(path = REPO_ROOT))
    Pkg.instantiate()
end

include(joinpath(REPO_ROOT, "validation", "reproducibility.jl"))
include(joinpath(REPO_ROOT, "validation", "manifest.jl"))
using .RepoReproducibility
using .RepoValidationManifest

manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))
features = isempty(options.features) ? RepoReproducibility.bundle_features(manifest) : options.features
bundles = RepoReproducibility.build_reproduction_bundles(
    manifest;
    repo_root = REPO_ROOT,
    output_root = options.output_root,
    features,
    rerun_evidence = options.rerun_evidence,
    summary_dir = options.summary_dir,
)

println("Built $(length(bundles)) reproduction bundle(s) in $(options.output_root)")
for bundle in bundles
    println(" - $(bundle.feature): $(bundle.bundle_dir)")
end

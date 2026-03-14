using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function parse_args(args)
    output_root = joinpath(REPO_ROOT, "validation", "release_outputs")
    rerun_evidence = true
    run_summary_replay = true
    features = Symbol[]
    replay_entry_ids = String[]
    stable_only = false
    for arg in args
        if startswith(arg, "--feature=")
            push!(features, Symbol(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--output-root=")
            output_root = normpath(split(arg, "=", limit = 2)[2])
        elseif arg == "--reuse-summaries"
            rerun_evidence = false
        elseif arg == "--no-replay"
            run_summary_replay = false
        elseif startswith(arg, "--replay-entry=")
            push!(replay_entry_ids, split(arg, "=", limit = 2)[2])
        elseif arg == "--stable-only"
            stable_only = true
        else
            error("Unknown argument: $arg")
        end
    end
    return (; output_root, rerun_evidence, run_summary_replay, replay_entry_ids, features, stable_only)
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
    run_summary_replay = options.run_summary_replay,
    replay_entry_ids = isempty(options.replay_entry_ids) ?
        RepoReleasePackaging.default_replay_entry_ids(manifest, bundle_features) :
        options.replay_entry_ids,
)

println("Release outputs written to $(outputs.output_root)")
println(" - report: $(outputs.report_path)")
println(" - bundle index: $(outputs.bundle_index_path)")
println(" - release index: $(outputs.release_index_path)")
println(" - provenance: $(outputs.provenance_path)")
println(" - replay report: $(outputs.replay_report_path)")

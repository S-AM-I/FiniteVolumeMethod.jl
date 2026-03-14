module RepoReproducibility

using Dates
using TOML

include(joinpath(@__DIR__, "manifest.jl"))
include(joinpath(@__DIR__, "evidence_runner.jl"))
using .RepoEvidenceRunner
using .RepoValidationManifest

const DEFAULT_BUNDLE_ROOT = joinpath(@__DIR__, "reproduction_bundles")

default_bundle_root() = DEFAULT_BUNDLE_ROOT

function bundle_features(manifest)
    return sort!(
        collect(Set(entry.feature for entry in manifest.scientific_evidence));
        by = string,
    )
end

function stable_claim_bundle_features(manifest)
    return sort!(
        [
            feature for (feature, entry) in manifest.features
                if entry.maturity == :stable && entry.role == :claim_bearing_solver
        ];
        by = string,
    )
end

function build_reproduction_bundles(
        manifest;
        repo_root::AbstractString,
        output_root::AbstractString = DEFAULT_BUNDLE_ROOT,
        features = bundle_features(manifest),
        rerun_evidence::Bool = true,
        summary_dir::Union{Nothing, AbstractString} = nothing,
    )
    mkpath(output_root)
    bundles = [
        build_feature_bundle(
                manifest,
                feature;
                repo_root,
                output_root,
                rerun_evidence,
                summary_dir,
            ) for feature in features
    ]
    write_bundle_index(output_root, bundles)
    return bundles
end

function build_feature_bundle(
        manifest,
        feature::Symbol;
        repo_root::AbstractString,
        output_root::AbstractString = DEFAULT_BUNDLE_ROOT,
        rerun_evidence::Bool = true,
        summary_dir::Union{Nothing, AbstractString} = nothing,
    )
    entries = RepoValidationManifest.scientific_evidence_for_feature(manifest, feature)
    isempty(entries) && throw(ArgumentError("No scientific evidence entries declared for feature `$feature`."))

    feature_entry = manifest.features[feature]
    bundle_dir = joinpath(output_root, string(feature))
    summaries_dir = joinpath(bundle_dir, "summaries")
    artifacts_dir = joinpath(bundle_dir, "artifacts")
    mkpath(summaries_dir)
    mkpath(artifacts_dir)

    summaries = _materialize_summaries(
        entries;
        repo_root,
        summaries_dir,
        rerun_evidence,
        source_summary_dir = summary_dir,
    )

    evidence_rows = Dict{String, Any}[]
    for entry in entries
        summary = summaries[entry.id]
        copied_artifacts, missing_artifacts = _copy_recorded_artifacts(
            entry,
            summary;
            repo_root,
            artifact_dir = artifacts_dir,
        )
        push!(
            evidence_rows,
            Dict{String, Any}(
                "id" => entry.id,
                "status" => get(summary, "status", "unknown"),
                "entrypoint" => entry.entrypoint,
                "summary_path" => relpath(joinpath(summaries_dir, "$(entry.id).toml"), bundle_dir),
                "recorded_result_count" => get(summary, "recorded_result_count", 0),
                "copied_artifacts" => copied_artifacts,
                "missing_artifacts" => missing_artifacts,
            ),
        )
    end

    bundle_manifest = Dict{String, Any}(
        "bundle_version" => 1,
        "generated_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "feature" => string(feature),
        "role" => string(feature_entry.role),
        "maturity" => string(feature_entry.maturity),
        "validation" => string(feature_entry.validation),
        "claim_policy" => string(_feature_claim_policy(feature_entry)),
        "solver_family" => isnothing(feature_entry.solver_family) ? "n_a" : string(feature_entry.solver_family),
        "precision_policy" => isnothing(feature_entry.precision_policy) ? "n_a" : string(feature_entry.precision_policy),
        "random_seed_policy" => isnothing(feature_entry.random_seed_policy) ? "n_a" : string(feature_entry.random_seed_policy),
        "backend_policy" => isnothing(feature_entry.backend_policy) ? "n_a" : string(feature_entry.backend_policy),
        "required_ladder_stages" => string.(feature_entry.required_ladder_stages),
        "summary" => feature_entry.summary,
        "limitations" => copy(feature_entry.limitations),
        "source_manifest" => "validation/manifest.toml",
        "evidence" => evidence_rows,
    )

    bundle_manifest_path = joinpath(bundle_dir, "bundle_manifest.toml")
    open(bundle_manifest_path, "w") do io
        TOML.print(io, bundle_manifest)
    end

    bundle_readme_path = joinpath(bundle_dir, "README.md")
    write(bundle_readme_path, _bundle_readme(feature, feature_entry, evidence_rows))

    return (
        feature = feature,
        bundle_dir = bundle_dir,
        bundle_manifest_path = bundle_manifest_path,
        bundle_readme_path = bundle_readme_path,
        summaries_dir = summaries_dir,
        artifacts_dir = artifacts_dir,
        evidence = evidence_rows,
    )
end

function _materialize_summaries(
        entries;
        repo_root::AbstractString,
        summaries_dir::AbstractString,
        rerun_evidence::Bool,
        source_summary_dir::Union{Nothing, AbstractString},
    )
    if rerun_evidence
        run_summaries = RepoEvidenceRunner.run_evidence_entries(entries; repo_root, output_dir = summaries_dir)
        return Dict(summary["id"] => summary for summary in run_summaries)
    end

    load_dir = isnothing(source_summary_dir) ? summaries_dir : source_summary_dir
    summaries = Dict{String, Dict{String, Any}}()
    for entry in entries
        source = joinpath(load_dir, "$(entry.id).toml")
        isfile(source) || throw(ArgumentError("Missing evidence summary for `$(entry.id)` at `$source`."))
        destination = joinpath(summaries_dir, "$(entry.id).toml")
        source != destination && cp(source, destination; force = true)
        summaries[entry.id] = TOML.parsefile(destination)
    end
    return summaries
end

function _copy_recorded_artifacts(entry, summary; repo_root::AbstractString, artifact_dir::AbstractString)
    copied = String[]
    missing = String[]
    for artifact in _artifact_names(summary)
        source = _find_artifact_path(repo_root, artifact, summary)
        if isnothing(source)
            push!(missing, artifact)
            continue
        end
        destination = joinpath(artifact_dir, basename(artifact))
        cp(source, destination; force = true)
        push!(copied, relpath(destination, dirname(artifact_dir)))
    end
    sort!(copied)
    sort!(missing)
    return copied, missing
end

function _artifact_names(summary)
    names = Set{String}()
    for record in get(summary, "recorded_results", Any[])
        for artifact in get(record, "artifacts", Any[])
            artifact_name = String(artifact)
            _is_materialized_artifact(artifact_name) && push!(names, artifact_name)
        end
    end
    return sort!(collect(names))
end

_is_materialized_artifact(name::AbstractString) =
    lowercase(name) != "summary_metrics" &&
    lowercase(name) != "reference_figure"

function _find_artifact_path(repo_root::AbstractString, artifact::AbstractString, summary)
    artifact_output_dir = get(summary, "artifact_output_dir", nothing)
    candidates = (
        isnothing(artifact_output_dir) ? nothing : joinpath(artifact_output_dir, basename(artifact)),
        joinpath(repo_root, artifact),
        joinpath(repo_root, "docs", "src", "figures", basename(artifact)),
    )
    for candidate in candidates
        !isnothing(candidate) && isfile(candidate) && return candidate
    end
    return nothing
end

function _bundle_readme(feature::Symbol, feature_entry, evidence_rows)
    io = IOBuffer()
    println(io, "# $(string(feature)) Reproduction Bundle")
    println(io)
    println(io, "- **Feature:** `$(feature)`")
    println(io, "- **Maturity:** `$(feature_entry.maturity)`")
    println(io, "- **Role:** `$(feature_entry.role)`")
    println(io, "- **Claim policy:** `$(_feature_claim_policy(feature_entry))`")
    println(io, "- **Summary:** $(feature_entry.summary)")
    println(io)
    println(io, "## Reproduce")
    println(io)
    println(io, "Run the bundle generator from the repository root:")
    println(io)
    println(io, "```bash")
    println(io, "julia --project=. scripts/build_reproduction_bundles.jl --feature=$(feature)")
    println(io, "```")
    println(io)
    println(io, "## Evidence")
    println(io)
    println(io, "| ID | Status | Summary | Artifacts | Entrypoint |")
    println(io, "|----|--------|---------|-----------|------------|")
    for row in evidence_rows
        artifacts = isempty(row["copied_artifacts"]) ? "n/a" : join(row["copied_artifacts"], ", ")
        println(
            io,
            "| $(row["id"]) | $(row["status"]) | $(row["summary_path"]) | $artifacts | $(row["entrypoint"]) |",
        )
    end
    println(io)
    return String(take!(io))
end

function write_bundle_index(output_root::AbstractString, bundles)
    index_manifest = Dict{String, Any}(
        "bundle_index_version" => 1,
        "generated_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "bundle_count" => length(bundles),
        "bundles" => [
            Dict{String, Any}(
                    "feature" => string(bundle.feature),
                    "bundle_dir" => relpath(bundle.bundle_dir, output_root),
                    "bundle_manifest" => relpath(bundle.bundle_manifest_path, output_root),
                    "bundle_readme" => relpath(bundle.bundle_readme_path, output_root),
                    "artifact_count" => isdir(bundle.artifacts_dir) ? length(readdir(bundle.artifacts_dir)) : 0,
                    "summary_count" => isdir(bundle.summaries_dir) ? length(filter(name -> endswith(name, ".toml"), readdir(bundle.summaries_dir))) : 0,
                ) for bundle in bundles
        ],
    )
    open(joinpath(output_root, "bundle_index.toml"), "w") do io
        TOML.print(io, index_manifest)
    end

    io = IOBuffer()
    println(io, "# Reproduction Bundles")
    println(io)
    println(io, "| Feature | Bundle README | Bundle Manifest | Summaries | Artifacts |")
    println(io, "|---------|---------------|-----------------|-----------|-----------|")
    for bundle in bundles
        summary_count = isdir(bundle.summaries_dir) ? length(filter(name -> endswith(name, ".toml"), readdir(bundle.summaries_dir))) : 0
        artifact_count = isdir(bundle.artifacts_dir) ? length(readdir(bundle.artifacts_dir)) : 0
        println(
            io,
            "| $(bundle.feature) | [`README.md`]($(relpath(bundle.bundle_readme_path, output_root))) | [`bundle_manifest.toml`]($(relpath(bundle.bundle_manifest_path, output_root))) | $summary_count | $artifact_count |",
        )
    end
    write(joinpath(output_root, "README.md"), String(take!(io)))
    return nothing
end

function _feature_claim_policy(entry)
    return if entry.role == :claim_bearing_solver
        if entry.maturity == :stable
            :publishable_scientific_claim
        elseif entry.maturity == :provisional
            :internal_research_only
        else
            :engineering_only
        end
    elseif entry.role == :research_support_tooling
        :reproducibility_infrastructure_only
    else
        :engineering_only
    end
end

end

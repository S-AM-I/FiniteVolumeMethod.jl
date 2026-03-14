module RepoReleasePackaging

using Dates
using SHA
using TOML

include(joinpath(@__DIR__, "manifest.jl"))
include(joinpath(@__DIR__, "evidence_runner.jl"))
include(joinpath(@__DIR__, "reproducibility.jl"))
include(joinpath(@__DIR__, "generate_report.jl"))
include(joinpath(@__DIR__, "summary_replay.jl"))
using .RepoEvidenceRunner
using .RepoReproducibility
using .RepoSummaryReplay
using .RepoValidationManifest
using .ValidationReport

const DEFAULT_RELEASE_OUTPUT_ROOT = joinpath(@__DIR__, "release_outputs")

default_release_output_root() = DEFAULT_RELEASE_OUTPUT_ROOT

function build_release_outputs(
        manifest;
        repo_root::AbstractString,
        output_root::AbstractString = DEFAULT_RELEASE_OUTPUT_ROOT,
        bundle_features = RepoReproducibility.stable_claim_bundle_features(manifest),
        rerun_evidence::Bool = true,
        run_summary_replay::Bool = true,
        replay_entry_ids = default_replay_entry_ids(manifest, bundle_features),
    )
    summaries_dir = joinpath(output_root, "summaries")
    bundles_dir = joinpath(output_root, "bundles")
    reports_dir = joinpath(output_root, "reports")
    report_path = joinpath(reports_dir, "validation_report.md")
    replay_dir = joinpath(output_root, "replay_summaries")
    replay_report_path = joinpath(output_root, "replay_report.toml")
    provenance_path = joinpath(output_root, "provenance.toml")
    mkpath(summaries_dir)
    mkpath(bundles_dir)
    mkpath(reports_dir)
    executed_entries = RepoValidationManifest.scientific_evidence_for_features(manifest, bundle_features)
    replay_entries = _replay_entries(executed_entries, replay_entry_ids)

    if rerun_evidence
        RepoEvidenceRunner.run_evidence_entries(
            executed_entries;
            repo_root,
            output_dir = summaries_dir,
        )
    end

    bundles = RepoReproducibility.build_reproduction_bundles(
        manifest;
        repo_root,
        output_root = bundles_dir,
        features = bundle_features,
        rerun_evidence = false,
        summary_dir = summaries_dir,
    )

    ValidationReport.generate(
        joinpath(repo_root, "validation", "manifest.toml"),
        report_path;
        summary_dir = summaries_dir,
        bundle_dir = bundles_dir,
        executed_entry_ids = [entry.id for entry in executed_entries],
    )

    replay_result = _run_summary_replay(
        summaries_dir,
        replay_dir,
        replay_entries;
        repo_root,
        run_summary_replay,
    )
    _write_replay_report(replay_report_path, replay_result)

    provenance = _release_provenance(
        repo_root,
        manifest,
        bundle_features,
        executed_entries,
        replay_result,
        rerun_evidence,
    )
    _write_provenance(provenance_path, provenance)

    _write_release_index(output_root, bundles, report_path, executed_entries, provenance_path, replay_report_path, replay_result)

    return (
        output_root = output_root,
        summaries_dir = summaries_dir,
        bundles_dir = bundles_dir,
        reports_dir = reports_dir,
        report_path = report_path,
        bundle_index_path = joinpath(bundles_dir, "bundle_index.toml"),
        release_index_path = joinpath(output_root, "README.md"),
        provenance_path = provenance_path,
        replay_report_path = replay_report_path,
        replay_summaries_dir = replay_dir,
        bundles = bundles,
    )
end

function default_replay_entry_ids(manifest, bundle_features)
    ids = String[]
    for feature in sort!(collect(bundle_features); by = string)
        entries = RepoValidationManifest.scientific_evidence_for_feature(manifest, feature)
        isempty(entries) && continue
        verification_entries = filter(entry -> entry.ladder_stage == :verification, entries)
        selected_entry = isempty(verification_entries) ? first(entries) : first(verification_entries)
        push!(ids, selected_entry.id)
    end
    return ids
end

function _write_release_index(output_root::AbstractString, bundles, report_path::AbstractString, executed_entries, provenance_path::AbstractString, replay_report_path::AbstractString, replay_result)
    summary_count = length(executed_entries)
    io = IOBuffer()
    println(io, "# Release Outputs")
    println(io)
    println(io, "- Generated: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")) (local time)")
    println(io, "- Validation report: [`$(basename(report_path))`](reports/$(basename(report_path)))")
    println(io, "- Evidence summaries: `$summary_count` TOML files in `summaries/`")
    println(io, "- Bundle index: [`bundle_index.toml`](bundles/bundle_index.toml)")
    println(io, "- Bundle overview: [`README.md`](bundles/README.md)")
    println(io, "- Provenance metadata: [`$(basename(provenance_path))`]($(basename(provenance_path)))")
    println(io, "- Summary replay report: [`$(basename(replay_report_path))`]($(basename(replay_report_path)))")
    println(io, "- Summary replay status: `$(replay_result.status)` across `$(length(replay_result.entry_ids))` selected evidence case(s)")
    println(io)
    println(io, "## Bundles")
    println(io)
    println(io, "| Feature | Bundle Directory | Artifacts | Summaries |")
    println(io, "|---------|------------------|-----------|-----------|")
    for bundle in bundles
        artifact_count = isdir(bundle.artifacts_dir) ? length(readdir(bundle.artifacts_dir)) : 0
        bundle_summary_count = isdir(bundle.summaries_dir) ? length(filter(name -> endswith(name, ".toml"), readdir(bundle.summaries_dir))) : 0
        println(
            io,
            "| $(bundle.feature) | [`$(bundle.feature)`](bundles/$(basename(bundle.bundle_dir))/README.md) | $artifact_count | $bundle_summary_count |",
        )
    end
    write(joinpath(output_root, "README.md"), String(take!(io)))
    return nothing
end

function _run_summary_replay(summaries_dir::AbstractString, replay_dir::AbstractString, replay_entries; repo_root::AbstractString, run_summary_replay::Bool)
    if !run_summary_replay || isempty(replay_entries)
        return (
            status = "not_run",
            entry_ids = String[],
            replay_dir = nothing,
            diffs = Dict{String, Vector{String}}(),
        )
    end
    mkpath(replay_dir)
    RepoEvidenceRunner.run_evidence_entries(replay_entries; repo_root, output_dir = replay_dir)
    entry_ids = [entry.id for entry in replay_entries]
    diffs = RepoSummaryReplay.compare_summary_directories(
        summaries_dir,
        replay_dir;
        entry_ids,
    )
    isempty(diffs) || error(_format_replay_diffs(diffs))
    return (
        status = "pass",
        entry_ids = entry_ids,
        replay_dir = replay_dir,
        diffs = diffs,
    )
end

function _write_replay_report(path::AbstractString, replay_result)
    report = Dict{String, Any}(
        "replay_version" => 1,
        "status" => replay_result.status,
        "entry_ids" => replay_result.entry_ids,
        "replay_dir" => isnothing(replay_result.replay_dir) ? "not_run" : replay_result.replay_dir,
        "diffs" => replay_result.diffs,
    )
    open(path, "w") do io
        TOML.print(io, report)
    end
    return nothing
end

function _release_provenance(repo_root, manifest, bundle_features, executed_entries, replay_result, rerun_evidence::Bool)
    validation_manifest_path = joinpath(repo_root, "validation", "manifest.toml")
    test_project_path = joinpath(repo_root, "test", "Project.toml")
    test_manifest_path = joinpath(repo_root, "test", "Manifest.toml")
    ordered_features = sort!(string.(collect(bundle_features)); by = identity)
    return Dict{String, Any}(
        "provenance_version" => 1,
        "generated_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "julia_version" => string(VERSION),
        "repo_root" => repo_root,
        "git_commit" => _git_output(repo_root, "rev-parse", "HEAD"),
        "git_branch" => _git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "manifest_version" => manifest.manifest_version,
        "support_policy" => string(manifest.support_policy),
        "bundle_features" => ordered_features,
        "executed_entry_ids" => [entry.id for entry in executed_entries],
        "rerun_evidence" => rerun_evidence,
        "run_summary_replay" => replay_result.status != "not_run",
        "replay_entry_ids" => replay_result.entry_ids,
        "validation_manifest_sha1" => bytes2hex(open(path -> sha1(read(path)), validation_manifest_path)),
        "test_project_sha1" => bytes2hex(open(path -> sha1(read(path)), test_project_path)),
        "test_manifest_sha1" => bytes2hex(open(path -> sha1(read(path)), test_manifest_path)),
    )
end

function _write_provenance(path::AbstractString, provenance)
    open(path, "w") do io
        TOML.print(io, provenance)
    end
    return nothing
end

function _replay_entries(executed_entries, replay_entry_ids)
    replay_id_set = Set(String.(collect(replay_entry_ids)))
    return [entry for entry in executed_entries if entry.id in replay_id_set]
end

function _git_output(repo_root::AbstractString, args::AbstractString...)
    cmd = Cmd(["git", "-C", repo_root, args...])
    return try
        readchomp(cmd)
    catch
        "unavailable"
    end
end

function _format_replay_diffs(diffs)
    io = IOBuffer()
    println(io, "Summary replay mismatches detected:")
    for id in sort!(collect(keys(diffs)); by = identity)
        println(io, "- $id")
        for diff in diffs[id]
            println(io, "  * $diff")
        end
    end
    return String(take!(io))
end

end

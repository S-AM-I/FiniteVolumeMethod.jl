module RepoReleaseAudit

using TOML

include(joinpath(@__DIR__, "manifest.jl"))
include(joinpath(@__DIR__, "reproducibility.jl"))
include(joinpath(@__DIR__, "release_packaging.jl"))
using .RepoReleasePackaging
using .RepoReproducibility
using .RepoValidationManifest

function audit_release(
        manifest;
        repo_root::AbstractString,
        output_root::AbstractString,
        rerun_evidence::Bool = true,
        run_summary_replay::Bool = true,
    )
    stable_features = RepoReproducibility.stable_claim_bundle_features(manifest)
    outputs = RepoReleasePackaging.build_release_outputs(
        manifest;
        repo_root,
        output_root,
        bundle_features = stable_features,
        rerun_evidence,
        run_summary_replay,
    )
    return summarize_release_audit(
        manifest,
        outputs;
        run_summary_replay,
    )
end

function summarize_release_audit(manifest, outputs; run_summary_replay::Bool = true)
    stable_features = RepoReproducibility.stable_claim_bundle_features(manifest)
    expected_feature_names = Set(string.(stable_features))
    expected_evidence_entries = RepoValidationManifest.scientific_evidence_for_features(manifest, stable_features)
    generated_page_features = Set(page.feature for page in manifest.generated_pages)

    bundle_index = TOML.parsefile(outputs.bundle_index_path)
    provenance = TOML.parsefile(outputs.provenance_path)
    replay_report = TOML.parsefile(outputs.replay_report_path)
    report = read(outputs.report_path, String)
    release_index = read(outputs.release_index_path, String)
    summary_files = filter(name -> endswith(name, ".toml"), readdir(outputs.summaries_dir))

    findings = String[]

    Set(entry["feature"] for entry in bundle_index["bundles"]) == expected_feature_names ||
        push!(findings, "bundle index does not match the stable claim-bearing feature set")
    Set(provenance["bundle_features"]) == expected_feature_names ||
        push!(findings, "provenance bundle_features does not match the stable claim-bearing feature set")
    provenance["manifest_version"] == manifest.manifest_version ||
        push!(findings, "provenance manifest_version does not match validation/manifest.toml")
    length(summary_files) == length(expected_evidence_entries) ||
        push!(findings, "release summaries count does not match the stable scientific evidence catalog")

    expected_replay_status = run_summary_replay ? "pass" : "not_run"
    replay_report["status"] == expected_replay_status ||
        push!(findings, "summary replay status is `$(replay_report["status"])`, expected `$expected_replay_status`")
    occursin("provenance.toml", release_index) ||
        push!(findings, "release index is missing the provenance reference")
    occursin("replay_report.toml", release_index) ||
        push!(findings, "release index is missing the replay report reference")
    occursin("## Executed Evidence Results", report) ||
        push!(findings, "validation report is missing the executed evidence section")
    occursin("## Reproduction Bundles", report) ||
        push!(findings, "validation report is missing the reproduction bundles section")

    for feature in stable_features
        feature_entry = manifest.features[feature]
        feature_name = string(feature)
        isempty(feature_entry.limitations) &&
            push!(findings, "stable feature `$feature_name` is missing documented limitations")
        isempty(RepoValidationManifest.scientific_evidence_for_feature(manifest, feature)) &&
            push!(findings, "stable feature `$feature_name` is missing scientific evidence")
        feature in generated_page_features ||
            push!(findings, "stable feature `$feature_name` is missing a maintained generated page")

        ladder = RepoValidationManifest.feature_ladder_coverage(manifest, feature)
        ladder.satisfied || push!(findings, "stable feature `$feature_name` is missing required evidence ladder stages")

        bundle_dir = joinpath(outputs.bundles_dir, feature_name)
        isfile(joinpath(bundle_dir, "bundle_manifest.toml")) ||
            push!(findings, "stable feature `$feature_name` is missing bundle_manifest.toml")
        isfile(joinpath(bundle_dir, "README.md")) ||
            push!(findings, "stable feature `$feature_name` is missing bundle README")
        isdir(joinpath(bundle_dir, "summaries")) ||
            push!(findings, "stable feature `$feature_name` is missing bundled summaries")
        isdir(joinpath(bundle_dir, "artifacts")) ||
            push!(findings, "stable feature `$feature_name` is missing bundled artifacts")
    end

    return (
        outputs = outputs,
        stable_features = stable_features,
        expected_evidence_entries = expected_evidence_entries,
        findings = findings,
        bundle_index = bundle_index,
        provenance = provenance,
        replay_report = replay_report,
        report = report,
        release_index = release_index,
    )
end

end

module RepoReleaseAudit

using TOML

include(joinpath(@__DIR__, "backend_parity.jl"))
include(joinpath(@__DIR__, "manifest.jl"))
include(joinpath(@__DIR__, "performance_baselines.jl"))
include(joinpath(@__DIR__, "reproducibility.jl"))
include(joinpath(@__DIR__, "release_packaging.jl"))
using .RepoBackendParity
using .RepoPerformanceBaselines
using .RepoReleasePackaging
using .RepoReproducibility
using .RepoValidationManifest

function audit_release(
        manifest;
        repo_root::AbstractString,
        output_root::AbstractString,
        rerun_evidence::Bool = true,
        run_summary_replay::Bool = true,
        run_performance_audit::Bool = true,
        run_backend_parity::Bool = true,
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
        run_performance_audit,
        run_backend_parity,
    )
end

function summarize_release_audit(
        manifest,
        outputs;
        run_summary_replay::Bool = true,
        run_performance_audit::Bool = true,
        run_backend_parity::Bool = true,
    )
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
    performance_audit = _performance_audit(outputs.output_root, run_performance_audit)
    backend_parity = _backend_parity_audit(outputs.output_root, run_backend_parity)

    findings = String[]
    warnings = String[]

    Set(entry["feature"] for entry in bundle_index["bundles"]) == expected_feature_names ||
        push!(findings, "bundle index does not match the stable claim-bearing feature set")
    Set(provenance["bundle_features"]) == expected_feature_names ||
        push!(findings, "provenance bundle_features does not match the stable claim-bearing feature set")
    provenance["manifest_version"] == manifest.manifest_version ||
        push!(findings, "provenance manifest_version does not match validation/manifest.toml")
    haskey(provenance, "reference_dataset_artifact") ||
        push!(findings, "provenance is missing the reference_dataset_artifact record")
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
    performance_audit.passed || append!(findings, performance_audit.findings)
    append!(warnings, performance_audit.warnings)
    backend_parity.passed || append!(findings, backend_parity.findings)
    append!(warnings, backend_parity.warnings)

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
        warnings = warnings,
        bundle_index = bundle_index,
        provenance = provenance,
        replay_report = replay_report,
        report = report,
        release_index = release_index,
        performance_audit = performance_audit,
        backend_parity = backend_parity,
    )
end

function _performance_audit(output_root::AbstractString, run_performance_audit::Bool)
    report_path = joinpath(output_root, "performance_report.toml")
    if !run_performance_audit
        result = (
            passed = true,
            findings = String[],
            warnings = String[],
            measurements = NamedTuple[],
            comparisons = NamedTuple[],
        )
        _write_performance_report(report_path, result)
        return result
    end

    measurements = RepoPerformanceBaselines.run_suite()
    baselines = RepoPerformanceBaselines.load_baselines()
    comparisons = RepoPerformanceBaselines.compare_to_baselines(measurements, baselines)
    findings = String[]
    warnings = String[]
    for comparison in comparisons
        for warning in comparison.warnings
            push!(warnings, "performance baseline `$(comparison.id)`: $warning")
        end
        comparison.passed && continue
        if isempty(comparison.reasons)
            push!(findings, "performance baseline `$(comparison.id)` failed without a reported reason")
        else
            for reason in comparison.reasons
                push!(findings, "performance baseline `$(comparison.id)`: $reason")
            end
        end
    end

    result = (
        passed = isempty(findings),
        findings = findings,
        warnings = warnings,
        measurements = measurements,
        comparisons = comparisons,
    )
    _write_performance_report(report_path, result)
    return result
end

function _backend_parity_audit(output_root::AbstractString, run_backend_parity::Bool)
    report_path = joinpath(output_root, "backend_parity_report.toml")
    if !run_backend_parity
        result = (
            passed = true,
            findings = String[],
            warnings = String[],
            summary = (
                status = :not_run,
                counts = Dict(:pass => 0, :fail => 0, :not_run => 0),
                results = NamedTuple[],
            ),
        )
        _write_backend_parity_report(report_path, result)
        return result
    end

    summary = RepoBackendParity.summarize_suite()
    findings = String[]
    warnings = String[]
    for result in summary.results
        if result.status == :fail
            isempty(result.failures) &&
                push!(findings, "backend parity `$(result.id)` failed without a reported reason")
            for failure in result.failures
                push!(findings, "backend parity `$(result.id)`: $failure")
            end
        elseif result.status == :not_run
            push!(warnings, "backend parity `$(result.id)` not run: $(result.rationale)")
        end
    end
    audit = (
        passed = isempty(findings),
        findings = findings,
        warnings = warnings,
        summary = summary,
    )
    _write_backend_parity_report(report_path, audit)
    return audit
end

function _write_performance_report(path::AbstractString, result)
    report = Dict{String, Any}(
        "performance_report_version" => 1,
        "status" => result.passed ? "pass" : "fail",
        "warnings" => copy(result.warnings),
        "failures" => copy(result.findings),
        "cases" => [
            Dict{String, Any}(
                    "id" => comparison.id,
                    "feature" => string(comparison.feature),
                    "status" => string(comparison.status),
                    "warning_only" => comparison.warning_only,
                    "release_blocker" => comparison.release_blocker,
                    "warnings" => copy(comparison.warnings),
                    "failures" => copy(comparison.reasons),
                ) for comparison in result.comparisons
        ],
    )
    open(path, "w") do io
        TOML.print(io, report)
    end
    return nothing
end

function _write_backend_parity_report(path::AbstractString, audit)
    report = Dict{String, Any}(
        "backend_parity_report_version" => 1,
        "status" => audit.passed ? "pass" : "fail",
        "warnings" => copy(audit.warnings),
        "failures" => copy(audit.findings),
        "counts" => Dict(
            "pass" => get(audit.summary.counts, :pass, 0),
            "fail" => get(audit.summary.counts, :fail, 0),
            "not_run" => get(audit.summary.counts, :not_run, 0),
        ),
        "cases" => [
            Dict{String, Any}(
                    "id" => result.id,
                    "feature" => string(result.feature),
                    "status" => string(result.status),
                    "backend" => result.backend,
                    "rationale" => result.rationale,
                    "metrics" => result.metrics,
                    "failures" => copy(result.failures),
                ) for result in audit.summary.results
        ],
    )
    open(path, "w") do io
        TOML.print(io, report)
    end
    return nothing
end

end

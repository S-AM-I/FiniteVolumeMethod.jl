module RepoReleasePackaging

using Dates

include(joinpath(@__DIR__, "manifest.jl"))
include(joinpath(@__DIR__, "evidence_runner.jl"))
include(joinpath(@__DIR__, "reproducibility.jl"))
include(joinpath(@__DIR__, "generate_report.jl"))
using .RepoEvidenceRunner
using .RepoReproducibility
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
    )
    summaries_dir = joinpath(output_root, "summaries")
    bundles_dir = joinpath(output_root, "bundles")
    reports_dir = joinpath(output_root, "reports")
    report_path = joinpath(reports_dir, "validation_report.md")
    mkpath(summaries_dir)
    mkpath(bundles_dir)
    mkpath(reports_dir)

    if rerun_evidence
        RepoEvidenceRunner.run_evidence_suite(
            manifest;
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
    )

    _write_release_index(output_root, bundles, report_path, summaries_dir)

    return (
        output_root = output_root,
        summaries_dir = summaries_dir,
        bundles_dir = bundles_dir,
        reports_dir = reports_dir,
        report_path = report_path,
        bundle_index_path = joinpath(bundles_dir, "bundle_index.toml"),
        release_index_path = joinpath(output_root, "README.md"),
        bundles = bundles,
    )
end

function _write_release_index(output_root::AbstractString, bundles, report_path::AbstractString, summaries_dir::AbstractString)
    summary_count = length(filter(name -> endswith(name, ".toml"), readdir(summaries_dir)))
    io = IOBuffer()
    println(io, "# Release Outputs")
    println(io)
    println(io, "- Generated: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")) (local time)")
    println(io, "- Validation report: [`$(basename(report_path))`](reports/$(basename(report_path)))")
    println(io, "- Evidence summaries: `$summary_count` TOML files in `summaries/`")
    println(io, "- Bundle index: [`bundle_index.toml`](bundles/bundle_index.toml)")
    println(io, "- Bundle overview: [`README.md`](bundles/README.md)")
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

end

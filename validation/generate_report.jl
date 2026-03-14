module ValidationReport

using Dates
using TOML

include(joinpath(@__DIR__, "manifest.jl"))
using .RepoValidationManifest

"""
    generate(manifest_path, output_path; summary_dir = nothing, bundle_dir = nothing,
             executed_entry_ids = nothing)

Generate a validation report from the manifest, optionally enriching it with
executed evidence summaries and reproduction-bundle links.
"""
function generate(
        manifest_path::AbstractString,
        output_path::AbstractString;
        summary_dir::Union{Nothing, AbstractString} = nothing,
        bundle_dir::Union{Nothing, AbstractString} = nothing,
        executed_entry_ids = nothing,
    )
    manifest = RepoValidationManifest.load_manifest(manifest_path)
    io = IOBuffer()
    capability_rows = RepoValidationManifest.capability_rows(manifest)
    stable_features = [f for (f, e) in manifest.features if e.maturity == :stable]
    provisional_features = [f for (f, e) in manifest.features if e.maturity == :provisional]
    experimental_features = [f for (f, e) in manifest.features if e.maturity == :experimental]
    summaries = _load_evidence_summaries(summary_dir)
    executed_entries = _executed_entries(manifest, summaries, executed_entry_ids)
    reported_entries = isnothing(executed_entry_ids) ? manifest.scientific_evidence : executed_entries

    println(io, "# Validation Report")
    println(io)
    println(io, "Generated: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")) (local time)")
    println(io)

    println(io, "## Research Contract")
    println(io)
    println(io, "- **Manifest version:** `$(manifest.manifest_version)`")
    println(io, "- **Julia support policy:** `$(manifest.support_policy)`")
    println(io, "- **Stable claim-bearing features:** Only stable `claim_bearing_solver` capabilities may support publication-grade scientific claims.")
    println(io, "- **Tooling features:** Dashboard and I/O features are treated as reproducibility infrastructure, not solver validation.")
    if !isempty(summaries)
        passed = count(summary -> get(summary, "status", "unknown") == "pass", values(summaries))
        println(io, "- **Executed evidence summaries loaded:** `$passed / $(length(summaries))` passing")
    end
    if !isnothing(executed_entry_ids)
        println(io, "- **Executed evidence subset:** `$(length(executed_entries))` selected evidence case(s)")
    end
    println(io)

    println(io, "## Capability Matrix")
    println(io)
    println(io, "| Feature | Role | Maturity | Claim Policy | Validation | Solver Family | Required Ladder | Summary | Limitations |")
    println(io, "|---------|------|----------|--------------|------------|---------------|-----------------|---------|-------------|")
    for row in capability_rows
        println(
            io,
            "| $(row.feature) | $(row.role) | $(row.maturity) | $(row.claim_policy) | $(row.validation) | $(row.solver_family) | $(isempty(row.required_ladder_stages) ? "n/a" : row.required_ladder_stages) | $(row.summary) | $(row.limitations) |",
        )
    end
    println(io)

    println(io, "## Scientific Evidence Cases")
    println(io)
    println(io, "| ID | Feature | Ladder Stage | Solver Family | Runtime | Category | Precision | Seed Policy | Metric | Expected Artifacts | Reference | Acceptance | Entrypoint |")
    println(io, "|----|---------|--------------|---------------|---------|----------|-----------|-------------|--------|--------------------|-----------|------------|------------|")
    for entry in reported_entries
        println(
            io,
            "| $(entry.id) | $(entry.feature) | $(entry.ladder_stage) | $(entry.solver_family) | $(entry.runtime_tier) | $(entry.category) | $(entry.precision_policy) | $(entry.random_seed_policy) | $(entry.metric) | $(join(entry.expected_artifacts, ", ")) | $(entry.reference_source) | $(entry.acceptance) | $(entry.entrypoint) |",
        )
    end
    println(io)

    if !isempty(summaries)
        println(io, "## Executed Evidence Results")
        println(io)
        println(io, "| ID | Status | Recorded Results | Summary File | Bundle Artifacts |")
        println(io, "|----|--------|------------------|--------------|------------------|")
        for entry in executed_entries
            summary = get(summaries, entry.id, nothing)
            if isnothing(summary)
                println(io, "| $(entry.id) | missing | 0 | n/a | n/a |")
                continue
            end
            bundle_artifacts = _bundle_artifact_links(output_path, bundle_dir, entry, summary)
            summary_link = _summary_link(output_path, summary_dir, entry.id)
            println(
                io,
                "| $(entry.id) | $(get(summary, "status", "unknown")) | $(get(summary, "recorded_result_count", 0)) | $summary_link | $(isempty(bundle_artifacts) ? "n/a" : join(bundle_artifacts, ", ")) |",
            )
        end
        println(io)
    end

    println(io, "## Evidence Ladder Coverage")
    println(io)
    println(io, "| Feature | Required Stages | Present Stages | Missing Stages | Status |")
    println(io, "|---------|-----------------|----------------|----------------|--------|")
    for row in RepoValidationManifest.evidence_ladder_rows(manifest)
        required = isempty(row.required) ? "n/a" : join(string.(row.required), ", ")
        present = isempty(row.present) ? "n/a" : join(string.(row.present), ", ")
        missing = isempty(row.missing) ? "none" : join(string.(row.missing), ", ")
        status = isempty(row.required) ? "not_enforced" : (row.satisfied ? "complete" : "incomplete")
        println(io, "| $(row.feature) | $required | $present | $missing | $status |")
    end
    println(io)

    println(io, "## Generated Pages Coverage")
    println(io)
    by_feature = Dict{Symbol, Vector{RepoValidationManifest.GeneratedPageEntry}}()
    for entry in manifest.generated_pages
        pages = get!(by_feature, entry.feature, RepoValidationManifest.GeneratedPageEntry[])
        push!(pages, entry)
    end
    for feature in sort!(collect(keys(by_feature)); by = string)
        pages = by_feature[feature]
        ci_count = count(e -> e.run_in_ci, pages)
        local_count = count(e -> e.run_locally, pages)
        println(io, "### $(feature) ($(length(pages)) pages, $(ci_count) in CI, $(local_count) run locally)")
        println(io)
        println(io, "| ID | Tier | CI | Local | Category | Metric | Source |")
        println(io, "|----|------|----|-------|----------|--------|--------|")
        for entry in pages
            ci = entry.run_in_ci ? "yes" : "no"
            local_ = entry.run_locally ? "yes" : "no"
            category = something(entry.category, :n_a)
            metric = something(entry.metric, "n/a")
            println(io, "| $(entry.id) | $(entry.validation_tier) | $ci | $local_ | $category | $metric | $(entry.source) |")
        end
        println(io)
    end

    if !isnothing(bundle_dir) && isdir(bundle_dir)
        println(io, "## Reproduction Bundles")
        println(io)
        println(io, "| Feature | Bundle Manifest | Bundle README | Artifact Count |")
        println(io, "|---------|-----------------|---------------|----------------|")
        for feature in sort!(readdir(bundle_dir); by = identity)
            feature_dir = joinpath(bundle_dir, feature)
            isdir(feature_dir) || continue
            manifest_path = joinpath(feature_dir, "bundle_manifest.toml")
            readme_path = joinpath(feature_dir, "README.md")
            artifact_dir = joinpath(feature_dir, "artifacts")
            artifact_count = isdir(artifact_dir) ? length(readdir(artifact_dir)) : 0
            manifest_link = isfile(manifest_path) ? _markdown_link(output_path, manifest_path) : "n/a"
            readme_link = isfile(readme_path) ? _markdown_link(output_path, readme_path) : "n/a"
            println(io, "| $feature | $manifest_link | $readme_link | $artifact_count |")
        end
        println(io)
    end

    println(io, "## Open Exclusions")
    println(io)
    if !isempty(provisional_features)
        println(io, "**Provisional** (not yet validated for scientific claims):")
        for f in sort!(provisional_features; by = string)
            println(io, "- `$f`: $(manifest.features[f].summary)")
        end
        println(io)
    end
    if !isempty(experimental_features)
        println(io, "**Experimental** (smoke tests only):")
        for f in sort!(experimental_features; by = string)
            println(io, "- `$f`: $(manifest.features[f].summary)")
        end
        println(io)
    end
    if !isempty(manifest.exclusions)
        println(io, "**Declared exclusions and demotions:**")
        for entry in manifest.exclusions
            feature = isnothing(entry.feature) ? "n/a" : string(entry.feature)
            println(io, "- `$(entry.id)` (`$(entry.status)`, feature: `$feature`): $(entry.reason)")
        end
        println(io)
    end

    println(io, "## Summary")
    println(io)
    println(io, "- **Total features:** $(length(manifest.features))")
    println(io, "- **Stable:** $(length(stable_features))")
    println(io, "- **Provisional:** $(length(provisional_features))")
    println(io, "- **Experimental:** $(length(experimental_features))")
    if isnothing(executed_entry_ids)
        println(io, "- **Scientific evidence cases:** $(length(manifest.scientific_evidence))")
    else
        println(io, "- **Scientific evidence cases in manifest:** $(length(manifest.scientific_evidence))")
        println(io, "- **Scientific evidence cases in this report:** $(length(reported_entries))")
    end
    println(io, "- **Generated pages:** $(length(manifest.generated_pages))")
    println(io, "- **Declared exclusions:** $(length(manifest.exclusions))")
    ladder_enforced = sum(!isempty(manifest.features[f].required_ladder_stages) for f in keys(manifest.features))
    ladder_complete = sum(isempty(row.required) || row.satisfied for row in RepoValidationManifest.evidence_ladder_rows(manifest))
    println(io, "- **Features with enforced evidence ladders:** $ladder_enforced")
    println(io, "- **Features currently satisfying their enforced ladder:** $ladder_complete")
    ci_pages = count(e -> e.run_in_ci, manifest.generated_pages)
    println(io, "- **Pages in CI:** $ci_pages")
    println(io)

    report = String(take!(io))
    mkpath(dirname(output_path))
    write(output_path, report)
    @info "Validation report written to $output_path"
    return report
end

function _executed_entries(manifest, summaries, executed_entry_ids)
    return if isnothing(executed_entry_ids)
        if isempty(summaries)
            manifest.scientific_evidence
        else
            ids = Set(keys(summaries))
            filter(entry -> entry.id in ids, manifest.scientific_evidence)
        end
    else
        ids = Set(String.(collect(executed_entry_ids)))
        filter(entry -> entry.id in ids, manifest.scientific_evidence)
    end
end

function _load_evidence_summaries(summary_dir::Union{Nothing, AbstractString})
    if isnothing(summary_dir) || !isdir(summary_dir)
        return Dict{String, Dict{String, Any}}()
    end
    summaries = Dict{String, Dict{String, Any}}()
    for file in sort!(filter(name -> endswith(name, ".toml"), readdir(summary_dir)); by = identity)
        path = joinpath(summary_dir, file)
        summary = TOML.parsefile(path)
        summaries[get(summary, "id", basename(path))] = summary
    end
    return summaries
end

function _summary_link(report_path::AbstractString, summary_dir::Union{Nothing, AbstractString}, id::AbstractString)
    if isnothing(summary_dir)
        return "n/a"
    end
    path = joinpath(summary_dir, "$(id).toml")
    return isfile(path) ? _markdown_link(report_path, path) : "n/a"
end

function _bundle_artifact_links(report_path, bundle_dir, entry, summary)
    if isnothing(bundle_dir)
        return String[]
    end
    artifact_dir = joinpath(bundle_dir, string(entry.feature), "artifacts")
    isdir(artifact_dir) || return String[]
    links = String[]
    for record in get(summary, "recorded_results", Any[])
        for artifact in get(record, "artifacts", Any[])
            artifact_name = String(artifact)
            artifact_path = joinpath(artifact_dir, basename(artifact_name))
            isfile(artifact_path) && push!(links, _markdown_link(report_path, artifact_path))
        end
    end
    return unique(links)
end

_markdown_link(report_path::AbstractString, target_path::AbstractString) =
    "[`$(basename(target_path))`]($(relpath(target_path, dirname(report_path))))"

end # module

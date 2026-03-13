module ValidationReport

using Dates

include(joinpath(@__DIR__, "manifest.jl"))
using .RepoValidationManifest

"""
    generate(manifest_path, output_path)

Generate a validation report from the manifest, summarising covered features,
executed evidence cases, maturity levels, and open exclusions.
"""
function generate(manifest_path::AbstractString, output_path::AbstractString)
    manifest = RepoValidationManifest.load_manifest(manifest_path)
    io = IOBuffer()
    capability_rows = RepoValidationManifest.capability_rows(manifest)
    stable_features = [f for (f, e) in manifest.features if e.maturity == :stable]
    provisional_features = [f for (f, e) in manifest.features if e.maturity == :provisional]
    experimental_features = [f for (f, e) in manifest.features if e.maturity == :experimental]

    println(io, "# Validation Report")
    println(io)
    println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS UTC"))")
    println(io)

    ## ── Research Contract ──
    println(io, "## Research Contract")
    println(io)
    println(io, "- **Julia support policy:** `$(manifest.support_policy)`")
    println(io, "- **Stable claim-bearing features:** Only stable `claim_bearing_solver` capabilities may support publication-grade scientific claims.")
    println(io, "- **Tooling features:** Dashboard and I/O features are treated as reproducibility infrastructure, not solver validation.")
    println(io)

    ## ── Capability Matrix ──
    println(io, "## Capability Matrix")
    println(io)
    println(io, "| Feature | Role | Maturity | Claim Policy | Validation | Solver Family | Summary | Limitations |")
    println(io, "|---------|------|----------|--------------|------------|---------------|---------|-------------|")
    for row in capability_rows
        println(
            io,
            "| $(row.feature) | $(row.role) | $(row.maturity) | $(row.claim_policy) | $(row.validation) | $(row.solver_family) | $(row.summary) | $(row.limitations) |",
        )
    end
    println(io)

    ## ── Scientific Evidence ──
    println(io, "## Scientific Evidence Cases")
    println(io)
    println(io, "| ID | Feature | Solver Family | Runtime | Category | Precision | Seed Policy | Metric | Expected Artifacts | Reference | Acceptance |")
    println(io, "|----|---------|---------------|---------|----------|-----------|-------------|--------|--------------------|-----------|------------|")
    for entry in manifest.scientific_evidence
        println(
            io,
            "| $(entry.id) | $(entry.feature) | $(entry.solver_family) | $(entry.runtime_tier) | $(entry.category) | $(entry.precision_policy) | $(entry.random_seed_policy) | $(entry.metric) | $(join(entry.expected_artifacts, ", ")) | $(entry.reference_source) | $(entry.acceptance) |",
        )
    end
    println(io)

    ## ── Generated Pages Coverage ──
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

    ## ── Open Exclusions ──
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

    ## ── Summary Statistics ──
    println(io, "## Summary")
    println(io)
    println(io, "- **Total features:** $(length(manifest.features))")
    println(io, "- **Stable:** $(length(stable_features))")
    println(io, "- **Provisional:** $(length(provisional_features))")
    println(io, "- **Experimental:** $(length(experimental_features))")
    println(io, "- **Scientific evidence cases:** $(length(manifest.scientific_evidence))")
    println(io, "- **Generated pages:** $(length(manifest.generated_pages))")
    println(io, "- **Declared exclusions:** $(length(manifest.exclusions))")
    ci_pages = count(e -> e.run_in_ci, manifest.generated_pages)
    println(io, "- **Pages in CI:** $ci_pages")
    println(io)

    report = String(take!(io))
    write(output_path, report)
    @info "Validation report written to $output_path"
    return report
end

end # module

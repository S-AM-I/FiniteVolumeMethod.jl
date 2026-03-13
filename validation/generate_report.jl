module ValidationReport

using Dates
using TOML

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

    println(io, "# Validation Report")
    println(io)
    println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS UTC"))")
    println(io)

    ## ── Capability Matrix ──
    println(io, "## Capability Matrix")
    println(io)
    println(io, "| Feature | Maturity | Validation | Summary |")
    println(io, "|---------|----------|------------|---------|")
    for row in RepoValidationManifest.capability_rows(manifest)
        println(io, "| $(row.feature) | $(row.maturity) | $(row.validation) | $(row.summary) |")
    end
    println(io)

    ## ── Scientific Evidence ──
    println(io, "## Scientific Evidence Cases")
    println(io)
    raw = TOML.parsefile(manifest_path)
    evidence_entries = get(raw, "scientific_evidence", [])
    println(io, "| ID | Feature | Runtime | Quantity | Reference | Threshold |")
    println(io, "|----|---------|---------|----------|-----------|-----------|")
    for entry in evidence_entries
        id = get(entry, "id", "")
        feature = get(entry, "feature", "")
        tier = get(entry, "runtime_tier", "")
        qty = get(entry, "quantity", "")
        ref = get(entry, "reference", "")
        thr = get(entry, "threshold", "")
        println(io, "| $id | $feature | $tier | $qty | $ref | $thr |")
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
        println(io, "| ID | Tier | CI | Local | Source |")
        println(io, "|----|------|----|-------|--------|")
        for entry in pages
            ci = entry.run_in_ci ? "yes" : "no"
            local_ = entry.run_locally ? "yes" : "no"
            println(io, "| $(entry.id) | $(entry.validation_tier) | $ci | $local_ | $(entry.source) |")
        end
        println(io)
    end

    ## ── Open Exclusions ──
    println(io, "## Open Exclusions")
    println(io)
    stable_features = [f for (f, e) in manifest.features if e.maturity == :stable]
    provisional_features = [f for (f, e) in manifest.features if e.maturity == :provisional]
    experimental_features = [f for (f, e) in manifest.features if e.maturity == :experimental]
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

    ## ── Summary Statistics ──
    println(io, "## Summary")
    println(io)
    println(io, "- **Total features:** $(length(manifest.features))")
    println(io, "- **Stable:** $(length(stable_features))")
    println(io, "- **Provisional:** $(length(provisional_features))")
    println(io, "- **Experimental:** $(length(experimental_features))")
    println(io, "- **Scientific evidence cases:** $(length(manifest.scientific_evidence))")
    println(io, "- **Generated pages:** $(length(manifest.generated_pages))")
    ci_pages = count(e -> e.run_in_ci, manifest.generated_pages)
    println(io, "- **Pages in CI:** $ci_pages")
    println(io)

    report = String(take!(io))
    write(output_path, report)
    @info "Validation report written to $output_path"
    return report
end

end # module

module RepoValidationManifest

using TOML

struct GeneratedPageEntry
    id::String
    page::String
    source::String
    feature::Symbol
    validation_tier::Symbol
    run_locally::Bool
    run_in_ci::Bool
end

struct ScientificEvidenceEntry
    id::String
    path::String
    feature::Symbol
    rationale::String
end

struct FeatureEntry
    feature::Symbol
    maturity::Symbol
    validation::Symbol
    summary::String
end

function load_manifest(path::AbstractString = joinpath(@__DIR__, "manifest.toml"))
    raw = TOML.parsefile(path)
    generated_pages = [
        GeneratedPageEntry(
                entry["id"],
                _repo_relpath(entry["page"]),
                _repo_relpath(entry["source"]),
                Symbol(entry["feature"]),
                Symbol(entry["validation_tier"]),
                get(entry, "run_locally", true),
                get(entry, "run_in_ci", false),
            ) for entry in get(raw, "generated_pages", [])
    ]
    scientific_evidence = [
        ScientificEvidenceEntry(
                entry["id"],
                _repo_relpath(entry["path"]),
                Symbol(entry["feature"]),
                entry["rationale"],
            ) for entry in get(raw, "scientific_evidence", [])
    ]
    features = Dict(
        Symbol(entry["feature"]) => FeatureEntry(
                Symbol(entry["feature"]),
                Symbol(entry["maturity"]),
                Symbol(entry["validation"]),
                entry["summary"],
            ) for entry in get(raw, "features", [])
    )
    return (;
        generated_pages = sort(generated_pages; by = entry -> entry.page),
        scientific_evidence = scientific_evidence,
        features,
    )
end

ci_generated_pages(manifest) = filter(entry -> entry.run_in_ci, manifest.generated_pages)
generated_page_paths(manifest) = Set(entry.page for entry in manifest.generated_pages)

function flatten_pages(pages)
    out = Set{String}()
    _flatten_pages!(out, pages)
    return out
end

function generated_navigation_paths(page_paths)
    allow_static = Set(
        [
            normpath("tutorials/overview.md"),
            normpath("wyos/overview.md"),
            normpath("verification/overview.md"),
            normpath("hyperbolic/overview.md"),
        ]
    )
    generated_prefixes = ("tutorials/", "wyos/", "verification/", "hyperbolic/tutorials/")
    return Set(
        path for path in page_paths
            if any(startswith(path, prefix) for prefix in generated_prefixes) && path ∉ allow_static
    )
end

function missing_generated_pages(manifest, page_paths)
    actual = Set(_repo_relpath(path) for path in page_paths)
    return sort!(collect(setdiff(generated_page_paths(manifest), actual)))
end

function unexpected_generated_navigation_pages(manifest, page_paths)
    expected = generated_page_paths(manifest)
    actual = generated_navigation_paths(Set(_repo_relpath(path) for path in page_paths))
    return sort!(collect(setdiff(actual, expected)))
end

function capability_rows(manifest)
    rows = NamedTuple[]
    for feature in sort!(collect(keys(manifest.features)); by = string)
        entry = manifest.features[feature]
        push!(
            rows, (
                feature = feature,
                maturity = entry.maturity,
                validation = entry.validation,
                summary = entry.summary,
            )
        )
    end
    return rows
end

function _flatten_pages!(out, pages)
    for page in pages
        if page[2] isa String
            push!(out, normpath(page[2]))
        else
            _flatten_pages!(out, page[2])
        end
    end
    return out
end

_repo_relpath(path::AbstractString) = replace(path, '\\' => '/')

end

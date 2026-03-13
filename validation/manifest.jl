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
    category::Union{Nothing, Symbol}
    reference_kind::Union{Nothing, Symbol}
    reference_source::Union{Nothing, String}
    metric::Union{Nothing, String}
    acceptance::Union{Nothing, String}
end

struct ScientificEvidenceEntry
    id::String
    path::String
    entrypoint::String
    feature::Symbol
    rationale::String
    runtime_tier::Symbol
    ladder_stage::Symbol
    category::Symbol
    reference_kind::Symbol
    reference_source::String
    metric::String
    acceptance::String
    solver_family::Symbol
    precision_policy::Symbol
    random_seed_policy::Symbol
    expected_artifacts::Vector{String}
    summary_required::Bool
end

struct FeatureEntry
    feature::Symbol
    maturity::Symbol
    validation::Symbol
    summary::String
    role::Symbol
    solver_family::Union{Nothing, Symbol}
    precision_policy::Union{Nothing, Symbol}
    random_seed_policy::Union{Nothing, Symbol}
    backend_policy::Union{Nothing, Symbol}
    required_ladder_stages::Vector{Symbol}
    limitations::Vector{String}
end

struct ExclusionEntry
    id::String
    path::String
    feature::Union{Nothing, Symbol}
    status::Symbol
    reason::String
end

function load_manifest(path::AbstractString = joinpath(@__DIR__, "manifest.toml"))
    raw = TOML.parsefile(path)
    manifest_version = get(raw, "manifest_version", 1)
    support_policy = Symbol(get(raw, "support_policy", "current_lts_and_stable"))
    features = Dict(
        Symbol(entry["feature"]) => FeatureEntry(
                Symbol(entry["feature"]),
                Symbol(entry["maturity"]),
                Symbol(entry["validation"]),
                entry["summary"],
                Symbol(entry["role"]),
                _optional_symbol(entry, "solver_family"),
                _optional_symbol(entry, "precision_policy"),
                _optional_symbol(entry, "random_seed_policy"),
                _optional_symbol(entry, "backend_policy"),
                _optional_symbol_vector(entry, "required_ladder_stages"),
                _optional_string_vector(entry, "limitations"),
            ) for entry in get(raw, "features", [])
    )
    generated_pages = [
        GeneratedPageEntry(
                entry["id"],
                _repo_relpath(entry["page"]),
                _repo_relpath(entry["source"]),
                Symbol(entry["feature"]),
                Symbol(entry["validation_tier"]),
                get(entry, "run_locally", true),
                get(entry, "run_in_ci", false),
                _optional_symbol(entry, "category"),
                _optional_symbol(entry, "reference_kind"),
                get(entry, "reference_source", nothing),
                get(entry, "metric", nothing),
                get(entry, "acceptance", nothing),
            ) for entry in get(raw, "generated_pages", [])
    ]
    scientific_evidence = [
        ScientificEvidenceEntry(
                entry["id"],
                _repo_relpath(entry["path"]),
                _repo_relpath(get(entry, "entrypoint", entry["path"])),
                Symbol(entry["feature"]),
                entry["rationale"],
                Symbol(entry["runtime_tier"]),
                _parse_ladder_stage(entry),
                Symbol(entry["category"]),
                Symbol(entry["reference_kind"]),
                entry["reference_source"],
                entry["metric"],
                entry["acceptance"],
                Symbol(entry["solver_family"]),
                Symbol(entry["precision_policy"]),
                Symbol(entry["random_seed_policy"]),
                _optional_string_vector(entry, "expected_artifacts"),
                get(entry, "summary_required", false),
            ) for entry in get(raw, "scientific_evidence", [])
    ]
    exclusions = [
        ExclusionEntry(
                entry["id"],
                _repo_relpath(entry["path"]),
                _optional_symbol(entry, "feature"),
                Symbol(entry["status"]),
                entry["reason"],
            ) for entry in get(raw, "exclusions", [])
    ]
    return (;
        manifest_version,
        support_policy,
        generated_pages = sort(generated_pages; by = entry -> entry.page),
        scientific_evidence = sort(scientific_evidence; by = entry -> entry.id),
        exclusions = sort(exclusions; by = entry -> entry.id),
        features,
    )
end

ci_generated_pages(manifest) = filter(entry -> entry.run_in_ci, manifest.generated_pages)
generated_page_paths(manifest) = Set(entry.page for entry in manifest.generated_pages)
verification_pages(manifest) = filter(entry -> startswith(entry.page, "verification/"), manifest.generated_pages)

function feature_claim_policy(entry::FeatureEntry)
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
                role = entry.role,
                solver_family = something(entry.solver_family, :n_a),
                claim_policy = feature_claim_policy(entry),
                required_ladder_stages = join(string.(entry.required_ladder_stages), ", "),
                summary = entry.summary,
                limitations = join(entry.limitations, "; "),
            )
        )
    end
    return rows
end

scientific_evidence_for_feature(manifest, feature::Symbol) =
    filter(entry -> entry.feature == feature, manifest.scientific_evidence)

function feature_ladder_coverage(manifest, feature::Symbol)
    entry = manifest.features[feature]
    required = copy(entry.required_ladder_stages)
    present = sort!(
        collect(Set(evidence.ladder_stage for evidence in scientific_evidence_for_feature(manifest, feature)));
        by = string,
    )
    missing = [stage for stage in required if stage ∉ present]
    return (
        feature = feature,
        required = required,
        present = present,
        missing = missing,
        satisfied = isempty(missing),
    )
end

function evidence_ladder_rows(manifest)
    return [feature_ladder_coverage(manifest, feature) for feature in sort!(collect(keys(manifest.features)); by = string)]
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

_optional_symbol(entry, key) = haskey(entry, key) ? Symbol(entry[key]) : nothing
_optional_symbol_vector(entry, key) = haskey(entry, key) ? Symbol.(entry[key]) : Symbol[]
_optional_string_vector(entry, key) = haskey(entry, key) ? String.(entry[key]) : String[]
_repo_relpath(path::AbstractString) = replace(path, '\\' => '/')
_parse_ladder_stage(entry) = haskey(entry, "ladder_stage") ? Symbol(entry["ladder_stage"]) : _default_ladder_stage(Symbol(entry["category"]))

function _default_ladder_stage(category::Symbol)
    return if category == :code_verification
        :verification
    elseif category == :analytical_benchmark
        :benchmark
    else
        :validation
    end
end

end

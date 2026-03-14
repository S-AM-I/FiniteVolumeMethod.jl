module RepoSummaryReplay

using TOML

const VOLATILE_KEYS = Set(["started_at", "finished_at", "summary_path", "artifact_output_dir"])

function compare_summary_directories(
        reference_dir::AbstractString,
        candidate_dir::AbstractString;
        entry_ids = nothing,
        atol::Float64 = 1.0e-12,
        rtol::Float64 = 1.0e-12,
    )
    ids = isnothing(entry_ids) ? _shared_entry_ids(reference_dir, candidate_dir) : String.(collect(entry_ids))
    diffs = Dict{String, Vector{String}}()
    for id in sort!(collect(ids); by = identity)
        reference_path = joinpath(reference_dir, "$(id).toml")
        candidate_path = joinpath(candidate_dir, "$(id).toml")
        isfile(reference_path) || throw(ArgumentError("Missing reference summary for `$id` at `$reference_path`."))
        isfile(candidate_path) || throw(ArgumentError("Missing candidate summary for `$id` at `$candidate_path`."))
        summary_diffs = compare_summaries(
            TOML.parsefile(reference_path),
            TOML.parsefile(candidate_path);
            atol,
            rtol,
        )
        !isempty(summary_diffs) && (diffs[id] = summary_diffs)
    end
    return diffs
end

function compare_summaries(reference_summary, candidate_summary; atol::Float64 = 1.0e-12, rtol::Float64 = 1.0e-12)
    diffs = String[]
    _compare_values!(
        diffs,
        "summary",
        _normalize_summary(reference_summary),
        _normalize_summary(candidate_summary);
        atol,
        rtol,
    )
    return diffs
end

function _normalize_summary(value)
    return if value isa AbstractDict
        normalized = Dict{String, Any}()
        for (key, item) in pairs(value)
            key_string = String(key)
            key_string in VOLATILE_KEYS && continue
            normalized[key_string] = _normalize_summary(item)
        end
        normalized
    elseif value isa AbstractVector
        [_normalize_summary(item) for item in value]
    else
        value
    end
end

function _compare_values!(diffs, path::AbstractString, reference, candidate; atol::Float64, rtol::Float64)
    if reference isa AbstractDict && candidate isa AbstractDict
        reference_keys = Set(keys(reference))
        candidate_keys = Set(keys(candidate))
        missing = sort!(collect(setdiff(reference_keys, candidate_keys)); by = identity)
        extra = sort!(collect(setdiff(candidate_keys, reference_keys)); by = identity)
        !isempty(missing) && push!(diffs, "$path is missing keys: $(join(missing, ", "))")
        !isempty(extra) && push!(diffs, "$path has unexpected keys: $(join(extra, ", "))")
        for key in sort!(collect(intersect(reference_keys, candidate_keys)); by = identity)
            _compare_values!(diffs, "$path.$key", reference[key], candidate[key]; atol, rtol)
        end
    elseif reference isa AbstractVector && candidate isa AbstractVector
        if length(reference) != length(candidate)
            push!(diffs, "$path length mismatch: expected $(length(reference)), got $(length(candidate))")
            return nothing
        end
        for i in eachindex(reference)
            _compare_values!(diffs, "$path[$i]", reference[i], candidate[i]; atol, rtol)
        end
    elseif reference isa Real && candidate isa Real
        isapprox(float(reference), float(candidate); atol, rtol) ||
            push!(diffs, "$path mismatch: expected $(reference), got $(candidate)")
    else
        reference == candidate || push!(diffs, "$path mismatch: expected $(reference), got $(candidate)")
    end
    return nothing
end

function _shared_entry_ids(reference_dir::AbstractString, candidate_dir::AbstractString)
    reference_ids = Set(_entry_ids(reference_dir))
    candidate_ids = Set(_entry_ids(candidate_dir))
    return intersect(reference_ids, candidate_ids)
end

function _entry_ids(dir::AbstractString)
    return [
        first(splitext(name)) for name in readdir(dir)
            if endswith(name, ".toml")
    ]
end

end

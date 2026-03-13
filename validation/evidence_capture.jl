const _EVIDENCE_RESULTS = Dict{String, Any}[]

evidence_results() = copy(_EVIDENCE_RESULTS)

function _capture_value(value)
    return if value isa Symbol
        string(value)
    elseif value isa NamedTuple
        Dict(string(key) => _capture_value(val) for (key, val) in pairs(value))
    elseif value isa AbstractDict
        Dict(string(key) => _capture_value(val) for (key, val) in pairs(value))
    elseif value isa AbstractVector
        [_capture_value(val) for val in value]
    elseif value isa Tuple
        [_capture_value(val) for val in value]
    elseif value isa Nothing || value isa Bool || value isa Integer || value isa AbstractFloat || value isa AbstractString
        value
    else
        string(value)
    end
end

function record_evidence_result(; status = :pass, metrics = Dict{String, Any}(), artifacts = String[], notes = String[], summary = nothing)
    record = Dict{String, Any}(
        "status" => string(status),
        "metrics" => _capture_value(metrics),
        "artifacts" => _capture_value(String.(artifacts)),
        "notes" => _capture_value(String.(notes)),
    )
    if !isnothing(summary)
        record["summary"] = _capture_value(summary)
    end
    push!(_EVIDENCE_RESULTS, record)
    return record
end

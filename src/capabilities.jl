include(joinpath(@__DIR__, "..", "validation", "manifest.jl"))
using .RepoValidationManifest

const _FEATURE_MANIFEST = RepoValidationManifest.load_manifest(joinpath(@__DIR__, "..", "validation", "manifest.toml"))
const _FEATURE_CAPABILITIES = Dict{Symbol, NamedTuple{(:maturity, :validation, :summary), Tuple{Symbol, Symbol, String}}}(
    feature => (
            maturity = entry.maturity,
            validation = entry.validation,
            summary = entry.summary,
        ) for (feature, entry) in _FEATURE_MANIFEST.features
)

supported_features() = sort!(collect(keys(_FEATURE_CAPABILITIES)); by = string)

function feature_maturity(feature::Symbol)
    haskey(_FEATURE_CAPABILITIES, feature) || throw(ArgumentError("Unknown feature: $feature"))
    return _FEATURE_CAPABILITIES[feature].maturity
end

function feature_validation_status(feature::Symbol)
    haskey(_FEATURE_CAPABILITIES, feature) || throw(ArgumentError("Unknown feature: $feature"))
    return _FEATURE_CAPABILITIES[feature].validation
end

function capability_matrix()
    return [
        (
                feature = feature,
                maturity = _FEATURE_CAPABILITIES[feature].maturity,
                validation = _FEATURE_CAPABILITIES[feature].validation,
                summary = _FEATURE_CAPABILITIES[feature].summary,
            ) for feature in supported_features()
    ]
end

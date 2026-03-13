include(joinpath(@__DIR__, "..", "validation", "manifest.jl"))
using .RepoValidationManifest

const _FEATURE_MANIFEST = RepoValidationManifest.load_manifest(joinpath(@__DIR__, "..", "validation", "manifest.toml"))
const _FEATURE_CAPABILITIES = Dict{
    Symbol,
    NamedTuple{
        (:maturity, :validation, :role, :solver_family, :required_ladder_stages, :claim_policy, :summary, :limitations),
        Tuple{Symbol, Symbol, Symbol, Union{Nothing, Symbol}, Vector{Symbol}, Symbol, String, Vector{String}},
    },
}(
    feature => (
            maturity = entry.maturity,
            validation = entry.validation,
            role = entry.role,
            solver_family = entry.solver_family,
            required_ladder_stages = copy(entry.required_ladder_stages),
            claim_policy = RepoValidationManifest.feature_claim_policy(entry),
            summary = entry.summary,
            limitations = copy(entry.limitations),
        ) for (feature, entry) in _FEATURE_MANIFEST.features
)

supported_features() = sort!(collect(keys(_FEATURE_CAPABILITIES)); by = string)

function _feature_capability(feature::Symbol)
    haskey(_FEATURE_CAPABILITIES, feature) || throw(ArgumentError("Unknown feature: $feature"))
    return _FEATURE_CAPABILITIES[feature]
end

feature_maturity(feature::Symbol) = _feature_capability(feature).maturity
feature_validation_status(feature::Symbol) = _feature_capability(feature).validation
feature_role(feature::Symbol) = _feature_capability(feature).role
feature_solver_family(feature::Symbol) = _feature_capability(feature).solver_family
feature_required_ladder_stages(feature::Symbol) = copy(_feature_capability(feature).required_ladder_stages)
feature_claim_policy(feature::Symbol) = _feature_capability(feature).claim_policy
feature_limitations(feature::Symbol) = copy(_feature_capability(feature).limitations)

function capability_matrix()
    return [
        (
                feature = feature,
                maturity = _FEATURE_CAPABILITIES[feature].maturity,
                validation = _FEATURE_CAPABILITIES[feature].validation,
                role = _FEATURE_CAPABILITIES[feature].role,
                solver_family = _FEATURE_CAPABILITIES[feature].solver_family,
                required_ladder_stages = copy(_FEATURE_CAPABILITIES[feature].required_ladder_stages),
                claim_policy = _FEATURE_CAPABILITIES[feature].claim_policy,
                summary = _FEATURE_CAPABILITIES[feature].summary,
                limitations = copy(_FEATURE_CAPABILITIES[feature].limitations),
            ) for feature in supported_features()
    ]
end

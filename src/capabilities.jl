include(joinpath(@__DIR__, "..", "validation", "manifest.jl"))
using .RepoValidationManifest

const _FEATURE_MANIFEST_PATH = joinpath(@__DIR__, "..", "validation", "manifest.toml")
Base.include_dependency(_FEATURE_MANIFEST_PATH)
const _FEATURE_MANIFEST = RepoValidationManifest.load_manifest(_FEATURE_MANIFEST_PATH)
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

"""Return a sorted list of all registered feature names."""
supported_features() = sort!(collect(keys(_FEATURE_CAPABILITIES)); by = string)

function _feature_capability(feature::Symbol)
    haskey(_FEATURE_CAPABILITIES, feature) || throw(ArgumentError("Unknown feature: $feature"))
    return _FEATURE_CAPABILITIES[feature]
end

"""Return the maturity level (`:stable`, `:experimental`, `:deprecated`) for `feature`."""
feature_maturity(feature::Symbol) = _feature_capability(feature).maturity
"""Return the validation status for `feature`."""
feature_validation_status(feature::Symbol) = _feature_capability(feature).validation
"""Return the role (`:core`, `:extension`, `:research`) for `feature`."""
feature_role(feature::Symbol) = _feature_capability(feature).role
"""Return the solver family for `feature`, or `nothing`."""
feature_solver_family(feature::Symbol) = _feature_capability(feature).solver_family
"""Return the required validation ladder stages for `feature`."""
feature_required_ladder_stages(feature::Symbol) = copy(_feature_capability(feature).required_ladder_stages)
"""Return the claim policy (`:publication_ready`, `:provisional`, `:manual_review`) for `feature`."""
feature_claim_policy(feature::Symbol) = _feature_capability(feature).claim_policy
"""Return a list of known limitations for `feature`."""
feature_limitations(feature::Symbol) = copy(_feature_capability(feature).limitations)

"""Return the full capability matrix as a vector of named tuples."""
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

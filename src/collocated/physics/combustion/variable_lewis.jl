# combustion/variable_lewis.jl — Variable Lewis number species transport
#
# The default species-transport closure assumes unity Lewis number
# (`Le_i = 1`), so the species mass diffusivity matches the thermal
# diffusivity `α_thermal`. Real hydrocarbon-air flames violate this
# strongly (H₂ at Le ≈ 0.3, C₇H₁₆ at Le ≈ 3).
#
# `VariableLewis{NS, T}` stores one `Le_i` per species; the effective
# species diffusivity is then `α_species_i = α_thermal / Le_i`.

"""
    VariableLewis{NS, T}

Per-species Lewis number wrapper for non-unity Le species transport.

# Fields
- `Le::NTuple{NS, T}` — Lewis numbers `Le_i = α_thermal / α_species_i`.
"""
struct VariableLewis{NS, T}
    Le::NTuple{NS, T}
end

"""
    VariableLewis(Le::NTuple{NS, <:Real}) -> VariableLewis{NS, T}

Construct a [`VariableLewis`](@ref) from an `NTuple` of per-species
Lewis numbers. All entries must be strictly positive.
"""
function VariableLewis(Le::NTuple{NS, <:Real}) where {NS}
    T = promote_type(map(typeof, Le)...)
    for (i, v) in enumerate(Le)
        v > 0 || error("VariableLewis: Le[$i] must be positive, got $v")
    end
    return VariableLewis{NS, T}(NTuple{NS, T}(Le))
end

"""
    VariableLewis(combustion_props; Le...) -> VariableLewis{NS, T}

Convenience constructor that accepts keyword-based per-species Lewis
numbers keyed by species name, e.g.
`VariableLewis(props; fuel = 1.1, oxidizer = 0.9, product = 1.0)`.
Species not given a value default to `1.0`.
"""
function VariableLewis(
        combustion_props::CombustionProperties{NS, T}; kwargs...,
    ) where {NS, T}
    Le = ntuple(Val(NS)) do i
        name = combustion_props.species_names[i]
        return T(get(kwargs, name, one(T)))
    end
    return VariableLewis{NS, T}(Le)
end

"""
    species_diffusivity(Le, alpha_thermal, i) -> value_or_vector

Return the effective species diffusivity for species `i` given
`Le::VariableLewis` and the thermal diffusivity `alpha_thermal`.

If `alpha_thermal isa Real`, returns `alpha_thermal / Le[i]`. If
`alpha_thermal isa AbstractVector`, returns a new vector of the same
length with `alpha_thermal[c] / Le[i]` per cell.
"""
function species_diffusivity(Le::VariableLewis{NS, T}, alpha_thermal::Real, i::Int) where {NS, T}
    (1 <= i <= NS) || error("species_diffusivity: index $i out of range 1:$NS")
    return T(alpha_thermal) / Le.Le[i]
end

function species_diffusivity(
        Le::VariableLewis{NS, T}, alpha_thermal::AbstractVector, i::Int,
    ) where {NS, T}
    (1 <= i <= NS) || error("species_diffusivity: index $i out of range 1:$NS")
    out = Vector{T}(undef, length(alpha_thermal))
    inv_Le = one(T) / Le.Le[i]
    @inbounds for c in eachindex(alpha_thermal)
        out[c] = T(alpha_thermal[c]) * inv_Le
    end
    return out
end

"""
    lewis_number(Le, i) -> T

Look up the Lewis number of species `i`.
"""
function lewis_number(Le::VariableLewis{NS}, i::Int) where {NS}
    (1 <= i <= NS) || error("lewis_number: index $i out of range 1:$NS")
    return Le.Le[i]
end

# combustion/edm.jl — Eddy Dissipation Model reaction rates
#
# Implements the Magnussen-Hjertager EDM for a one-step
# fuel + oxidizer → product reaction. Requires turbulence k and ε
# fields; falls back to a constant mixing time scale when unavailable.

# Fallback mixing time scale [s] when no turbulence model is available
const _EDM_FALLBACK_TAU_MIX = 0.1

"""
    _species_index(props::CombustionProperties{NS}, name::Symbol) -> Int

Look up the index of species `name` in `props.species_names`.
Throws an error if the species is not found.
"""
function _species_index(props::CombustionProperties{NS}, name::Symbol) where {NS}
    for i in 1:NS
        props.species_names[i] === name && return i
    end
    return error("Species :$name not found in $(props.species_names)")
end

"""
    compute_edm_reaction_rates(
        edm, species_state, combustion_props,
        k_field, eps_field, density, mesh,
    ) -> NTuple{NS, Vector{T}}

Compute per-species per-cell volumetric reaction rates using the
Eddy Dissipation Model.

For a 3-species system (fuel, oxidizer, product), the EDM fuel
consumption rate is:

    ω_fuel = -ρ · A · (ε/k) · min(Y_fuel, Y_ox / s)

with a product-limited variant:

    ω_fuel_product = -ρ · A · B · (ε/k) · Y_product / (1 + s)

The final rate is the minimum magnitude (most negative = slowest):

    ω_fuel = max(ω_fuel_mixing, ω_fuel_product)

Species rates follow from stoichiometry:
- `ω_oxidizer = s · ω_fuel`
- `ω_product = -(1 + s) · ω_fuel`

# Arguments
- `edm::EddyDissipationModel{T}` — model constants
- `species_state::SpeciesState{NS, T}` — current mass fractions
- `combustion_props::CombustionProperties{NS, T}` — thermochemical properties
- `k_field::Union{Nothing, Vector{T}}` — turbulent kinetic energy (from RANS)
- `eps_field::Union{Nothing, Vector{T}}` — dissipation rate (from RANS)
- `density::T` — fluid density
- `mesh::UnstructuredFVMMesh` — mesh

Returns an `NTuple{NS, Vector{T}}` of reaction rates [kg/(m³·s)].
Negative values indicate consumption, positive indicate production.
"""
function compute_edm_reaction_rates(
        edm::EddyDissipationModel{T},
        species_state::SpeciesState{NS, T},
        combustion_props::CombustionProperties{NS, T},
        k_field::Union{Nothing, Vector{T}},
        eps_field::Union{Nothing, Vector{T}},
        density::T,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T, NS}
    nc = length(mesh.cell_volumes)
    A = edm.A_edm
    B = edm.B_edm
    s = combustion_props.stoich_ratio

    # Look up species by name (no assumed ordering)
    fuel_idx = _species_index(combustion_props, :fuel)
    ox_idx = _species_index(combustion_props, :oxidizer)
    has_product = any(n -> n === :product, combustion_props.species_names)
    prod_idx = has_product ? _species_index(combustion_props, :product) : 0

    Y_fuel = species_state.Y[fuel_idx].internal
    Y_ox = species_state.Y[ox_idx].internal
    Y_prod = has_product ? species_state.Y[prod_idx].internal : nothing

    # Allocate output
    omega = ntuple(_ -> zeros(T, nc), Val(NS))

    for c in 1:nc
        # Mixing rate: ε/k, with fallback
        if k_field !== nothing && eps_field !== nothing
            k_c = max(k_field[c], T(1.0e-20))
            mixing_rate = eps_field[c] / k_c
        else
            mixing_rate = one(T) / T(_EDM_FALLBACK_TAU_MIX)
        end

        # Mixing-limited rate
        omega_fuel_mix = -density * A * mixing_rate * min(Y_fuel[c], Y_ox[c] / s)

        # Product-limited rate (avoid reaction in product-rich regions)
        # Only apply when product is present above a small threshold;
        # otherwise the mixing-limited rate alone controls ignition.
        omega_fuel_c = omega_fuel_mix
        if has_product && Y_prod[c] > T(1.0e-10)
            omega_fuel_prod = -density * A * B * mixing_rate * Y_prod[c] / (one(T) + s)
            omega_fuel_c = max(omega_fuel_mix, omega_fuel_prod)
        end

        # Store species rates from stoichiometry
        omega[fuel_idx][c] = omega_fuel_c              # fuel consumed
        omega[ox_idx][c] = s * omega_fuel_c            # oxidizer consumed
        if has_product
            omega[prod_idx][c] = -(one(T) + s) * omega_fuel_c  # product formed
        end

        # Additional species (if NS > 3) get zero rate
    end

    return omega
end

"""
    compute_heat_release(reaction_rates, combustion_props) -> Vector{T}

Compute volumetric heat release from combustion:

    S_h[c] = -ω_fuel[c] · ΔH

Since `ω_fuel` is negative for fuel consumption and `ΔH` is positive
for exothermic reactions, `S_h` is positive (heat source).

Returns heat release per unit volume [W/m³].
"""
function compute_heat_release(
        reaction_rates::NTuple{NS, Vector{T}},
        combustion_props::CombustionProperties{NS, T},
    ) where {NS, T}
    fuel_idx = _species_index(combustion_props, :fuel)
    omega_fuel = reaction_rates[fuel_idx]
    dH = combustion_props.heat_of_combustion
    nc = length(omega_fuel)
    S_h = Vector{T}(undef, nc)
    for c in 1:nc
        S_h[c] = -omega_fuel[c] * dH
    end
    return S_h
end

# combustion/edc.jl — Eddy Dissipation Concept (EDC) reaction rates
#
# Implements the Magnussen (2005) EDC fine-structure reactor model.
# Unlike the simpler EDM, EDC resolves a fine-structure volume fraction
# and residence time derived from turbulence quantities, enabling
# finite-rate chemistry within the fine structures.

# Fallback mixing rate [1/s] when no turbulence model is available
const _EDC_FALLBACK_MIXING_RATE = 10.0

"""
    compute_edc_reaction_rates(
        edc, species_state, combustion_props,
        k_field, eps_field, density, nu, mesh,
    ) -> NTuple{NS, Vector{T}}

Compute per-species per-cell volumetric reaction rates using the
Eddy Dissipation Concept.

The EDC fine-structure model computes:

    gamma_star = C_gamma * (nu * epsilon / k^2)^(1/4)
    tau_star = C_tau * sqrt(nu / epsilon)
    omega_fuel = -rho * gamma_star^2 / (tau_star * (1 - gamma_star^3)) * (Y_fuel - Y_fuel_star)

where `Y_fuel_star` is the equilibrium mass fraction (= 0 for complete
combustion).

When `k_field` and `eps_field` are not available, the model falls back
to an EDM-like mixing rate using `_EDC_FALLBACK_MIXING_RATE`.

Species rates follow from stoichiometry:
- `omega_oxidizer = s * omega_fuel`
- `omega_product = -(1 + s) * omega_fuel`

# Arguments
- `edc::EddyDissipationConcept{T}` --- model constants (C_gamma, C_tau)
- `species_state::SpeciesState{NS, T}` --- current mass fractions
- `combustion_props::CombustionProperties{NS, T}` --- thermochemical properties
- `k_field::Union{Nothing, Vector{T}}` --- turbulent kinetic energy [m^2/s^2]
- `eps_field::Union{Nothing, Vector{T}}` --- dissipation rate [m^2/s^3]
- `density::T` --- fluid density [kg/m^3]
- `nu::T` --- kinematic viscosity [m^2/s]
- `mesh::UnstructuredFVMMesh` --- mesh

Returns an `NTuple{NS, Vector{T}}` of reaction rates [kg/(m^3 s)].
Negative values indicate consumption, positive indicate production.
"""
function compute_edc_reaction_rates(
        edc::EddyDissipationConcept{T},
        species_state::SpeciesState{NS, T},
        combustion_props::CombustionProperties{NS, T},
        k_field::Union{Nothing, Vector{T}},
        eps_field::Union{Nothing, Vector{T}},
        density::T,
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T, NS}
    nc = length(mesh.cell_volumes)
    C_gamma = edc.C_gamma
    C_tau = edc.C_tau
    s = combustion_props.stoich_ratio

    # Species lookup
    fuel_idx = _species_index(combustion_props, :fuel)
    ox_idx = _species_index(combustion_props, :oxidizer)
    has_product = any(n -> n === :product, combustion_props.species_names)
    prod_idx = has_product ? _species_index(combustion_props, :product) : 0

    Y_fuel = species_state.Y[fuel_idx].internal
    Y_ox = species_state.Y[ox_idx].internal

    # Allocate output
    omega = ntuple(_ -> zeros(T, nc), Val(NS))

    for c in 1:nc
        has_turb = k_field !== nothing && eps_field !== nothing

        if has_turb
            k_c = max(k_field[c], T(1.0e-20))
            eps_c = max(eps_field[c], T(1.0e-20))

            # Fine-structure volume fraction
            gamma_star = C_gamma * (nu * eps_c / k_c^2)^(one(T) / T(4))
            # Clamp gamma_star to (0, 1) for physical bounds
            gamma_star = clamp(gamma_star, T(1.0e-10), T(0.99))

            # Fine-structure time scale
            tau_star = C_tau * sqrt(nu / eps_c)
            tau_star = max(tau_star, T(1.0e-20))

            # EDC reaction rate
            # Y_fuel_star = 0 for complete combustion
            denom = tau_star * (one(T) - gamma_star^3)
            denom = max(denom, T(1.0e-20))

            omega_fuel_c = -density * gamma_star^2 / denom * Y_fuel[c]
        else
            # Fallback: EDM-like mixing rate
            mixing_rate = T(_EDC_FALLBACK_MIXING_RATE)
            omega_fuel_c = -density * mixing_rate * min(Y_fuel[c], Y_ox[c] / s)
        end

        # Store species rates from stoichiometry
        omega[fuel_idx][c] = omega_fuel_c              # fuel consumed
        omega[ox_idx][c] = s * omega_fuel_c            # oxidizer consumed
        if has_product
            omega[prod_idx][c] = -(one(T) + s) * omega_fuel_c  # product formed
        end
    end

    return omega
end

# combustion/arrhenius.jl — Finite-rate Arrhenius reaction rates for collocated solver
#
# Computes species source terms from Arrhenius kinetics:
#   k_f = A * T^b * exp(-E_a / (R * T))
#   omega_fuel = -rho * k_f * prod(Y_k^n_k)
#
# Can be used standalone or blended with EDM via the finite-rate / eddy-dissipation
# (FR/ED) approach.

const _R_UNIVERSAL = 8.314  # J/(mol·K)

# ── Collocated Arrhenius reaction type ────────────────────────────

"""
    CollocatedArrheniusReaction{NS, T}

One-step Arrhenius reaction for the collocated combustion solver.

    fuel + s * oxidizer → (1 + s) * product

Rate:  `k_f = A * T^b * exp(-E_a / (R_univ * T))`
Fuel consumption: `ω_fuel = -ρ * k_f * Y_fuel^n_fuel * Y_ox^n_ox`

# Fields
- `A::T` — pre-exponential factor [1/s or consistent units]
- `b::T` — temperature exponent
- `E_a::T` — activation energy [J/mol]
- `n_fuel::T` — fuel concentration exponent (typically 1)
- `n_ox::T` — oxidizer concentration exponent (typically 1)
"""
struct CollocatedArrheniusReaction{T}
    A::T
    b::T
    E_a::T
    n_fuel::T
    n_ox::T
end

"""
    CollocatedArrheniusReaction(; A, b = 0.0, E_a, n_fuel = 1.0, n_ox = 1.0)

Construct a [`CollocatedArrheniusReaction`](@ref).
"""
function CollocatedArrheniusReaction(;
        A::Real = 1.0e10,
        b::Real = 0.0,
        E_a::Real = 1.0e5,
        n_fuel::Real = 1.0,
        n_ox::Real = 1.0,
    )
    T = promote_type(typeof(A), typeof(b), typeof(E_a), typeof(n_fuel), typeof(n_ox))
    return CollocatedArrheniusReaction{T}(T(A), T(b), T(E_a), T(n_fuel), T(n_ox))
end

# ── Rate computation ──────────────────────────────────────────────

"""
    compute_arrhenius_reaction_rates(
        reaction, species_state, combustion_props,
        T_field, density, mesh,
    ) -> NTuple{NS, Vector{T}}

Compute per-species per-cell volumetric reaction rates using
finite-rate Arrhenius kinetics.

The forward rate constant is:
```
    k_f(T) = A · T^b · exp(-E_a / (R · T))
```

The fuel consumption rate is:
```
    ω_fuel = -ρ · k_f · Y_fuel^n_fuel · Y_ox^n_ox
```

Species rates follow from stoichiometry (same as EDM):
- `ω_oxidizer = s · ω_fuel`
- `ω_product = -(1 + s) · ω_fuel`

# Arguments
- `reaction::CollocatedArrheniusReaction{T}` — kinetic parameters
- `species_state::SpeciesState{NS, T}` — current mass fractions
- `combustion_props::CombustionProperties{NS, T}` — thermochemical properties
- `T_field::Union{CollocatedScalarField{T}, Vector{T}}` — temperature field
- `density::T` — fluid density
- `mesh::UnstructuredFVMMesh` — mesh

Returns `NTuple{NS, Vector{T}}` of reaction rates [kg/(m��·s)].
"""
function compute_arrhenius_reaction_rates(
        reaction::CollocatedArrheniusReaction{T},
        species_state::SpeciesState{NS, T},
        combustion_props::CombustionProperties{NS, T},
        T_field::Union{CollocatedScalarField{T}, Vector{T}},
        density::T,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T, NS}
    nc = length(mesh.cell_volumes)
    s = combustion_props.stoich_ratio
    R = T(_R_UNIVERSAL)

    fuel_idx = _species_index(combustion_props, :fuel)
    ox_idx = _species_index(combustion_props, :oxidizer)
    has_product = any(n -> n === :product, combustion_props.species_names)
    prod_idx = has_product ? _species_index(combustion_props, :product) : 0

    Y_fuel = species_state.Y[fuel_idx].internal
    Y_ox = species_state.Y[ox_idx].internal

    T_vals = T_field isa CollocatedScalarField ? T_field.internal : T_field

    omega = ntuple(_ -> zeros(T, nc), Val(NS))

    for c in 1:nc
        T_c = max(T_vals[c], T(200))  # floor temperature for stability

        # Arrhenius rate constant
        k_f = reaction.A * T_c^reaction.b * exp(-reaction.E_a / (R * T_c))

        # Fuel consumption rate
        Y_f_c = max(Y_fuel[c], zero(T))
        Y_o_c = max(Y_ox[c], zero(T))
        omega_fuel_c = -density * k_f * Y_f_c^reaction.n_fuel * Y_o_c^reaction.n_ox

        omega[fuel_idx][c] = omega_fuel_c
        omega[ox_idx][c] = s * omega_fuel_c
        if has_product
            omega[prod_idx][c] = -(one(T) + s) * omega_fuel_c
        end
    end

    return omega
end

# ── FR/ED blending ────────────────────────────────────────────────

"""
    compute_fred_reaction_rates(
        reaction, edm, species_state, combustion_props,
        T_field, k_turb, eps_turb, density, mesh,
    ) -> NTuple{NS, Vector{T}}

Compute reaction rates using the finite-rate / eddy-dissipation (FR/ED)
model: the effective rate at each cell is the minimum of the Arrhenius
and EDM rates (by magnitude).

This blending ensures that the reaction is chemistry-limited when
Arrhenius rates are slow (e.g., low temperature) and mixing-limited
when Arrhenius rates are fast (fully turbulent flame).
"""
function compute_fred_reaction_rates(
        reaction::CollocatedArrheniusReaction{T},
        edm::EddyDissipationModel{T},
        species_state::SpeciesState{NS, T},
        combustion_props::CombustionProperties{NS, T},
        T_field::Union{CollocatedScalarField{T}, Vector{T}},
        k_turb::Union{Nothing, Vector{T}},
        eps_turb::Union{Nothing, Vector{T}},
        density::T,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T, NS}
    # Compute both rate sets
    omega_arr = compute_arrhenius_reaction_rates(
        reaction, species_state, combustion_props, T_field, density, mesh,
    )
    omega_edm = compute_edm_reaction_rates(
        edm, species_state, combustion_props, k_turb, eps_turb, density, mesh,
    )

    nc = length(mesh.cell_volumes)
    fuel_idx = _species_index(combustion_props, :fuel)
    s = combustion_props.stoich_ratio
    has_product = any(n -> n === :product, combustion_props.species_names)
    prod_idx = has_product ? _species_index(combustion_props, :product) : 0

    # Blend: take min magnitude (most limiting) for fuel rate
    omega = ntuple(_ -> zeros(T, nc), Val(NS))
    for c in 1:nc
        # Both rates are negative for fuel consumption; take the one closest to zero
        omega_f = max(omega_arr[fuel_idx][c], omega_edm[fuel_idx][c])
        omega[fuel_idx][c] = omega_f
        ox_idx = _species_index(combustion_props, :oxidizer)
        omega[ox_idx][c] = s * omega_f
        if has_product
            omega[prod_idx][c] = -(one(T) + s) * omega_f
        end
    end

    return omega
end

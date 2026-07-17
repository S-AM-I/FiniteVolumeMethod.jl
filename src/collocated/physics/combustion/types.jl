# combustion/types.jl — Core types for combustion and species transport
#
# Defines thermochemical properties, species state fields, and the
# Eddy Dissipation Model (EDM) for turbulence-chemistry interaction.

# ── Combustion properties ───────────────────────────────────────────

"""
    CombustionProperties{NS, T}

Thermochemical properties for a multi-species reacting system.

# Type parameters
- `NS` — number of species
- `T` — floating-point type

# Fields
- `species_names::NTuple{NS, Symbol}` — species identifiers (e.g. `:fuel`, `:oxidizer`, `:product`)
- `molecular_weights::NTuple{NS, T}` — molar masses `M_i` [kg/mol]
- `diffusivities::NTuple{NS, T}` — laminar mass diffusivities `D_i` [m²/s]
- `Sc_t::T` — turbulent Schmidt number
- `stoich_ratio::T` — mass stoichiometric oxidizer-to-fuel ratio
- `heat_of_combustion::T` — heat of combustion `ΔH` [J/kg_fuel] (positive = exothermic)
"""
struct CombustionProperties{NS, T}
    species_names::NTuple{NS, Symbol}
    molecular_weights::NTuple{NS, T}
    diffusivities::NTuple{NS, T}
    Sc_t::T
    stoich_ratio::T
    heat_of_combustion::T
end

"""
    CombustionProperties(; species_names, molecular_weights, diffusivities, Sc_t, stoich_ratio, heat_of_combustion)

Construct [`CombustionProperties`](@ref) with keyword arguments.

Defaults correspond to a simple CH4/O2/CO2 system:
- 3 species: `:fuel`, `:oxidizer`, `:product`
- Molecular weights: 16, 32, 44 g/mol
- Diffusivities: 2×10⁻⁵ m²/s (all species)
- Turbulent Schmidt number: 0.7
- Stoichiometric O/F ratio: 4.0 (kg O₂ per kg CH₄)
- Heat of combustion: 50 MJ/kg
"""
function CombustionProperties(;
        species_names::NTuple{NS, Symbol} = (:fuel, :oxidizer, :product),
        molecular_weights::NTuple{NS, Real} = (16.0, 32.0, 44.0),
        diffusivities::NTuple{NS, Real} = (2.0e-5, 2.0e-5, 2.0e-5),
        Sc_t::Real = 0.7,
        stoich_ratio::Real = 4.0,
        heat_of_combustion::Real = 50.0e6,
    ) where {NS}
    T = promote_type(
        eltype(molecular_weights), eltype(diffusivities),
        typeof(Sc_t), typeof(stoich_ratio), typeof(heat_of_combustion),
    )
    return CombustionProperties{NS, T}(
        species_names,
        NTuple{NS, T}(molecular_weights),
        NTuple{NS, T}(diffusivities),
        T(Sc_t),
        T(stoich_ratio),
        T(heat_of_combustion),
    )
end

# ── Species state ───────────────────────────────────────────────────

"""
    SpeciesState{NS, T}

Mutable state holding mass fraction fields for all species.

# Type parameters
- `NS` — number of species
- `T` — floating-point type

# Fields
- `Y::NTuple{NS, CollocatedScalarField{T}}` — mass fraction fields per species
"""
mutable struct SpeciesState{NS, T}
    Y::NTuple{NS, CollocatedScalarField{T}}
end

"""
    SpeciesState(mesh, combustion_props; Y_init...)

Construct a [`SpeciesState`](@ref) on `mesh` with optional initial mass fractions.

Keyword arguments set initial values per species name, e.g.
`SpeciesState(mesh, props; fuel = 0.0, oxidizer = 0.233)`.
Unspecified species default to 0.
"""
function SpeciesState(
        mesh::UnstructuredFVMMesh{Dim, T},
        combustion_props::CombustionProperties{NS, T};
        kwargs...,
    ) where {Dim, T, NS}
    fields = ntuple(Val(NS)) do i
        name = combustion_props.species_names[i]
        init_val = T(get(kwargs, name, zero(T)))
        return CollocatedScalarField(name, mesh; value = init_val)
    end
    return SpeciesState{NS, T}(fields)
end

# ── Eddy Dissipation Model ──────────────────────────────────────────

"""
    EddyDissipationModel{T}

Magnussen-Hjertager Eddy Dissipation Model for turbulence-chemistry
interaction.

The EDM assumes that the reaction rate is controlled by turbulent
mixing rather than finite-rate chemistry. The rate depends on the
ratio ε/k from the RANS turbulence model.

# Fields
- `A_edm::T` — EDM constant for mixing-limited rate (default 4.0)
- `B_edm::T` — EDM constant for product-limited rate (default 0.5)
"""
struct EddyDissipationModel{T}
    A_edm::T
    B_edm::T
end

"""
    EddyDissipationModel(; A_edm = 4.0, B_edm = 0.5)

Construct an [`EddyDissipationModel`](@ref) with default constants.
"""
function EddyDissipationModel(; A_edm::Real = 4.0, B_edm::Real = 0.5)
    T = promote_type(typeof(A_edm), typeof(B_edm))
    return EddyDissipationModel{T}(T(A_edm), T(B_edm))
end

# ── Eddy Dissipation Concept (placeholder) ─────────────────────────

"""
    EddyDissipationConcept{T}

EDC fine-structure reactor model (Magnussen, 2005).

Unlike the simpler [`EddyDissipationModel`](@ref), EDC resolves a
fine-structure volume fraction and residence time derived from
turbulence quantities, enabling finite-rate chemistry within the
fine structures.

The implementation is in `combustion/edc.jl` via
[`compute_edc_reaction_rates`](@ref).

# Fields
- `C_gamma::T` — fine-structure volume fraction constant (default 2.1377)
- `C_tau::T` — fine-structure residence time constant (default 0.4082)
"""
struct EddyDissipationConcept{T}
    C_gamma::T
    C_tau::T
end

"""
    EddyDissipationConcept(; C_gamma = 2.1377, C_tau = 0.4082)

Construct an [`EddyDissipationConcept`](@ref) with default constants.
"""
function EddyDissipationConcept(; C_gamma::Real = 2.1377, C_tau::Real = 0.4082)
    T = promote_type(typeof(C_gamma), typeof(C_tau))
    return EddyDissipationConcept{T}(T(C_gamma), T(C_tau))
end

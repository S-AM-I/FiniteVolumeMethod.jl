# Phase 8: Combustion & Species Transport

**Date**: 2026-04-06
**Status**: Design
**Depends on**: Phase 0 (collocated operators), Phase 1 (incompressible NS), Phase 2a (RANS turbulence), Phase 3 (energy equation)

## Goal

Add multi-species transport with reaction source terms and the Eddy Dissipation Model (EDM) for turbulence-chemistry interaction. Couples heat release to the energy equation for reacting flow simulation.

## Architecture

### Four Files in `src/combustion/`

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `types.jl` | CombustionProperties, SpeciesState, EddyDissipationModel | ~100 |
| `species_transport.jl` | assemble_species!, solve_species! for each mass fraction | ~100 |
| `edm.jl` | EDM reaction rate: min(Arrhenius, mixing-limited) | ~100 |
| `solvers.jl` | solve_simple_reacting wrapper combining flow + turbulence + species + energy | ~180 |

Wired into Layer 2 after radiation includes.

## Type Design

### Combustion Properties

```julia
struct CombustionProperties{NS, T}
    species_names::NTuple{NS, Symbol}       # e.g., (:fuel, :oxidizer, :product)
    molecular_weights::NTuple{NS, T}        # M_i [kg/mol]
    diffusivities::NTuple{NS, T}            # D_i laminar mass diffusivity [m²/s]
    Sc_t::T                                  # turbulent Schmidt number (default 0.7)
    stoich_ratio::T                          # mass stoichiometric O/F ratio
    heat_of_combustion::T                    # ΔH [J/kg_fuel] (positive = exothermic)
end
```

Convenience constructor for a simple fuel/oxidizer/product system:
```julia
CombustionProperties(; 
    species_names = (:fuel, :oxidizer, :product),
    molecular_weights = (16.0, 32.0, 44.0),  # CH4, O2, CO2
    diffusivities = (2.0e-5, 2.0e-5, 2.0e-5),
    Sc_t = 0.7,
    stoich_ratio = 4.0,  # kg_O2 / kg_fuel
    heat_of_combustion = 50e6,  # J/kg_fuel (methane)
)
```

### Species State

```julia
mutable struct SpeciesState{NS, T}
    Y::NTuple{NS, CollocatedScalarField{T}}  # mass fractions
end
```

Constructor from mesh + initial values:
```julia
SpeciesState(mesh, combustion_props; Y_init...)
```

### Eddy Dissipation Model

```julia
struct EddyDissipationModel{T}
    A_edm::T   # EDM constant (default 4.0)
    B_edm::T   # product-limited constant (default 0.5)
end

EddyDissipationModel(; A_edm = 4.0, B_edm = 0.5)
```

## Species Transport Equation

For each species i (incompressible, constant ρ):
```
∂Y_i/∂t + div(phi · Y_i) = div(D_eff_i · grad(Y_i)) + ω_i / ρ
```

where `D_eff_i = D_i + ν_t / Sc_t` is the effective mass diffusivity.

Assembly per species:
```julia
function assemble_species!(
    eq::CollocatedEquation{T},
    Y_i::CollocatedScalarField{T},
    phi::FaceFluxField{T},
    D_eff::Union{T, Vector{T}},
    mesh::UnstructuredFVMMesh{Dim, T},
    bcs_Yi::Dict{Symbol, <:AbstractBoundaryCondition};
    dt::Union{Nothing, T} = nothing,
)
```

Steps:
1. `assemble_convection!(eq, phi, mesh, bcs_Yi)` — advection
2. `assemble_laplacian!(eq, D_eff, mesh, bcs_Yi)` — diffusion
3. If transient: `assemble_ddt_euler!(eq, one(T), Y_i.internal, mesh, dt)`

The reaction source `ω_i / ρ` is added explicitly to the RHS after assembly.

### Solve All Species

```julia
function solve_species!(
    species_state::SpeciesState{NS, T},
    phi::FaceFluxField{T},
    combustion_props::CombustionProperties{NS, T},
    reaction_rates::NTuple{NS, Vector{T}},
    nu_t::Union{Nothing, Vector{T}},
    density::T,
    mesh::UnstructuredFVMMesh{Dim, T},
    bcs_species::Dict{Symbol, Dict{Symbol, <:AbstractBoundaryCondition}};
    dt = nothing,
    linear_solver = nothing,
)
```

Loops over species, assembles and solves each transport equation, clips Y_i to [0, 1].

## Eddy Dissipation Model (EDM)

### Reaction Rate

For a simple one-step fuel + oxidizer → product reaction:

```julia
function compute_edm_reaction_rates(
    edm::EddyDissipationModel{T},
    species_state::SpeciesState{NS, T},
    combustion_props::CombustionProperties{NS, T},
    k_field::Union{Nothing, Vector{T}},   # turbulent kinetic energy (from RANS)
    eps_field::Union{Nothing, Vector{T}},  # dissipation rate (from RANS)
    mesh::UnstructuredFVMMesh{Dim, T},
) -> NTuple{NS, Vector{T}}  # reaction rate per species per cell
```

The EDM reaction rate for fuel consumption:
```
ω_fuel = -ρ · A · (ε/k) · min(Y_fuel, Y_oxidizer / s)
```

where `A` is the EDM constant and `s` is the stoichiometric ratio.

The product-limited rate (prevents reaction in product-rich regions):
```
ω_fuel_product = -ρ · A · B · (ε/k) · Y_product / (1 + s)
```

Final rate: `ω_fuel = min(ω_fuel_mixing, ω_fuel_product)` (most negative = slowest).

Species rates from stoichiometry:
- `ω_oxidizer = s · ω_fuel`
- `ω_product = -(1 + s) · ω_fuel`

When turbulence fields are not available (no RANS model), falls back to a simple finite-rate approach using a constant mixing time scale.

### Heat Release

```julia
function compute_heat_release(
    reaction_rates::NTuple{NS, Vector{T}},
    combustion_props::CombustionProperties{NS, T},
) -> Vector{T}
```

`S_h[c] = -ω_fuel[c] · ΔH` (positive for exothermic, since ω_fuel is negative for consumption).

Added to energy equation RHS: `eq.b[c] += S_h[c] * V_c / (ρ · Cp)`.

## Solver Integration

```julia
function solve_simple_reacting(
    prob::IncompressibleProblem{Dim, T},
    thermal_props::FluidThermalProperties{Dim, T},
    combustion_props::CombustionProperties{NS, T},
    edm::EddyDissipationModel{T};
    bcs_T, bcs_species,
    turb_model = nothing, turb_bcs = ...,
    Y_init = Dict{Symbol, T}(),
    T_init = thermal_props.T_ref,
    linear_solver = nothing, verbose = false,
) -> Tuple{SolveResult, ThermalState, SpeciesState}
```

SIMPLE loop:
```
for iter = 1:max_iterations:
    1. Momentum + pressure (with nu_eff from turbulence)
    2. Turbulence (provides k, ε for EDM)
    3. Compute EDM reaction rates
    4. Solve species transport with reaction sources
    5. Compute heat release from reaction rates
    6. Solve energy with heat release + radiation (if present)
    7. Check convergence
```

## Boundary Conditions for Species

```julia
bcs_species = Dict(
    :fuel => Dict(:inlet => ParabolicDirichlet(1.0), :wall => ParabolicNeumann(0.0), ...),
    :oxidizer => Dict(:inlet => ParabolicDirichlet(0.0), :coflow => ParabolicDirichlet(0.233), ...),
    :product => Dict(:inlet => ParabolicDirichlet(0.0), :wall => ParabolicNeumann(0.0), ...),
)
```

## Export List

```julia
export CombustionProperties, SpeciesState, EddyDissipationModel
export assemble_species!, solve_species!
export compute_edm_reaction_rates, compute_heat_release
export solve_simple_reacting
```

## Validation

- **1D counterflow diffusion flame**: Fuel from left, oxidizer from right. Verify species profiles cross at stoichiometric mixture fraction. Verify temperature peak at flame location.
- **Well-stirred reactor**: Uniform mixing (no spatial gradients). Verify steady-state Y_fuel matches EDM prediction.

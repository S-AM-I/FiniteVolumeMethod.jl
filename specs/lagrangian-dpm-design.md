---
date: 2026-04-07
---

# Phase 11: Lagrangian DPM Enhancement

**Status**: Design
**Depends on**: Phase 0 (collocated operators), Phase 1 (incompressible NS)

## Goal

Enhance the existing Lagrangian particle tracking with drag force models, particle heat transfer, and two-way momentum/energy coupling between particles and the Eulerian fluid field.

## Architecture

### Four Files in `src/lagrangian/`

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `drag_models.jl` | AbstractDragModel, SchillerNaumann, StokesDrag, compute_drag_force | ~80 |
| `heat_transfer.jl` | RanzMarshall, compute_particle_heat_transfer | ~60 |
| `two_way_coupling.jl` | compute_momentum_source, compute_energy_source (PSI-cell) | ~80 |
| `particle_solver.jl` | advance_particles! (drag + heat + position integration) | ~120 |

Wired into Layer 2 after combustion includes.

## Existing Infrastructure

From `src/parabolic/particles.jl`:
- `LagrangianParticle{N, T}` — position, velocity, cell_index, id, active, properties dict
- `ParticleTracker{N, T}` — particle collection with ID counter
- `inject_particles!` — create particles at positions
- `advect_particles!` — move particles through mesh (basic Euler integration)
- `find_cell_index` — locate particle in mesh cell

The `properties::Dict{Symbol, Any}` field on `LagrangianParticle` is used to store particle physical properties (diameter, density, temperature, Cp, mass) without modifying the existing struct.

## Particle Properties Convention

Stored in `particle.properties`:
- `:diameter::Float64` — particle diameter [m]
- `:density::Float64` — particle material density [kg/m³]
- `:temperature::Float64` — particle temperature [K]
- `:Cp::Float64` — particle specific heat [J/(kg·K)]
- `:mass::Float64` — particle mass [kg] (= π/6 · d³ · ρ_p)

Helper to initialize:
```julia
function set_particle_properties!(p::LagrangianParticle; 
    diameter, density, temperature = 300.0, Cp = 1000.0)
    p.properties[:diameter] = diameter
    p.properties[:density] = density
    p.properties[:temperature] = temperature
    p.properties[:Cp] = Cp
    p.properties[:mass] = π / 6 * diameter^3 * density
end
```

## Drag Models

### Abstract Type

```julia
abstract type AbstractDragModel end
```

### Stokes Drag

```julia
struct StokesDrag <: AbstractDragModel end
```

Drag coefficient: `Cd = 24 / Re_p` (valid for Re_p << 1).

Force: `F_drag = 3π · μ · d · (U_f - U_p)` = `(m_p / τ_p) · (U_f - U_p)` where `τ_p = ρ_p · d² / (18μ)`.

### Schiller-Naumann

```julia
struct SchillerNaumann <: AbstractDragModel end
```

Drag correction: `f(Re) = 1 + 0.15 · Re^0.687` for Re < 1000, capped at `f = 1 + 0.15 · 1000^0.687` for Re ≥ 1000.

Force: `F_drag = (m_p / τ_p) · f(Re_p) · (U_f - U_p)`

### Particle Reynolds Number

```julia
Re_p = ρ_f · |U_f - U_p| · d / μ_f
```

### Compute Drag Force

```julia
function compute_drag_force(
    model::AbstractDragModel,
    U_fluid::SVector{Dim, T},
    U_particle::SVector{Dim, T},
    diameter::T, density_p::T,
    rho_f::T, mu_f::T,
) -> SVector{Dim, T}
```

## Particle Heat Transfer

### Ranz-Marshall Correlation

```julia
struct RanzMarshall <: AbstractHeatTransferModel end
```

Nusselt number: `Nu = 2 + 0.6 · Re_p^0.5 · Pr^0.33`

Heat transfer rate: `q = π · d · k_f · Nu · (T_f - T_p)` [W]

Temperature change: `dT_p/dt = q / (m_p · Cp_p)`

```julia
function compute_particle_heat_transfer(
    model::RanzMarshall,
    T_fluid::T, T_particle::T,
    U_fluid::SVector{Dim, T}, U_particle::SVector{Dim, T},
    diameter::T, rho_f::T, mu_f::T, k_f::T, Pr::T,
) -> T  # heat transfer rate q [W]
```

## Two-Way Coupling

### PSI-Cell Method

Particle forces and heat transfer are distributed to the fluid cell containing the particle:

```julia
function compute_momentum_source(
    tracker::ParticleTracker{Dim, T},
    drag_model::AbstractDragModel,
    U::CollocatedVectorField{Dim, T},
    rho_f::T, mu_f::T,
    mesh::UnstructuredFVMMesh{Dim, T},
) -> Vector{SVector{Dim, T}}  # per-cell source
```

Per cell: `S_mom[c] = -(1/V_c) · Σ_{p in c} F_drag_p`

The negative sign is because the drag force on the particle is reaction force on the fluid.

```julia
function compute_energy_source(
    tracker::ParticleTracker{Dim, T},
    heat_model,
    T_field::CollocatedScalarField{T},
    U::CollocatedVectorField{Dim, T},
    rho_f::T, mu_f::T, k_f::T, Pr::T,
    mesh::UnstructuredFVMMesh{Dim, T},
) -> Vector{T}  # per-cell source
```

Per cell: `S_energy[c] = -(1/V_c) · Σ_{p in c} q_p`

## Particle Solver

### Single Time Step Advancement

```julia
function advance_particles!(
    tracker::ParticleTracker{Dim, T},
    U::CollocatedVectorField{Dim, T},
    mesh::UnstructuredFVMMesh{Dim, T},
    dt::T;
    drag_model::AbstractDragModel = SchillerNaumann(),
    heat_model = nothing,
    T_field::Union{Nothing, CollocatedScalarField{T}} = nothing,
    rho_f::T = one(T),
    mu_f::T = T(1e-3),
    k_f::T = T(0.026),
    Pr::T = T(0.7),
    gravity::SVector{Dim, T} = zero(SVector{Dim, T}),
)
```

For each active particle:
1. Get fluid velocity at particle cell: `U_f = U.internal[p.cell_index]`
2. Compute drag force: `F_drag = compute_drag_force(drag_model, U_f, p.velocity, ...)`
3. Compute gravity force: `F_grav = m_p · g`
4. Update velocity: `U_p_new = U_p + dt · (F_drag + F_grav) / m_p`
5. Update position: `x_p_new = x_p + dt · U_p_new`
6. If heat model: compute heat transfer, update particle temperature
7. Update cell index (find new containing cell)
8. Deactivate if particle leaves domain

## Integration with Flow Solver

The particle solver is called between flow iterations or time steps. The two-way coupling sources are passed as `body_force` to `assemble_momentum!` and as explicit source to the energy equation.

No dedicated wrapper solver is provided — users call `advance_particles!` and `compute_momentum_source`/`compute_energy_source` within their time loop. This matches the modular pattern and avoids coupling to specific solver wrappers.

## Export List

```julia
# Drag
export AbstractDragModel, StokesDrag, SchillerNaumann, compute_drag_force

# Heat transfer
export AbstractParticleHeatTransfer, RanzMarshall, compute_particle_heat_transfer

# Two-way coupling
export compute_momentum_source, compute_energy_source

# Particle solver
export advance_particles!, set_particle_properties!
```

## Validation

- **Single particle settling**: Sphere falling under gravity in quiescent fluid. Terminal velocity `U_t = (ρ_p - ρ_f) · g · d² / (18μ)` for Stokes drag.
- **Particle in uniform crossflow**: Verify drag force magnitude matches Schiller-Naumann correlation at known Re.

---
date: 2026-04-06
---

# Phase 9: Radiation

**Status**: Design
**Depends on**: Phase 0 (collocated operators), Phase 3 (energy equation)

## Goal

Add thermal radiation modeling via the P1 approximation. Solves a single diffusion equation for incident radiation G, computes the radiative source term for the energy equation, and provides a combined thermal+radiation solver wrapper.

## Architecture

### Three Files in `src/radiation/`

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `types.jl` | AbstractRadiationModel, P1Model, RadiationState, Stefan-Boltzmann constant | ~60 |
| `p1.jl` | assemble_p1!, solve_p1_radiation, compute_radiation_source | ~120 |
| `solvers.jl` | solve_simple_thermal_radiation wrapper | ~150 |

Wired into Layer 2 after multiphase includes.

## Type Design

### Radiation Model

```julia
abstract type AbstractRadiationModel end

struct P1Model{T} <: AbstractRadiationModel
    a::T  # absorption coefficient [1/m]
end
```

The absorption coefficient is constant (spatially uniform). A future enhancement could accept per-cell `Vector{T}` for participating media with variable absorption (e.g., WSGGM).

Convenience constructor:
```julia
P1Model(; a = 0.1)
```

### Radiation State

```julia
mutable struct RadiationState{T}
    G::CollocatedScalarField{T}  # incident radiation [W/m²]
end

RadiationState(mesh; G_init = 0.0)
```

### Constants

```julia
const STEFAN_BOLTZMANN = 5.670374419e-8  # σ [W/(m²·K⁴)]
```

## P1 Radiation Equation

### Governing Equation

```
-div(Γ · grad(G)) + a·G = 4·a·σ·T⁴
```

where `Γ = 1/(3a)` is the radiation diffusion coefficient.

Rearranging for the collocated assembly framework (where Laplacian assembles `-div(Γ·grad)`):

```
A_laplacian·G + a·V·G = 4·a·σ·T⁴·V
```

The Laplacian contributes to diagonal (positive-definite), absorption `a·V` adds to diagonal, and emission `4aσT⁴·V` goes to RHS.

### Assembly

```julia
function assemble_p1!(
    eq::CollocatedEquation{T},
    rad_model::P1Model{T},
    T_field::CollocatedScalarField{T},
    mesh::UnstructuredFVMMesh{Dim, T},
    bcs_G::Dict{Symbol, <:AbstractBoundaryCondition},
)
```

Steps:
1. `assemble_laplacian!(eq, Γ, mesh, bcs_G)` where `Γ = 1/(3a)`
2. Absorption (implicit): `eq.A[c,c] += a · V_c` for all cells
3. Emission (explicit RHS): `eq.b[c] += 4 · a · σ · T[c]⁴ · V_c`

### Standalone Solver

```julia
function solve_p1_radiation(
    rad_model::P1Model{T},
    T_field::CollocatedScalarField{T},
    mesh::UnstructuredFVMMesh{Dim, T},
    bcs_G::Dict{Symbol, <:AbstractBoundaryCondition};
    linear_solver = nothing,
) -> CollocatedScalarField{T}
```

Assembles and solves the P1 equation, returns the G field.

### Radiation Source for Energy

```julia
function compute_radiation_source(
    rad_model::P1Model{T},
    G::CollocatedScalarField{T},
    T_field::CollocatedScalarField{T},
) -> Vector{T}
```

Per cell: `S_rad[c] = a · G[c] - 4 · a · σ · T[c]⁴`

Positive means net radiative absorption (fluid heats up). Negative means net emission (fluid cools).

This source is added to the energy equation's RHS:
```
eq_energy.b[c] += S_rad[c] * V_c / (rho * Cp)
```
(divided by ρ·Cp since the energy equation is scaled by 1/(ρ·Cp)).

## Boundary Conditions for G

### Marshak BC (Opaque Wall)

The standard P1 wall BC is the Marshak condition:
```
G + (2/3a) · ∂G/∂n = 4σT_wall⁴
```

This maps to `ParabolicRobin(1, 2/(3a), 4σT_wall⁴)` where the Robin BC is `a·G + b·∂G/∂n = c`.

Convenience constructor:
```julia
function marshak_wall_bc(rad_model::P1Model{T}, T_wall::T) -> ParabolicRobin
    return ParabolicRobin(1.0, 2.0 / (3.0 * rad_model.a), 4.0 * STEFAN_BOLTZMANN * T_wall^4)
end
```

### Inlet/Outlet

Fixed incident radiation from known temperature:
```julia
radiation_inlet_bc(T_inlet) = ParabolicDirichlet(4 * STEFAN_BOLTZMANN * T_inlet^4)
```

## Solver Integration

### Combined Thermal + Radiation Solver

```julia
function solve_simple_thermal_radiation(
    prob::IncompressibleProblem{Dim, T},
    thermal_props::FluidThermalProperties{Dim, T},
    rad_model::P1Model{T};
    bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
    bcs_G::Dict{Symbol, <:AbstractBoundaryCondition},
    turb_model = nothing,
    turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
    T_init = thermal_props.T_ref,
    linear_solver = nothing,
    verbose = false,
) -> Tuple{SolveResult{Dim, T}, ThermalState{T}, RadiationState{T}}
```

Loop:
```
for iter = 1:max_iterations:
    1. Update k_eff, nu_eff, buoyancy (same as Phase 3)
    2. Momentum + pressure + correction
    3. Turbulence (optional)
    4. Solve energy with radiation source in RHS
    5. Solve P1 radiation for G
    6. Update radiation source from new G and T
    7. Check convergence
```

The radiation source `S_rad` computed from the previous iteration's G and T is added to the energy equation explicitly. This lagged coupling is standard and converges within the outer SIMPLE iterations.

## Export List

```julia
# Types
export AbstractRadiationModel, P1Model, RadiationState, STEFAN_BOLTZMANN

# P1 equation
export assemble_p1!, solve_p1_radiation, compute_radiation_source

# BCs
export marshak_wall_bc, radiation_inlet_bc

# Solver
export solve_simple_thermal_radiation
```

## Validation

- **1D slab**: Two parallel plates at T_hot and T_cold with participating medium (constant a). Analytical G profile is exponential. Compare numerical G and radiative heat flux to analytical solution.
- **Optically thick limit**: For large a·L (optical thickness), the P1 model should recover diffusion-like behavior with radiative conductivity k_rad = 16σT³/(3a).

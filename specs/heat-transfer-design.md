---
date: 2026-04-06
---

# Phase 3: Conjugate Heat Transfer & Buoyancy

**Status**: Design
**Depends on**: Phase 1 (Incompressible NS) — complete, Phase 2a (RANS Turbulence) — complete

## Goal

Add thermal modeling to the incompressible solver: fluid energy transport with turbulent heat flux, Boussinesq buoyancy for natural convection, solid-region conduction, and Dirichlet-Neumann conjugate coupling between fluid and solid domains.

## Architecture

### Three Components

1. **Fluid energy equation** — scalar transport of temperature, solved segregated after turbulence in SIMPLE/PISO/PIMPLE. Uses Phase 0 operators (convection, Laplacian, ddt).
2. **Boussinesq buoyancy** — body force `-ρ_ref·β·(T - T_ref)·g` added to momentum RHS via a new `body_force` keyword on `assemble_momentum!`.
3. **Solid conduction + conjugate coupling** — standalone Laplacian solve on a solid mesh, coupled to the fluid via Dirichlet-Neumann iteration at the interface.

### Integration Pattern

Same as Phase 2a turbulence: wrapper solvers that call existing building blocks with extra steps.

```
SIMPLE iteration (thermal):
    1. Compute nu_eff, k_eff, buoyancy force
    2. Assemble + solve momentum with nu_eff + body_force
    3. Pressure solve + correction
    4. Solve turbulence equations (if turbulent)
    5. Solve energy equation with k_eff          ← NEW
    6. Update k_eff from nu_t and Pr_t           ← NEW
    7. Update buoyancy force from T              ← NEW
    8. Check convergence
```

### File Layout

All new files in `src/thermal/`:

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `types.jl` | Thermal property types, ThermalState, problem definitions | ~120 |
| `energy_equation.jl` | `assemble_energy!`, `solve_energy!` for fluid T transport | ~100 |
| `buoyancy.jl` | `compute_buoyancy_source` body force vector | ~40 |
| `solid_conduction.jl` | `assemble_solid_conduction!`, `solve_solid_conduction` | ~80 |
| `conjugate.jl` | `solve_conjugate_ht` Dirichlet-Neumann iteration | ~120 |
| `solvers.jl` | `solve_simple_thermal`, `solve_incompressible_thermal` | ~200 |

Wired into Layer 2 after turbulence includes.

## Type Design

### Fluid Thermal Properties

```julia
struct FluidThermalProperties{Dim, T}
    Cp::T               # specific heat capacity [J/(kg·K)]
    k::T                # laminar thermal conductivity [W/(m·K)]
    Pr_t::T             # turbulent Prandtl number (default 0.85)
    beta::T             # thermal expansion coefficient [1/K] (0 = no buoyancy)
    T_ref::T            # reference temperature for Boussinesq [K]
    g::SVector{Dim, T}  # gravity vector [m/s²]
end
```

Convenience constructor with keyword defaults:
```julia
FluidThermalProperties(; Cp=1005.0, k=0.026, Pr_t=0.85, beta=0.0, T_ref=300.0,
    g=SVector(0.0, -9.81))
```

When `beta == 0`, buoyancy is disabled (forced convection only).

### Thermal State

```julia
mutable struct ThermalState{T}
    T_field::CollocatedScalarField{T}  # temperature field
    k_eff::Vector{T}                    # effective conductivity per cell
end
```

Constructed from mesh with initial temperature:
```julia
ThermalState(mesh; T_init = 300.0)
```

### Effective Conductivity

```julia
k_eff[c] = k_laminar + rho * Cp * nu_t[c] / Pr_t
```

When no turbulence model is present, `nu_t` is zero and `k_eff = k_laminar`.

### Solid Thermal Properties

```julia
struct SolidThermalProperties{T}
    rho::T      # density [kg/m³]
    Cp::T       # specific heat [J/(kg·K)]
    k::T        # thermal conductivity [W/(m·K)]
    Q_gen::T    # volumetric heat generation [W/m³] (default 0)
end
```

### Conjugate Heat Transfer Problem

```julia
struct ConjugateHeatTransferProblem{Dim, T, FM, SM}
    # Fluid side
    fluid_prob::IncompressibleProblem{Dim, T}
    fluid_thermal::FluidThermalProperties{Dim, T}
    fluid_bcs_T::Dict{Symbol, AbstractBoundaryCondition}  # temperature BCs
    # Solid side
    solid_mesh::SM
    solid_thermal::SolidThermalProperties{T}
    solid_bcs_T::Dict{Symbol, AbstractBoundaryCondition}
    # Coupling
    interface_fluid_patch::Symbol  # patch name on fluid mesh
    interface_solid_patch::Symbol  # patch name on solid mesh
    # Iteration control
    max_coupling_iterations::Int   # default 50
    coupling_tolerance::T          # default 1e-4
end
```

## Energy Equation

### Transport Equation

For incompressible flow with constant `ρ·Cp`:
```
ρ·Cp · (∂T/∂t + div(phi·T)) = div(k_eff · grad(T)) + S_h
```

In the collocated framework, `phi` is the face volumetric flux from the incompressible solver. The equation is assembled as:

```julia
function assemble_energy!(
    eq::CollocatedEquation{T},
    T_field::CollocatedScalarField{T},
    phi::FaceFluxField{T},
    k_eff::Union{T, Vector{T}},
    mesh::UnstructuredFVMMesh{Dim, T},
    bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
    rho_Cp::T = one(T),
    dt::Union{Nothing, T} = nothing,
)
```

Assembly steps:
1. `assemble_convection!(eq, phi, mesh, bcs_T)` — convective transport `div(phi·T)`
2. `assemble_laplacian!(eq, k_eff, mesh, bcs_T)` — diffusion `div(k_eff·grad(T))`
3. If transient: `assemble_ddt_euler!(eq, rho_Cp, T_old, mesh, dt)`

Note: `rho_Cp` scales the convection and temporal terms. For the convection term, the face flux `phi` already carries the volume flow rate, so `rho_Cp · div(phi·T)` is the convective energy flux. The Laplacian uses `k_eff` directly (not scaled by `rho_Cp`) since it represents `div(k·grad(T))`.

Actually, the standard form puts `rho·Cp` on both convection and temporal terms, while diffusion has `k`. In our `CollocatedEquation`, convection and diffusion both contribute to the same `A` matrix. The cleanest approach: scale the entire equation by `1/(rho·Cp)` so convection uses `phi` directly and diffusion uses `alpha_eff = k_eff/(rho·Cp)` (thermal diffusivity). This matches OpenFOAM's `alphaEff` approach.

Revised assembly:
1. `assemble_convection!(eq, phi, mesh, bcs_T)` — uses phi directly (already volumetric)
2. `assemble_laplacian!(eq, alpha_eff, mesh, bcs_T)` — where `alpha_eff[c] = k_eff[c] / (rho * Cp)`
3. If transient: `assemble_ddt_euler!(eq, one(T), T_old, mesh, dt)` — density=1 since we divided by rho·Cp

### Effective Conductivity Update

```julia
function update_k_eff!(
    thermal_state::ThermalState{T},
    thermal_props::FluidThermalProperties{Dim, T},
    nu_t::Union{Nothing, Vector{T}},
    density::T,
)
    k_lam = thermal_props.k
    for c in eachindex(thermal_state.k_eff)
        k_t = nu_t === nothing ? zero(T) : density * thermal_props.Cp * nu_t[c] / thermal_props.Pr_t
        thermal_state.k_eff[c] = k_lam + k_t
    end
end
```

## Boussinesq Buoyancy

### Body Force

```julia
function compute_buoyancy_source(
    T_field::CollocatedScalarField{T},
    props::FluidThermalProperties{Dim, T},
    density::T,
) -> Vector{SVector{Dim, T}}
```

Returns per-cell force vector:
```
F_b[c] = -density * beta * (T[c] - T_ref) * g
```

When `beta == 0`, returns zeros (no allocation needed if caller checks).

### Momentum Integration

`assemble_momentum!` gets a `body_force` keyword:

```julia
function assemble_momentum!(eq, state, prob, component;
    dt = nothing, scheme = CONV_UPWIND, nu_eff = prob.nu,
    body_force::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
)
    # ... existing assembly ...
    
    # Body force (buoyancy, etc.)
    if body_force !== nothing
        for c in 1:nc
            eq.b[c] += body_force[c][component] * mesh.cell_volumes[c]
        end
    end
end
```

This is backward-compatible (default is `nothing`).

## Solid Conduction

### Equation

```
ρ·Cp · ∂T/∂t = div(k · grad(T)) + Q_gen
```

For steady state (no ∂T/∂t), this reduces to a single Laplacian solve:
```
div(k · grad(T)) = -Q_gen
```

### Assembly

```julia
function assemble_solid_conduction!(
    eq::CollocatedEquation{T},
    solid::SolidThermalProperties{T},
    mesh::UnstructuredFVMMesh{Dim, T},
    bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
    dt::Union{Nothing, T} = nothing,
    T_old::Union{Nothing, Vector{T}} = nothing,
)
    # Diffusion
    assemble_laplacian!(eq, solid.k, mesh, bcs_T)
    
    # Source
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        eq.b[c] += solid.Q_gen * mesh.cell_volumes[c]
    end
    
    # Temporal
    if dt !== nothing && T_old !== nothing
        rho_Cp = solid.rho * solid.Cp
        assemble_ddt_euler!(eq, rho_Cp, T_old, mesh, dt)
    end
end
```

### Standalone Solver

```julia
function solve_solid_conduction(
    mesh, solid_props, bcs_T;
    dt = nothing, T_old = nothing, linear_solver = nothing,
) -> CollocatedScalarField{T}
```

Returns the temperature field. For steady state, a single linear solve. For transient, called per time step.

## Conjugate Heat Transfer

### Dirichlet-Neumann Coupling

The fluid and solid share a boundary interface. The coupling iterates:

```
Initialize: T_interface = T_ref

for coupling_iter = 1:max_iterations:
    1. Set T_interface as Dirichlet BC on fluid interface patch
    2. Solve fluid (SIMPLE with energy) → get fluid T field
    3. Compute interface heat flux: q_n = -k_f · ∂T_f/∂n at interface faces
    4. Set q_n as Neumann BC on solid interface patch  
    5. Solve solid conduction → get solid T field
    6. Extract new T_interface from solid face values
    7. Under-relax: T_interface = (1-α)·T_old + α·T_new  (α ≈ 0.5)
    8. Check convergence: max|T_new - T_old| < tolerance
```

### Interface Heat Flux Computation

```julia
function compute_interface_heat_flux(
    T_field::CollocatedScalarField{T},
    k::T,
    mesh::UnstructuredFVMMesh{Dim, T},
    interface_patch::Symbol,
) -> Dict{Int, T}  # face_index => heat_flux
```

For each interface face: `q_f = -k · (T_boundary - T_cell) / d_cell_to_face`

### Solver

```julia
function solve_conjugate_ht(
    cht_prob::ConjugateHeatTransferProblem;
    turb_model = nothing,
    turb_bcs = ...,
    linear_solver = nothing,
    verbose = false,
) -> (SolveResult, ThermalState, CollocatedScalarField)
```

Returns fluid solve result, fluid thermal state, and solid temperature field.

## Thermal Solver Wrappers

### Fluid-Only (Forced/Natural Convection)

```julia
function solve_simple_thermal(
    prob::IncompressibleProblem{Dim, T},
    thermal_props::FluidThermalProperties{Dim, T};
    bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
    turb_model = nothing,
    turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
    T_init::T = thermal_props.T_ref,
    linear_solver = nothing,
    verbose = false,
) -> Tuple{SolveResult{Dim, T}, ThermalState{T}}
```

Loop:
1. `update_k_eff!` from `nu_t` and `Pr_t`
2. Compute buoyancy force if `beta > 0`
3. Assemble + solve momentum with `nu_eff` + `body_force`
4. Pressure solve + correction
5. Solve turbulence (if present)
6. `assemble_energy!` + solve → update `T_field`
7. Check convergence

Transient variant:
```julia
function solve_incompressible_thermal(
    prob, thermal_props, tspan, dt;
    bcs_T, turb_model = nothing, turb_bcs = ...,
    T_init = ..., linear_solver = nothing, verbose = false,
) -> Tuple{SolveResult, ThermalState}
```

## Boundary Conditions

Reuses existing BC types:
- `ParabolicDirichlet(T_value)` — fixed temperature (wall, inlet)
- `ParabolicNeumann(q_flux)` — fixed heat flux (insulated = 0, heated wall)
- `ParabolicRobin(h, 1, h*T_inf)` — convective BC `h·(T - T_inf)` where h is heat transfer coefficient

Convenience constructors:
```julia
thermal_inlet_bc(T_inlet) = ParabolicDirichlet(T_inlet)
thermal_insulated_bc() = ParabolicNeumann(0.0)
thermal_heated_wall_bc(q) = ParabolicNeumann(q)
thermal_convective_bc(h, T_inf) = ParabolicRobin(h, 1.0, h * T_inf)
```

## Export List

```julia
# Types
export FluidThermalProperties, SolidThermalProperties, ThermalState
export ConjugateHeatTransferProblem

# Energy equation
export assemble_energy!, solve_energy!, update_k_eff!

# Buoyancy
export compute_buoyancy_source

# Solid conduction
export assemble_solid_conduction!, solve_solid_conduction

# Conjugate
export solve_conjugate_ht, compute_interface_heat_flux

# Solver wrappers
export solve_simple_thermal, solve_incompressible_thermal

# BC convenience
export thermal_inlet_bc, thermal_insulated_bc, thermal_heated_wall_bc, thermal_convective_bc
```

## Validation

- **De Vahl Davis heated cavity**: Natural convection in a square cavity with differentially heated vertical walls. Compare Nusselt number at the hot wall to benchmark values at Ra = 10³, 10⁴. Pass: Nu within 5% of published data.
- **Conjugate pipe flow**: Heated solid cylinder with internal fluid flow. Compare interface temperature profile to analytical solution.

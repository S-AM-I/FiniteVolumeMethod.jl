# Phase 2a: RANS Turbulence Models

**Date**: 2026-04-06
**Status**: Design
**Depends on**: Phase 1 (Incompressible NS) — complete

## Goal

Add RANS turbulence models (k-ε, k-ω, k-ω SST, Spalart-Allmaras) to the incompressible solver. Each model adds 1-2 scalar transport equations solved segregated within the SIMPLE/PISO/PIMPLE loop, producing a per-cell turbulent viscosity `ν_t` that modifies the momentum equation's effective viscosity.

## Architecture

### Integration with Phase 1

The SIMPLE/PISO/PIMPLE loops gain a turbulence hook after velocity correction:

```
for each SIMPLE iteration (or time step):
    1. Assemble + solve momentum with nu_eff = nu + nu_t
    2. Pressure solve + correction
    3. Solve turbulence transport equations         ← NEW
    4. Update nu_t from turbulence fields           ← NEW
    5. Check convergence
```

The momentum assembler `assemble_momentum!` currently takes `prob.nu` (scalar). With turbulence, it uses `nu_eff::Vector{T}` (per-cell). The Phase 0 `assemble_laplacian!` already accepts `Vector{T}` for variable diffusivity, so no changes to Phase 0 are needed.

### Approach: Separate turbulence solver functions

Rather than modifying `IncompressibleProblem` (which would break the existing API), the turbulence integration is provided as wrapper solvers:

```julia
solve_simple_turbulent(prob, turb_model; kwargs...)
solve_incompressible_turbulent(prob, turb_model, tspan, dt; kwargs...)
```

These call the same building blocks (assemble_momentum!, pressure solve, correction) but add the turbulence step. The `assemble_momentum!` function gets an optional `nu_eff` keyword argument.

### File Layout

All new files in `src/turbulence/`:

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `interface.jl` | Abstract types, dispatch interface, RANSTurbulenceState | ~100 |
| `strain_rate.jl` | Strain rate tensor from velocity gradients | ~60 |
| `k_epsilon_rans.jl` | k-ε collocated assembly (wraps existing StandardKEpsilon) | ~120 |
| `k_omega.jl` | Standard k-ω (Wilcox 1988) | ~130 |
| `k_omega_sst.jl` | k-ω SST (extends existing KappaOmegaSST) | ~180 |
| `spalart_allmaras.jl` | One-equation SA model | ~150 |
| `wall_functions.jl` | Wall treatment for collocated solver | ~80 |
| `solvers.jl` | Turbulent SIMPLE/PISO/PIMPLE wrappers | ~150 |

Wired into Layer 2 after incompressible includes.

## Type Design

### Abstract Hierarchy

```julia
abstract type AbstractTurbulenceModel end
abstract type AbstractRANSModel <: AbstractTurbulenceModel end
# Future: abstract type AbstractLESModel <: AbstractTurbulenceModel end
```

`AbstractTurbulenceModel` is already exported from the parabolic solver family but with a different meaning (it's used for `ParabolicKEpsilon`). We reuse the same name since it's the correct abstraction — the collocated RANS models will be new subtypes.

### Concrete RANS Model Types

**k-ε**: Reuse `StandardKEpsilon` directly. It already has all coefficients. We just add collocated assembly methods that dispatch on it.

**k-ω (Wilcox)**:
```julia
struct KOmega{T} <: AbstractRANSModel
    beta_star::T    # 0.09 — k destruction coefficient
    alpha::T        # 5/9 — ω production coefficient
    beta::T         # 3/40 — ω destruction coefficient
    sigma_k::T      # 0.5 — k diffusion Prandtl number
    sigma_omega::T  # 0.5 — ω diffusion Prandtl number
end
```

**k-ω SST**: Wraps existing `KappaOmegaSST` coefficients, adds the blending/limiter logic:
```julia
struct KOmegaSSTModel{T} <: AbstractRANSModel
    coeffs::KappaOmegaSST{T}
end
```
Named `KOmegaSSTModel` to avoid collision with the existing `KappaOmegaSST` coefficients struct.

**Spalart-Allmaras**:
```julia
struct SpalartAllmaras{T} <: AbstractRANSModel
    cb1::T     # 0.1355
    cb2::T     # 0.622
    sigma::T   # 2/3
    kappa::T   # 0.41
    cw2::T     # 0.3
    cw3::T     # 2.0
    cv1::T     # 7.1
    ct3::T     # 1.2
    ct4::T     # 0.5
end
```

### Turbulence State

```julia
mutable struct RANSTurbulenceState{T}
    fields::Dict{Symbol, CollocatedScalarField{T}}
    nu_t::Vector{T}
end
```

Constructed per model:
- k-ε: fields `:k` and `:epsilon`
- k-ω / SST: fields `:k` and `:omega`
- SA: field `:nu_tilde`

### Required Interface (3 functions per model)

```julia
"""Compute turbulent viscosity from current turbulence fields."""
function turbulent_viscosity!(
    nu_t::Vector{T}, model::AbstractRANSModel,
    turb_state::RANSTurbulenceState{T},
    mesh::UnstructuredFVMMesh{Dim, T},
) where {Dim, T}

"""Solve turbulence transport equations (modifies turb_state in-place)."""
function solve_turbulence!(
    turb_state::RANSTurbulenceState{T},
    model::AbstractRANSModel,
    U::CollocatedVectorField{Dim, T},
    phi::FaceFluxField{T},
    nu::T,
    mesh::UnstructuredFVMMesh{Dim, T},
    bcs_turb::Dict{Symbol, Dict{Symbol, <:AbstractBoundaryCondition}};
    dt::Union{Nothing, T} = nothing,
    linear_solver = nothing,
) where {Dim, T}

"""Return number of turbulence fields."""
n_turbulence_fields(::AbstractRANSModel) -> Int
"""Return names of turbulence fields."""
turbulence_field_names(::AbstractRANSModel) -> Tuple{Vararg{Symbol}}
```

## Model Details

### k-ε (Standard)

Transport equations:
```
∂k/∂t + div(phi*k) = div((ν + ν_t/σ_k) * grad(k)) + P_k - ε
∂ε/∂t + div(phi*ε) = div((ν + ν_t/σ_ε) * grad(ε)) + C1ε*(ε/k)*P_k - C2ε*(ε²/k)
```

Assembly per equation:
1. `assemble_convection!(eq, phi, mesh, bcs_k)` — convective transport
2. `assemble_laplacian!(eq, gamma_k, mesh, bcs_k)` — diffusion with `γ_k[c] = ν + ν_t[c]/σ_k`
3. Source term linearization (implicit for stability):
   - k: `S_C = P_k`, `S_P = -ε/k` → `eq.b[c] += P_k * V_c`, `eq.A[c,c] += (ε[c]/k[c]) * V_c`
   - ε: `S_C = C1ε*(ε/k)*P_k`, `S_P = -C2ε*(ε/k)` → similar pattern
4. If transient: `assemble_ddt_euler!(eq, density, field_old, mesh, dt)`

Production `P_k = ν_t * |S|²` where `|S|` is the strain rate magnitude computed from velocity gradients.

Turbulent viscosity: `ν_t = C_μ * k² / ε` (reuses existing `compute_turbulent_viscosity`).

### k-ω (Wilcox 1988)

Transport equations:
```
∂k/∂t + div(phi*k) = div((ν + σ_k*ν_t) * grad(k)) + P_k - β*·k·ω
∂ω/∂t + div(phi*ω) = div((ν + σ_ω*ν_t) * grad(ω)) + α*(ω/k)*P_k - β·ω²
```

Turbulent viscosity: `ν_t = k / ω`

Source linearization:
- k: `S_C = P_k`, `S_P = -β*·ω`
- ω: `S_C = α*(ω/k)*P_k`, `S_P = -β·ω`

### k-ω SST (Menter)

Blends k-ω (near wall) with k-ε (far field) using blending function F1:
```
φ = F1·φ_1 + (1-F1)·φ_2
```
where φ is any model constant (σ_k, σ_ω, β, α).

Key additions over standard k-ω:
1. **F1 blending function**: based on distance to wall, k, ω, ν
2. **Cross-diffusion term** in ω equation: `2*(1-F1)*σ_ω2/ω * grad(k)·grad(ω)`
3. **SST viscosity limiter**: `ν_t = a1*k / max(a1*ω, |S|*F2)` where F2 is a second blending function

The blending requires wall distance `d_wall` per cell, computed once at setup.

### Spalart-Allmaras

Single equation for modified turbulent viscosity `ν̃`:
```
∂ν̃/∂t + div(phi*ν̃) = cb1·S̃·ν̃ + (1/σ)·[div((ν+ν̃)·grad(ν̃)) + cb2·|grad(ν̃)|²] - cw1·fw·(ν̃/d)²
```

Turbulent viscosity: `ν_t = ν̃ · fv1` where `fv1 = χ³/(χ³ + cv1³)`, `χ = ν̃/ν`

Production: `P = cb1 · S̃ · ν̃` where `S̃ = |S| + (ν̃/(κ²d²)) · fv2`

Destruction: `D = cw1 · fw · (ν̃/d)²`

Requires wall distance `d_wall` per cell (shared with SST).

## Wall Distance Computation

Both SST and SA need cell-to-wall distance. Computed once at problem setup:

```julia
function compute_wall_distance(
    mesh::UnstructuredFVMMesh{Dim, T},
    wall_patches::Vector{Symbol},
) -> Vector{T}
```

Simple approach: for each cell, find minimum distance to any wall boundary face center. This is O(ncells × nwall_faces) but only computed once. For large meshes, a future optimization can use a Poisson equation approach.

## Wall Treatment

### Wall functions (high-Re)

For k-ε and k-ω at wall boundaries, standard wall functions provide:
- Momentum: wall shear stress via log law → modifies momentum equation coefficients
- k: fixed value `k_w = u_τ² / √C_μ` or zero-gradient
- ε: fixed value `ε_w = C_μ^0.75 · k^1.5 / (κ·y)`
- ω: fixed value `ω_w = u_τ / (√β* · κ · y)`

Reuses existing `compute_friction_velocity`, `k_wall_value`, `epsilon_wall_value`.

### Low-Re (resolve viscous sublayer)

For SA model: `ν̃ = 0` at the wall (Dirichlet). No wall functions needed — the model includes damping terms (fv1, fv2, fw) that handle the near-wall region.

## Boundary Conditions for Turbulence

Turbulence BCs are provided as a separate dict keyed by field name:

```julia
bcs_turb = Dict(
    :k => Dict(:wall => ParabolicDirichlet(k_wall), :inlet => ParabolicDirichlet(k_inlet), ...),
    :epsilon => Dict(:wall => ParabolicDirichlet(eps_wall), :inlet => ParabolicDirichlet(eps_inlet), ...),
)
```

Convenience constructors generate these from inlet turbulence intensity and length scale:

```julia
function turbulence_inlet_bc(model::StandardKEpsilon, U_inlet, intensity, length_scale)
    k = 1.5 * (U_inlet * intensity)^2
    epsilon = 0.09^0.75 * k^1.5 / length_scale
    return Dict(:k => ParabolicDirichlet(k), :epsilon => ParabolicDirichlet(epsilon))
end
```

## Momentum Integration

`assemble_momentum!` gets an optional keyword argument for effective viscosity:

```julia
function assemble_momentum!(eq, state, prob, component;
    dt = nothing,
    scheme = CONV_UPWIND,
    nu_eff::Union{T, Vector{T}} = prob.nu,   # ← NEW: default to laminar
) where {Dim, T}
```

The Laplacian call changes from `assemble_laplacian!(eq, prob.nu, ...)` to `assemble_laplacian!(eq, nu_eff, ...)`. This is backward-compatible — existing code that doesn't pass `nu_eff` gets the same behavior.

## Solver Wrappers

### Turbulent SIMPLE

```julia
function solve_simple_turbulent(
    prob::IncompressibleProblem{Dim, T},
    turb_model::AbstractRANSModel;
    turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
    initial_turb_state = nothing,
    linear_solver = nothing,
    verbose = false,
) -> (SolveResult{Dim, T}, RANSTurbulenceState{T})
```

Loop:
1. Compute `nu_eff[c] = prob.nu + turb_state.nu_t[c]`
2. Assemble + solve momentum with `nu_eff`
3. Pressure solve + correction
4. `solve_turbulence!(turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs)`
5. `turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)`
6. Check convergence

### Turbulent transient

```julia
function solve_incompressible_turbulent(
    prob::IncompressibleProblem{Dim, T},
    turb_model::AbstractRANSModel,
    tspan::Tuple{T, T},
    dt::T;
    turb_bcs = ...,
    kwargs...,
) -> (SolveResult{Dim, T}, RANSTurbulenceState{T})
```

Same pattern but with PISO/PIMPLE time stepping.

## Export List

```julia
# Abstract types (AbstractTurbulenceModel already exported)
export AbstractRANSModel

# Model types
export KOmega, KOmegaSSTModel, SpalartAllmaras
# StandardKEpsilon and KappaOmegaSST already exported

# State
export RANSTurbulenceState

# Interface
export turbulent_viscosity!, solve_turbulence!
export n_turbulence_fields, turbulence_field_names

# Solvers
export solve_simple_turbulent, solve_incompressible_turbulent

# Utilities
export compute_wall_distance, compute_strain_rate
export turbulence_inlet_bc
```

## Validation

- **k-ε channel flow**: Compare velocity profile at Re_τ = 395 to log-law analytical profile. Pass: u+ within 10% of `1/κ·ln(y+) + B` in the log layer.
- **SA flat plate**: Compare skin friction coefficient to Blasius laminar solution at low Re, turbulent correlation at high Re.

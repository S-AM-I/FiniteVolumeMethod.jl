---
date: 2026-04-06
---

# Phase 7: Multiphase — Volume of Fluid (VOF)

**Status**: Design
**Depends on**: Phase 0 (collocated operators), Phase 1 (incompressible NS)

## Goal

Add two-phase immiscible flow simulation using the Volume of Fluid (VOF) method. Tracks the interface via a volume fraction field α, computes mixture properties, applies interface compression for sharpness, enforces boundedness, and includes the Continuum Surface Force (CSF) model for surface tension.

## Architecture

### Six Files in `src/multiphase/`

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `types.jl` | TwoPhaseProperties, VOFState, convenience constructors | ~80 |
| `alpha_transport.jl` | assemble_alpha!, interface compression flux computation | ~120 |
| `boundedness.jl` | clip_alpha! limiter ensuring 0 ≤ α ≤ 1 with conservation | ~60 |
| `mixture.jl` | compute_mixture_density!, compute_mixture_viscosity! | ~50 |
| `surface_tension.jl` | compute_curvature, compute_surface_tension_force (CSF) | ~100 |
| `solvers.jl` | solve_vof transient solver wrapper | ~180 |

Wired into Layer 2 after thermal includes.

## Type Design

### Two-Phase Properties

```julia
struct TwoPhaseProperties{T}
    rho1::T    # density of fluid 1 (α = 1), e.g., water = 1000
    rho2::T    # density of fluid 2 (α = 0), e.g., air = 1.225
    mu1::T     # dynamic viscosity of fluid 1, e.g., water = 1e-3
    mu2::T     # dynamic viscosity of fluid 2, e.g., air = 1.8e-5
    sigma::T   # surface tension coefficient [N/m] (0 = disabled)
end
```

Convenience constructor with keyword defaults for water/air:
```julia
TwoPhaseProperties(; rho1=1000.0, rho2=1.225, mu1=1e-3, mu2=1.8e-5, sigma=0.072)
```

### VOF State

```julia
mutable struct VOFState{T}
    alpha::CollocatedScalarField{T}  # volume fraction [0, 1]
    rho::Vector{T}                    # mixture density per cell
    mu::Vector{T}                     # mixture dynamic viscosity per cell
end
```

Constructor from mesh + initial alpha:
```julia
VOFState(mesh; alpha_init = 0.0)
```

Initializes `rho` and `mu` from `alpha_init` using default water/air properties (updated by `update_mixture_properties!` before first solve step).

## Alpha Transport Equation

### Standard Transport

```
∂α/∂t + div(phi · α) = 0
```

Assembled using Phase 0 operators:
```julia
function assemble_alpha!(
    eq::CollocatedEquation{T},
    alpha::CollocatedScalarField{T},
    phi::FaceFluxField{T},
    mesh::UnstructuredFVMMesh{Dim, T};
    dt::T,
)
```

Assembly:
1. `assemble_convection!(eq, phi, mesh, bcs_alpha)` — advection by flow
2. `assemble_ddt_euler!(eq, one(T), alpha.internal, mesh, dt)` — temporal

BCs for alpha: typically `ParabolicDirichlet(1.0)` at inlets carrying fluid 1, `ParabolicDirichlet(0.0)` at inlets carrying fluid 2, `ParabolicNeumann(0.0)` (zero-gradient) at walls and outlets.

### Interface Compression

To prevent interface smearing, add a compressive flux:
```
div(phi_c · α(1-α))
```

The compression flux at each internal face:
```
phi_c_f = C_alpha · |phi_f| · (n_interface · S_f) / |S_f|
```
where `C_alpha` is the compression coefficient (typically 1.0), `n_interface = ∇α/|∇α|` is the interface normal, and the term `(n_interface · S_f)/|S_f|` ensures compression acts only normal to the interface.

The compression is assembled as an explicit source to the alpha equation RHS, not as an implicit operator, since it's nonlinear in α.

```julia
function compute_compression_flux(
    alpha::CollocatedScalarField{T},
    phi::FaceFluxField{T},
    mesh::UnstructuredFVMMesh{Dim, T};
    C_alpha::T = one(T),
) -> Vector{T}  # per-face compression flux
```

The compression contribution to the alpha equation:
```julia
# For each face f:
alpha_f = interpolated alpha at face
compression = phi_c_f * alpha_f * (1 - alpha_f)
eq.b[P] -= compression  # subtract from owner
eq.b[N] += compression  # add to neighbour (if internal)
```

## Boundedness Limiter

After solving the alpha transport, α may violate [0, 1] bounds due to numerical diffusion or compression. A simple conservative limiter:

```julia
function clip_alpha!(alpha::CollocatedScalarField{T}, mesh) where {T}
```

1. Compute total α·V before clipping (conservation check)
2. Clip: `alpha[c] = clamp(alpha[c], 0, 1)` for all cells
3. Compute total α·V after clipping
4. Redistribute the difference proportionally to maintain global conservation

For a more sophisticated MULES-like approach (face-based flux limiting), a future enhancement can replace this. The clip+redistribute is sufficient for basic validation.

## Mixture Properties

```julia
function update_mixture_properties!(
    vof_state::VOFState{T},
    props::TwoPhaseProperties{T},
) where {T}
```

Per cell:
```
ρ[c] = α[c] · ρ₁ + (1 - α[c]) · ρ₂
μ[c] = α[c] · μ₁ + (1 - α[c]) · μ₂
```

These are called after each alpha transport step and before the momentum solve.

## Surface Tension (CSF)

The Continuum Surface Force model adds a body force at the interface:

```
F_st = σ · κ · ∇α
```

### Curvature Computation

```julia
function compute_curvature(
    alpha::CollocatedScalarField{T},
    mesh::UnstructuredFVMMesh{Dim, T},
) -> Vector{T}  # per-cell curvature κ
```

`κ = -div(∇α / |∇α|)`

Steps:
1. Compute `∇α` via Green-Gauss gradient
2. Normalize: `n_hat[c] = ∇α[c] / max(|∇α[c]|, ε)` (avoid division by zero away from interface)
3. Compute `div(n_hat)` using the explicit divergence operator (face-sum of interpolated n_hat · S_f)
4. `κ[c] = -div(n_hat)[c]`

### Surface Tension Force

```julia
function compute_surface_tension_force(
    alpha::CollocatedScalarField{T},
    props::TwoPhaseProperties{T},
    mesh::UnstructuredFVMMesh{Dim, T},
) -> Union{Nothing, Vector{SVector{Dim, T}}}
```

Returns `nothing` when `sigma == 0`.

Per cell: `F_st[c] = σ · κ[c] · ∇α[c]`

This is passed as `body_force` to `assemble_momentum!` (same keyword as buoyancy from Phase 3).

## Momentum with Variable Density

Single-phase incompressible flow assumes constant ρ. With VOF, density varies per cell. The key changes:

1. **Momentum equation**: Uses per-cell kinematic viscosity `ν[c] = μ[c] / ρ[c]` via the existing `nu_eff::Vector{T}` keyword.

2. **Pressure equation**: The pressure diffusivity becomes `D[c] = V[c] / (ρ[c] · A_P[c])` instead of `V[c] / A_P[c]`. This requires a modified pressure assembly that accounts for variable density.

3. **Body forces**: Both gravity `ρ·g` (not Boussinesq — full density variation) and surface tension `σ·κ·∇α` act as body forces.

### Gravity with Variable Density

For VOF, gravity is not Boussinesq — it uses the actual mixture density:
```
F_gravity[c] = ρ[c] · g
```

This is different from the Phase 3 Boussinesq source which uses `F = -ρ_ref·β·(T-T_ref)·g`.

## Solver Wrapper

```julia
function solve_vof(
    mesh::UnstructuredFVMMesh{Dim, T},
    props::TwoPhaseProperties{T},
    bcs_U::Dict{Symbol, AbstractBoundaryCondition},
    bcs_p::Dict{Symbol, AbstractBoundaryCondition},
    bcs_alpha::Dict{Symbol, AbstractBoundaryCondition},
    tspan::Tuple{T, T},
    dt::T;
    alpha_init::Union{T, Function} = zero(T),
    g::SVector{Dim, T} = zero(SVector{Dim, T}),
    C_alpha::T = one(T),
    algorithm::AbstractPVCoupling = PISO(),
    linear_solver = nothing,
    save_every::Int = 1,
    verbose::Bool = false,
) -> Tuple{SolveResult{Dim, T}, VOFState{T}}
```

The `alpha_init` can be a constant or a function `f(x) -> T` that maps cell center position to initial alpha (for defining complex initial interface shapes like dam break).

### Time-stepping Loop

```
for each time step:
    1. Solve alpha equation (explicit advection + compression)
    2. Apply boundedness limiter
    3. Update mixture properties (rho, mu)
    4. Compute body forces (gravity + surface tension)
    5. Compute nu_eff = mu / rho per cell
    6. PISO/PIMPLE step: momentum + pressure with nu_eff, body_force
       (pressure equation uses rho-weighted diffusivity)
    7. Save snapshot if needed
```

### Variable-Density Pressure Equation

The standard pressure equation from Phase 1 uses `D[c] = V[c] / A_P[c]`. For variable density VOF, this becomes:

```julia
function _vof_pressure_diffusivity(
    A_P::Vector{T}, rho::Vector{T}, mesh,
) -> Vector{T}
    D = Vector{T}(undef, nc)
    for c in 1:nc
        D[c] = mesh.cell_volumes[c] / (rho[c] * A_P[c])
    end
    return D
end
```

Rather than modifying `assemble_pressure!` (which would break single-phase), the VOF solver assembles the pressure equation directly using `assemble_laplacian!` with this density-weighted diffusivity.

## Export List

```julia
# Types
export TwoPhaseProperties, VOFState

# Alpha transport
export assemble_alpha!, compute_compression_flux

# Boundedness
export clip_alpha!

# Mixture
export update_mixture_properties!

# Surface tension
export compute_curvature, compute_surface_tension_force

# Solver
export solve_vof
```

## Validation

- **Dam break (2D)**: Column of water (α=1 in left half) collapses under gravity. Compare front position vs. time to Koshizuka & Oka (1996) or Martin & Moyce (1952) experimental data.
- **Static bubble**: Circular region of α=1 at rest. Verify pressure jump matches Young-Laplace (Δp = σ/R) and spurious currents are bounded.

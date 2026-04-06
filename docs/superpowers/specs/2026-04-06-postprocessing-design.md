# Phase 12: Post-Processing

**Date**: 2026-04-06
**Status**: Design
**Depends on**: Phase 0 (collocated operators), Phase 1 (incompressible NS)

## Goal

Provide derived field computations, wall surface metrics, integrated force coefficients, and field sampling for the collocated solver. These are essential for validating Phases 1-3 against benchmarks (skin friction, Nusselt number, drag/lift coefficients).

## Architecture

### File Layout

New files in `src/postprocessing/`:

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `field_operations.jl` | Vorticity, Q-criterion, enstrophy, Courant number | ~120 |
| `wall_quantities.jl` | Wall shear stress, y+, heat flux, Nusselt number | ~150 |
| `forces.jl` | Pressure/viscous forces, Cd/Cl/Cm coefficients | ~100 |
| `sampling.jl` | Line sampling, point interpolation | ~80 |

Wired into Layer 4 (`extensions_tooling_output.jl`) since these are output/diagnostic tools with no solver dependencies beyond reading fields.

### No New Dependencies

All functions operate on existing `CollocatedScalarField`, `CollocatedVectorField`, `UnstructuredFVMMesh`. Uses `gradient()` from Phase 0 for velocity/temperature gradients. Uses `extract_boundary_patches()` and mesh helpers from `src/collocated/types.jl`.

## Field Operations

### Velocity Gradient Tensor

All derived fields start from the velocity gradient tensor `∂u_i/∂x_j`. Computed once per call using Phase 0's `gradient()` on each velocity component:

```julia
function _compute_velocity_gradient(
    U::CollocatedVectorField{Dim, T},
    mesh::UnstructuredFVMMesh{Dim, T},
) -> Vector{Vector{SVector{Dim, T}}}
```

Returns `grad_U[d]` = gradient of component `d`, where `grad_U[d][c]` is an `SVector{Dim,T}` of partial derivatives at cell `c`. This is computed by creating a temporary `CollocatedScalarField` for each component and calling `gradient()`.

Note: This is the same approach used in `compute_strain_rate` (Phase 2a). To avoid duplication, the velocity gradient computation should be factored into a shared helper. However, `compute_strain_rate` lives in `src/turbulence/strain_rate.jl` which is in Layer 2. The post-processing code is in Layer 4 and can call any Layer 2 function. So we can reuse `compute_strain_rate` where needed, but the velocity gradient tensor itself is computed fresh here since `compute_strain_rate` only returns the magnitude, not the full tensor.

### Vorticity

```julia
function compute_vorticity(
    U::CollocatedVectorField{Dim, T},
    mesh::UnstructuredFVMMesh{Dim, T},
) -> Vector{T}  # 2D: scalar ω_z per cell
```

For 2D: `ω_z = ∂v/∂x - ∂u/∂y` (scalar per cell).

```julia
function compute_vorticity(
    U::CollocatedVectorField{3, T},
    mesh::UnstructuredFVMMesh{3, T},
) -> Vector{SVector{3, T}}  # 3D: vector ω per cell
```

For 3D: `ω = ∇ × U = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y)`.

### Q-Criterion

```julia
function compute_q_criterion(
    U::CollocatedVectorField{Dim, T},
    mesh::UnstructuredFVMMesh{Dim, T},
) -> Vector{T}
```

`Q = 0.5 * (|Ω|² - |S|²)` where `Ω_ij = 0.5*(∂u_i/∂x_j - ∂u_j/∂x_i)` is the rotation rate tensor and `S_ij` is the strain rate tensor. Positive Q identifies vortex cores.

For 2D: `Q = -0.5 * (∂u/∂x * ∂v/∂y - ∂u/∂y * ∂v/∂x)` (simplified from det of velocity gradient).

Actually the cleaner 2D form: `Q = 0.5*(Ω_12² - S_11² - S_22² - 2*S_12²)` but the general formula works for both dimensions.

### Enstrophy

```julia
function compute_enstrophy(
    U::CollocatedVectorField{Dim, T},
    mesh::UnstructuredFVMMesh{Dim, T},
) -> Vector{T}
```

`ε = |ω|²` — vorticity magnitude squared per cell.

### Courant Number

```julia
function compute_courant_number(
    phi::FaceFluxField{T},
    mesh::UnstructuredFVMMesh{Dim, T},
    dt::T,
) -> Vector{T}
```

Per cell: `Co = dt * sum_f |phi_f| / (2 * V_c)` where the sum is over faces of cell `c`.

## Wall Quantities

All wall quantities operate on a named boundary patch and use the owner cell's field values to approximate near-wall gradients.

### Wall Shear Stress

```julia
function compute_wall_shear_stress(
    U::CollocatedVectorField{Dim, T},
    nu::T,
    mesh::UnstructuredFVMMesh{Dim, T},
    patch::Symbol,
) -> Vector{SVector{Dim, T}}  # per boundary face of patch
```

For each wall face:
1. Get owner cell velocity `U_P` and wall distance `d = |x_f - x_P|`
2. Compute wall-normal direction `n = (x_f - x_P) / d`
3. Tangential velocity: `U_tan = U_P - (U_P · n) * n`
4. Wall shear stress: `τ_w = nu * U_tan / d` (assuming linear velocity profile near wall)

Returns a vector of `SVector{Dim, T}` with one entry per face in the patch.

### y+ (Wall Distance in Plus Units)

```julia
function compute_y_plus(
    U::CollocatedVectorField{Dim, T},
    nu::T,
    mesh::UnstructuredFVMMesh{Dim, T},
    patch::Symbol,
) -> Vector{T}  # per boundary face of patch
```

For each wall face:
1. Compute `τ_w` magnitude from wall shear stress
2. Friction velocity: `u_τ = sqrt(|τ_w| / ρ)` (ρ=1 for incompressible)
3. Wall distance: `y = |x_f - x_P|`
4. `y+ = y * u_τ / ν`

### Wall Heat Flux

```julia
function compute_wall_heat_flux(
    T_field::CollocatedScalarField{T},
    k::T,
    mesh::UnstructuredFVMMesh{Dim, T},
    patch::Symbol,
) -> Vector{T}  # per boundary face of patch
```

For each wall face: `q_w = -k * (T_wall - T_cell) / d` where `T_wall` is the boundary face value and `d` is the cell-to-face distance.

### Nusselt Number

```julia
function compute_nusselt_number(
    T_field::CollocatedScalarField{T},
    k::T,
    mesh::UnstructuredFVMMesh{Dim, T},
    patch::Symbol;
    T_ref::T,
    L_ref::T,
) -> Vector{T}  # per boundary face of patch
```

`Nu = q_w * L_ref / (k * (T_wall - T_ref))` where `q_w` is the wall heat flux.

## Forces

### Pressure and Viscous Forces

```julia
function compute_forces(
    p::CollocatedScalarField{T},
    U::CollocatedVectorField{Dim, T},
    nu::T,
    mesh::UnstructuredFVMMesh{Dim, T},
    patch::Symbol,
) -> NamedTuple{(:pressure, :viscous), Tuple{SVector{Dim, T}, SVector{Dim, T}}}
```

Pressure force: `F_p = -sum_f p_f * S_f` (face pressure × area vector).
Viscous force: `F_v = sum_f τ_w_f * A_f` (wall shear stress × face area).

Returns a named tuple `(pressure = SVector, viscous = SVector)`.

### Force Coefficients

```julia
function force_coefficients(
    pressure_force::SVector{Dim, T},
    viscous_force::SVector{Dim, T};
    rho_ref::T,
    U_ref::T,
    A_ref::T,
    drag_direction::SVector{Dim, T} = SVector{Dim}(one(T), zeros(T, Dim-1)...),
    lift_direction::SVector{Dim, T} = SVector{Dim}(zeros(T, Dim-1)..., one(T)),
) -> NamedTuple{(:Cd, :Cl, :Cd_pressure, :Cd_viscous)}
```

Dynamic pressure: `q = 0.5 * rho_ref * U_ref²`
- `Cd = (F_total · drag_dir) / (q * A_ref)`
- `Cl = (F_total · lift_dir) / (q * A_ref)`
- `Cd_pressure`, `Cd_viscous` for pressure/viscous contributions separately

## Sampling

### Line Sampling

```julia
function sample_line(
    field::CollocatedScalarField{T},
    mesh::UnstructuredFVMMesh{Dim, T},
    p1::SVector{Dim, T},
    p2::SVector{Dim, T},
    n_points::Int,
) -> NamedTuple{(:positions, :distances, :values)}
```

Generates `n_points` evenly spaced along the line from `p1` to `p2`. For each point, finds the nearest cell center and returns its field value (0th-order interpolation). Returns positions (Vector{SVector}), distances along line (Vector{T}), and values (Vector{T}).

Uses existing `find_cell_containing` for cell lookup, falling back to nearest-cell-center brute force search.

### Vector Field Line Sampling

```julia
function sample_line(
    field::CollocatedVectorField{Dim, T},
    mesh::UnstructuredFVMMesh{Dim, T},
    p1::SVector{Dim, T},
    p2::SVector{Dim, T},
    n_points::Int,
) -> NamedTuple{(:positions, :distances, :values)}
```

Same but returns `values::Vector{SVector{Dim, T}}`.

### Point Sampling

```julia
function sample_field_at_point(
    field::CollocatedScalarField{T},
    mesh::UnstructuredFVMMesh{Dim, T},
    point::SVector{Dim, T},
) -> T
```

Nearest-cell-center lookup. Returns the field value at the cell containing (or nearest to) the point.

## Export List

```julia
# Field operations
export compute_vorticity, compute_q_criterion, compute_enstrophy, compute_courant_number

# Wall quantities
export compute_wall_shear_stress, compute_y_plus, compute_wall_heat_flux, compute_nusselt_number

# Forces
export compute_forces, force_coefficients

# Sampling
export sample_line, sample_field_at_point
```

## Validation

- **Vorticity**: Rigid body rotation U=(−ωy, ωx) should give uniform ω_z = 2ω.
- **Wall shear stress**: Poiseuille flow in a channel — τ_w should match analytical 6μU_mean/h².
- **Courant number**: Uniform flow on uniform mesh — Co should be constant.
- **Line sampling**: Sample along centerline of a channel flow, verify velocity profile shape.

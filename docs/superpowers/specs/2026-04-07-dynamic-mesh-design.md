# Phase 10: Dynamic/Moving Mesh

**Date**: 2026-04-07
**Status**: Design
**Depends on**: Phase 0 (collocated operators), Phase 1 (incompressible NS)

## Goal

Add ALE (Arbitrary Lagrangian-Eulerian) mesh motion with Laplacian and solid-body motion solvers. Transport equations use corrected fluxes `phi - phi_mesh` to account for mesh velocity.

## Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/dynamic_mesh/types.jl` | AbstractMotionSolver, SolidBodyMotion, LaplacianMotion, MeshMotionState | ~80 |
| `src/dynamic_mesh/solid_body.jl` | Prescribed rotation/translation | ~60 |
| `src/dynamic_mesh/laplacian_motion.jl` | Diffusion equation for displacement | ~80 |
| `src/dynamic_mesh/mesh_update.jl` | Apply displacement, recompute geometry, compute phi_mesh | ~100 |
| `src/dynamic_mesh/ale.jl` | ALE flux correction + solve_ale wrapper | ~120 |

## Types

```julia
abstract type AbstractMotionSolver end

struct SolidBodyMotion{Dim, T, F} <: AbstractMotionSolver
    displacement_func::F  # t -> SVector{Dim, T}
end

struct LaplacianMotion{T} <: AbstractMotionSolver
    gamma::T  # diffusivity (uniform or distance-based factor)
end

mutable struct MeshMotionState{Dim, T}
    displacement::Vector{SVector{Dim, T}}  # per-cell displacement
    phi_mesh::Vector{T}                     # face sweep flux
    V_old::Vector{T}                        # cell volumes at previous time
end
```

## Motion Solvers

SolidBodyMotion: `displacement[c] = func(t)` for all cells. No PDE.

LaplacianMotion: solve `div(gamma * grad(d)) = 0` with Dirichlet on moving boundaries, zero on fixed. Uses Phase 0 Laplacian. One solve per spatial dimension.

## Mesh Update

After computing displacement, update mesh geometry:
1. `cell_centers[:, c] += displacement[c]`
2. Recompute face centers from owner/neighbour cell centers
3. Recompute face normals and areas (approximate: scale by volume ratio)
4. Recompute cell volumes
5. Compute face sweep flux: `phi_mesh[f] = (V_new - V_old) / dt` distributed to faces

## ALE Transport

Replace face flux in convection: `phi_ale = phi - phi_mesh`. The corrected flux is passed to all Phase 0 operators transparently.

## Export List

```julia
export AbstractMotionSolver, SolidBodyMotion, LaplacianMotion, MeshMotionState
export compute_displacement!, update_mesh!, compute_mesh_flux!
export ale_corrected_flux, solve_ale
```

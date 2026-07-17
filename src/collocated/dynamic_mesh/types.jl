# dynamic_mesh/types.jl — Core types for ALE mesh motion (Phase 10)
#
# Defines the motion solver hierarchy and the mutable state that tracks
# per-cell displacement, face sweep flux (phi_mesh), and old cell volumes.

# ── Abstract hierarchy ──────────────────────────────────────────────

@doc """
    AbstractMotionSolver

Supertype for mesh motion algorithms.  Concrete subtypes include
[`SolidBodyMotion`](@ref) (prescribed displacement) and
[`LaplacianMotion`](@ref) (diffusion-based displacement).
"""
abstract type AbstractMotionSolver end

# ── SolidBodyMotion ─────────────────────────────────────────────────

@doc """
    SolidBodyMotion{Dim, T, F} <: AbstractMotionSolver

Prescribed rigid-body mesh displacement where every cell receives the
same displacement vector at each time.

# Fields
- `displacement_func::F` — callable `t -> SVector{Dim, T}` returning the
  displacement at time `t`

# Example
```julia
motion = SolidBodyMotion{2, Float64}(t -> SVector(0.1 * t, 0.0))
```
"""
struct SolidBodyMotion{Dim, T, F} <: AbstractMotionSolver
    displacement_func::F
end

@doc """
    SolidBodyMotion{Dim, T}(displacement_func) where {Dim, T}

Construct a [`SolidBodyMotion`](@ref) from a callable `t -> SVector{Dim, T}`.
"""
function SolidBodyMotion{Dim, T}(displacement_func::F) where {Dim, T, F}
    return SolidBodyMotion{Dim, T, F}(displacement_func)
end

# ── LaplacianMotion ─────────────────────────────────────────────────

@doc """
    LaplacianMotion{T} <: AbstractMotionSolver

Diffusion-based mesh motion solver.  Computes displacement by solving
`div(gamma * grad(d)) = 0` per spatial dimension with Dirichlet
boundary conditions on moving and fixed boundaries.

Uses the Phase 0 Laplacian assembly (`assemble_laplacian!`) and
linear solver infrastructure (`_dispatch_solve`).

# Fields
- `gamma::T` — uniform diffusivity coefficient (default `1.0`)
"""
struct LaplacianMotion{T} <: AbstractMotionSolver
    gamma::T
end

@doc """
    LaplacianMotion(; gamma = 1.0)

Construct a [`LaplacianMotion`](@ref) with the given diffusivity.
"""
function LaplacianMotion(; gamma::T = 1.0) where {T}
    return LaplacianMotion{T}(gamma)
end

# ── MeshMotionState ─────────────────────────────────────────────────

@doc """
    MeshMotionState{Dim, T}

Mutable state for ALE mesh motion.  Tracks the per-cell displacement
field, face sweep flux, and cell volumes from the previous time step.

# Fields
- `displacement::Vector{SVector{Dim, T}}` — displacement at each cell center
- `phi_mesh::Vector{T}` — face sweep flux (mesh velocity contribution)
- `V_old::Vector{T}` — cell volumes at the previous time level
"""
mutable struct MeshMotionState{Dim, T}
    displacement::Vector{SVector{Dim, T}}
    phi_mesh::Vector{T}
    V_old::Vector{T}
end

@doc """
    MeshMotionState(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}

Construct a zero-initialized [`MeshMotionState`](@ref) for the given mesh.
Displacement is zero, phi_mesh is zero, and `V_old` is a copy of the
current cell volumes.
"""
function MeshMotionState(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    displacement = fill(zero(SVector{Dim, T}), nc)
    phi_mesh = zeros(T, nf)
    V_old = copy(mesh.cell_volumes)
    return MeshMotionState{Dim, T}(displacement, phi_mesh, V_old)
end

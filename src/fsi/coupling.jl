# fsi/coupling.jl — Partitioned fluid-structure interaction (Stage 7b)
#
# Dirichlet-Neumann iteration:
#   1. Fluid sees solid displacement as mesh-motion BC (Dirichlet on
#      fluid-solid interface).
#   2. Fluid solves one PISO/PIMPLE step with moving-mesh ALE.
#   3. Fluid tractions on the interface are handed to the solid
#      as a prescribed Neumann BC.
#   4. Solid solves one equilibrium step → new displacement.
#   5. Aitken relaxation of the new displacement vs. previous
#      iterate accelerates convergence.
#   6. Repeat until the interface residual drops below tolerance.
#
# References: Mok et al. (2001); Küttler & Wall (2008), Comput. Mech.
# 43, 61-72. Clean-room implementation.

using LinearAlgebra: dot, norm
using StaticArrays: SVector

"""
    AitkenRelaxation{T}

State for Aitken-Δ² under-relaxation of the FSI interface displacement.
Given successive residuals `r^{k-1}` and `r^k`:

    ω^k = ω^{k-1} · (r^{k-1} · (r^{k-1} - r^k)) / ‖r^{k-1} - r^k‖²

Clamped to [ω_min, ω_max] for robustness.
"""
mutable struct AitkenRelaxation{T}
    omega::T
    omega_min::T
    omega_max::T
    prev_residual::Union{Nothing, Vector{T}}
end
AitkenRelaxation(; omega0::Real = 0.5, omega_min::Real = 0.01, omega_max::Real = 1.0) =
    AitkenRelaxation{typeof(float(omega0))}(
    float(omega0), float(omega_min), float(omega_max), nothing,
)

"""
    update_aitken!(relax::AitkenRelaxation, r_current)

Advance the Aitken relaxation state using `r_current = predicted - actual`.
Returns the updated `ω` to use for the current iterate.
"""
function update_aitken!(relax::AitkenRelaxation{T}, r_current::AbstractVector{T}) where {T}
    if relax.prev_residual === nothing
        relax.prev_residual = copy(r_current)
        return relax.omega
    end
    r_prev = relax.prev_residual
    dr = similar(r_current)
    @inbounds for i in eachindex(r_current)
        dr[i] = r_prev[i] - r_current[i]
    end
    denom = dot(dr, dr)
    if denom > eps(T)
        # Aitken: ω^k = −ω^{k-1} · (r_prev · dr) / |dr|²
        # Sign convention per Küttler-Wall: positive when residuals shrink.
        relax.omega = -relax.omega * dot(r_prev, dr) / denom
        relax.omega = clamp(relax.omega, relax.omega_min, relax.omega_max)
    end
    relax.prev_residual = copy(r_current)
    return relax.omega
end

"""
    FSIInterface{Dim, T}

Bookkeeping for a single fluid-solid interface: the pair of face-index
lists on the two sides plus the exchange arrays for the coupled
variables.

# Fields
- `fluid_face_indices` — interface face indices on the fluid mesh.
- `solid_face_indices` — corresponding interface face indices on the solid mesh.
- `displacement::Vector{SVector{Dim, T}}` — current interface displacement (sent to fluid as mesh motion).
- `traction::Vector{SVector{Dim, T}}` — current interface traction (sent to solid as BC).
"""
struct FSIInterface{Dim, T}
    fluid_face_indices::Vector{Int}
    solid_face_indices::Vector{Int}
    displacement::Vector{SVector{Dim, T}}
    traction::Vector{SVector{Dim, T}}
end

function FSIInterface{Dim, T}(
        fluid_faces::Vector{Int}, solid_faces::Vector{Int},
    ) where {Dim, T}
    length(fluid_faces) == length(solid_faces) ||
        error("FSIInterface: fluid/solid face lists must be equal length")
    n = length(fluid_faces)
    return FSIInterface{Dim, T}(
        fluid_faces, solid_faces,
        fill(zero(SVector{Dim, T}), n),
        fill(zero(SVector{Dim, T}), n),
    )
end

"""
    interface_residual_norm(d_new, d_old) -> Float64

L2 norm of the interface-displacement update across iterations — the
FSI convergence criterion. When this falls below a user tolerance, the
partitioned coupling has converged for the current time step.
"""
function interface_residual_norm(
        d_new::AbstractVector{SVector{Dim, T}},
        d_old::AbstractVector{SVector{Dim, T}},
    ) where {Dim, T}
    length(d_new) == length(d_old) || error("length mismatch")
    acc = zero(T)
    @inbounds for i in eachindex(d_new)
        diff = d_new[i] - d_old[i]
        acc += dot(diff, diff)
    end
    return sqrt(acc)
end

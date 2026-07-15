# collocated/interpolation.jl — Face interpolation schemes for collocated FVM
#
# Provides cell→face interpolation methods required by the divergence and
# gradient operators.  The Rhie-Chow momentum interpolation is the key
# ingredient for pressure-velocity coupling on collocated grids (prevents
# checkerboard pressure oscillations).

# ── Linear (central) interpolation ───────────────────────────────────

"""
    interpolate_linear(phi::CollocatedScalarField, mesh, f::Int) -> T

Distance-weighted linear interpolation of a scalar field to internal face `f`:
`φ_f = w * φ_P + (1-w) * φ_N` where `w = face_weight(mesh, f)`.
"""
function interpolate_linear(
        phi::CollocatedScalarField{T}, mesh::UnstructuredFVMMesh, f::Int,
    ) where {T}
    w = face_weight(mesh, f)
    P = owner(mesh, f)
    N = neighbour(mesh, f)
    return w * phi.internal[P] + (one(T) - w) * phi.internal[N]
end

"""
    interpolate_linear(U::CollocatedVectorField, mesh, f::Int) -> SVector

Distance-weighted linear interpolation of a vector field to face `f`.
"""
function interpolate_linear(
        U::CollocatedVectorField{Dim, T}, mesh::UnstructuredFVMMesh, f::Int,
    ) where {Dim, T}
    w = face_weight(mesh, f)
    P = owner(mesh, f)
    N = neighbour(mesh, f)
    return w * U.internal[P] + (one(T) - w) * U.internal[N]
end

# ── Upwind interpolation ────────────────────────────────────────────

"""
    interpolate_upwind(phi::CollocatedScalarField, mesh, f::Int, flux_sign) -> T

First-order upwind: returns `φ_P` if the face flux is positive (from owner
to neighbour), `φ_N` otherwise.  `flux_sign` is the sign of the face mass
flux (positive = owner→neighbour direction).
"""
function interpolate_upwind(
        phi::CollocatedScalarField{T}, mesh::UnstructuredFVMMesh, f::Int,
        flux_sign::T,
    ) where {T}
    P = owner(mesh, f)
    N = neighbour(mesh, f)
    return flux_sign >= zero(T) ? phi.internal[P] : phi.internal[N]
end

# ── Blended interpolation ───────────────────────────────────────────

"""
    interpolate_blended(phi, mesh, f, flux_sign, blend) -> T

Blend between upwind and linear: `φ_f = β * φ_linear + (1-β) * φ_upwind`.
`blend ∈ [0, 1]`: 0 = pure upwind, 1 = pure central.
"""
function interpolate_blended(
        phi::CollocatedScalarField{T}, mesh::UnstructuredFVMMesh, f::Int,
        flux_sign::T, blend::T,
    ) where {T}
    phi_lin = interpolate_linear(phi, mesh, f)
    phi_up = interpolate_upwind(phi, mesh, f, flux_sign)
    return blend * phi_lin + (one(T) - blend) * phi_up
end

# ── Boundary face value lookup ───────────────────────────────────────

"""
    boundary_value(phi::CollocatedScalarField, mesh, f::Int) -> T

Look up the boundary face value for face `f`.  Performs a linear search
in `boundary_face_indices`; for hot loops, pre-build a face→boundary
index map via `build_boundary_map`.
"""
function boundary_value(
        phi::CollocatedScalarField{T}, mesh::UnstructuredFVMMesh, f::Int,
    ) where {T}
    idx = findfirst(==(f), phi.boundary_face_indices)
    idx === nothing && error("Face $f is not in boundary_face_indices")
    return phi.boundary[idx]
end

"""
    build_boundary_map(field[, mesh]) -> Vector{Int}

Build a mapping from mesh face index → index into `field.boundary`.

Returns a `Vector{Int}` of length equal to the total number of mesh faces
(`size(mesh.face_cells, 2)` if `mesh` is given, otherwise the largest
boundary face index seen). Entry `f` is the 1-based index into
`field.boundary` if face `f` is one of the field's boundary faces, and `0`
otherwise. `bmap[f]` has the same call syntax as a `Dict{Int, Int}` for
existing callers, but is O(1) with no hashing and allocates exactly once.

Hot-loop rule: allocate this once per solver iteration (or cache it on a
state object) and reuse for every face-value / Rhie-Chow / gradient call.
"""
function build_boundary_map(field::Union{CollocatedScalarField, CollocatedVectorField})
    # Size-to-max-boundary-index path — safe because every caller guards
    # `bmap[f]` behind an `is_internal_face(mesh, f)` check, so `f` is
    # always a boundary face when this vector is indexed.
    max_idx = isempty(field.boundary_face_indices) ? 0 :
        maximum(field.boundary_face_indices)
    bmap = zeros(Int, max_idx)
    @inbounds for (i, f) in pairs(field.boundary_face_indices)
        bmap[f] = i
    end
    return bmap
end

function build_boundary_map(
        field::Union{CollocatedScalarField, CollocatedVectorField},
        mesh::UnstructuredFVMMesh,
    )
    nf = size(mesh.face_cells, 2)
    bmap = zeros(Int, nf)
    @inbounds for (i, f) in pairs(field.boundary_face_indices)
        bmap[f] = i
    end
    return bmap
end

"""
    face_value(phi::CollocatedScalarField, mesh, f, bmap) -> T

Return the face value of `phi` at face `f`: linear interpolation for
internal faces, boundary lookup (via `bmap`) for boundary faces.
"""
function face_value(
        phi::CollocatedScalarField{T}, mesh::UnstructuredFVMMesh, f::Int,
        bmap::AbstractVector{Int},
    ) where {T}
    if is_internal_face(mesh, f)
        return interpolate_linear(phi, mesh, f)
    else
        return phi.boundary[bmap[f]]
    end
end

# ── Face flux from velocity ──────────────────────────────────────────

"""
    compute_face_flux!(phi::FaceFluxField, U::CollocatedVectorField, mesh)

Compute volumetric face flux `phi_f = U_f . S_f` using linear
interpolation of the velocity field.  For boundary faces, uses the
boundary value of `U`.
"""
function compute_face_flux!(
        phi::FaceFluxField{T},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    ubmap = build_boundary_map(U, mesh)

    @inbounds for f in 1:nf
        S_f = face_normal_area(mesh, f)
        if is_internal_face(mesh, f)
            U_f = interpolate_linear(U, mesh, f)
        else
            bi = ubmap[f]
            U_f = U.boundary[bi]
        end
        phi.values[f] = dot(U_f, S_f)
    end
    return nothing
end

# ── Rhie-Chow momentum interpolation ────────────────────────────────

"""
    rhie_chow_correction!(
        phi::FaceFluxField, U::CollocatedVectorField, p::CollocatedScalarField,
        A_P_diag::Vector, mesh; under_relax = 1.0,
    )

Apply Rhie-Chow pressure-velocity coupling correction to face fluxes.

The corrected flux is:
```
ϕ_f = U_f · S_f - D_f · (∇p)_f · S_f + D_f · (overline{∇p})_f · S_f
```
where `D_f = V_f / A_P_f` is the momentum equation diagonal inverse
interpolated to the face, and the second term replaces the interpolated
pressure gradient with a compact face-normal gradient.

This suppresses checkerboard pressure oscillations inherent to
collocated grids.

# Arguments
- `phi` — face flux field (modified in-place)
- `U` — cell-centered velocity
- `p` — cell-centered pressure
- `A_P_diag` — diagonal coefficients from the momentum equation (`a_P` per cell)
- `mesh` — `UnstructuredFVMMesh`
- `under_relax` — under-relaxation factor for the correction (default 1.0)
"""
function rhie_chow_correction!(
        phi::FaceFluxField{T},
        U::CollocatedVectorField{Dim, T},
        p::CollocatedScalarField{T},
        A_P_diag::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        under_relax::T = one(T),
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    ubmap = build_boundary_map(U, mesh)

    # Compute cell-center pressure gradient for Rhie-Chow
    grad_p = gradient(p, mesh)

    @inbounds for f in 1:nf
        S_f = face_normal_area(mesh, f)

        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            # Interpolate velocity to face
            U_f = w * U.internal[P] + (one(T) - w) * U.internal[N]

            # Face coefficient D_f = (V/a_P)_f via HARMONIC mean — must be
            # identical to `_face_diffusivity` used by the pressure
            # Laplacian assembly, otherwise the corrected fluxes cannot be
            # divergence-free when A_P varies in space.
            D_P = mesh.cell_volumes[P] / A_P_diag[P]
            D_N = mesh.cell_volumes[N] / A_P_diag[N]
            denom = w * D_N + (one(T) - w) * D_P
            D_f = denom > zero(T) ? D_P * D_N / denom : zero(T)

            # Compact face-normal pressure gradient: (p_N - p_P) / |d| * |S_f|
            d_vec, d_mag = owner_neighbour_distance(mesh, f)
            grad_p_compact = (p.internal[N] - p.internal[P]) / d_mag * mesh.face_areas[f]

            # Interpolated cell-center pressure gradient at face
            grad_p_interp = w * grad_p[P] + (one(T) - w) * grad_p[N]
            grad_p_interp_dot_S = dot(grad_p_interp, S_f)

            # Rhie-Chow: phi = U_f·S_f - D_f * (compact - interpolated)
            phi.values[f] = dot(U_f, S_f) -
                under_relax * D_f * (grad_p_compact - grad_p_interp_dot_S)
        else
            # Boundary: use boundary velocity directly
            bi = ubmap[f]
            U_f = U.boundary[bi]
            phi.values[f] = dot(U_f, S_f)
        end
    end
    return nothing
end

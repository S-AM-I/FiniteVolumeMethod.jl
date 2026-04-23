# collocated/gradient.jl — Cell-centered gradient reconstruction on polyhedral meshes
#
# Implements Green-Gauss gradient reconstruction with optional iterative
# non-orthogonal correction, suitable for arbitrary unstructured meshes.
# Follows the OpenFOAM `Gauss linear` gradient scheme semantics.

# ── Green-Gauss gradient (single cell) ──────────────────────────────

"""
    green_gauss_gradient(
        phi::CollocatedScalarField, mesh::UnstructuredFVMMesh{Dim, T},
        cell::Int, bmap::AbstractVector{Int},
    ) -> SVector{Dim, T}

Compute the Green-Gauss gradient of scalar `phi` at cell `cell`:

```math
(\\nabla \\phi)_P = \\frac{1}{V_P} \\sum_f \\phi_f \\, \\mathbf{S}_f
```

where `φ_f` is the linearly interpolated face value and `S_f` is the
outward face area vector.
"""
function green_gauss_gradient(
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        cell::Int,
        bmap::AbstractVector{Int},
    ) where {Dim, T}
    mesh.cell_faces === nothing && error("cell_faces required for gradient reconstruction")
    grad = zero(SVector{Dim, T})
    V_P = mesh.cell_volumes[cell]

    for f in mesh.cell_faces[cell]
        phi_f = face_value(phi, mesh, f, bmap)
        S_f = face_normal_area(mesh, f)

        # face_normals point from owner → neighbour; flip sign if this cell
        # is the neighbour
        if owner(mesh, f) == cell
            grad = grad + phi_f * S_f
        else
            grad = grad - phi_f * S_f
        end
    end

    return grad / V_P
end

# ── Gradient for all cells ──────────────────────────────────────────

"""
    gradient!(
        grad_phi::Vector{SVector{Dim, T}},
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        n_corrections::Int = 0,
    )

Compute the cell-centered gradient of `phi` at all cells, storing
results in `grad_phi`.

With `n_corrections = 0` (default), performs a single-pass Green-Gauss
gradient.  With `n_corrections > 0`, applies iterative correction for
non-orthogonal meshes: each correction pass uses the current gradient
to compute a more accurate face value via:

```
φ_f = φ_f^{linear} + (∇φ)_f · (x_f - x_{f,projected})
```

# Arguments
- `grad_phi` — output vector, length `ncells`, overwritten in-place
- `phi` — input scalar field
- `mesh` — `UnstructuredFVMMesh` (must have `cell_faces`)
- `n_corrections` — number of non-orthogonal correction sweeps (0 = none)
"""
function gradient!(
        grad_phi::Vector{SVector{Dim, T}},
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        n_corrections::Int = 0,
        scratch::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        bmap::Union{Nothing, Vector{Int}} = nothing,
    ) where {Dim, T}
    mesh.cell_faces === nothing && error("cell_faces required for gradient reconstruction")
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    bmap_eff = bmap === nothing ? build_boundary_map(phi, mesh) : bmap

    # --- Initial Green-Gauss pass ---
    fill!(grad_phi, zero(SVector{Dim, T}))
    @inbounds for f in 1:nf
        phi_f = face_value(phi, mesh, f, bmap_eff)
        S_f = face_normal_area(mesh, f)

        P = owner(mesh, f)
        grad_phi[P] = grad_phi[P] + phi_f * S_f

        N = neighbour(mesh, f)
        if N != 0
            grad_phi[N] = grad_phi[N] - phi_f * S_f
        end
    end

    @inbounds for c in 1:nc
        grad_phi[c] = grad_phi[c] / mesh.cell_volumes[c]
    end

    # --- Iterative non-orthogonal correction ---
    if n_corrections > 0
        scratch_buf = scratch === nothing ? Vector{SVector{Dim, T}}(undef, nc) : scratch
        length(scratch_buf) == nc || error("scratch buffer length $(length(scratch_buf)) ≠ ncells $nc")
        for _ in 1:n_corrections
            _corrected_gradient_pass!(grad_phi, scratch_buf, phi, mesh, bmap_eff)
        end
    end

    return nothing
end

"""
Internal helper: one correction pass of the iterative Green-Gauss scheme.
`scratch` is a caller-provided buffer of length `ncells`; it is overwritten
then copied back into `grad_phi`. No per-call allocation.
"""
function _corrected_gradient_pass!(
        grad_phi::Vector{SVector{Dim, T}},
        scratch::Vector{SVector{Dim, T}},
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bmap::Vector{Int},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    fill!(scratch, zero(SVector{Dim, T}))

    @inbounds for f in 1:nf
        P = owner(mesh, f)
        S_f = face_normal_area(mesh, f)

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            # Corrected face value using current gradient
            phi_f_linear = w * phi.internal[P] + (one(T) - w) * phi.internal[N]
            grad_f = w * grad_phi[P] + (one(T) - w) * grad_phi[N]

            # Non-orthogonal correction: project gradient onto face offset
            x_f = face_center(mesh, f)
            x_mid = w * cell_center(mesh, P) + (one(T) - w) * cell_center(mesh, N)
            correction = dot(grad_f, x_f - x_mid)

            phi_f = phi_f_linear + correction
        else
            phi_f = phi.boundary[bmap[f]]
        end

        scratch[P] = scratch[P] + phi_f * S_f

        N = neighbour(mesh, f)
        if N != 0
            scratch[N] = scratch[N] - phi_f * S_f
        end
    end

    @inbounds for c in 1:nc
        grad_phi[c] = scratch[c] / mesh.cell_volumes[c]
    end

    return nothing
end

# ── Convenience: allocating version ──────────────────────────────────

"""
    gradient(phi, mesh; n_corrections = 0) -> Vector{SVector{Dim, T}}

Allocating version of `gradient!`.
"""
function gradient(
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        n_corrections::Int = 0,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    grad_phi = Vector{SVector{Dim, T}}(undef, nc)
    gradient!(grad_phi, phi, mesh; n_corrections)
    return grad_phi
end

# ── Weighted least-squares gradient ─────────────────────────────────

"""
    least_squares_gradient!(
        grad_phi::Vector{SVector{Dim, T}},
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        bmap::Union{Nothing, Vector{Int}} = nothing,
    )

Compute the weighted least-squares (LSQ) gradient of `phi` at every cell
and store the result in `grad_phi`. For each cell `P` with face-neighbours
`N ∈ 𝓝(P)` (plus boundary faces, which contribute via face-centre values),
fit a linear polynomial to the offsets

```math
\\min_{\\mathbf{g}_P} \\; \\sum_{N} w_{PN}\\,
    \\left( \\phi_N - \\phi_P - \\mathbf{g}_P \\cdot \\mathbf{d}_{PN} \\right)^2
```

with inverse-distance-squared weights `w_{PN} = 1 / |d_{PN}|^2`. The normal
equations are `M · g = r`, where

```
M = Σ_N w_{PN} · d_{PN} ⊗ d_{PN}
r = Σ_N w_{PN} · (φ_N − φ_P) · d_{PN}
```

and `d_{PN}` is either `x_N − x_P` (internal neighbour) or `x_f − x_P`
(boundary face, which uses the boundary-face value of `phi`). LSQ gradients
are exact for linear fields on arbitrary polyhedral meshes and are
preferred over Green-Gauss on skewed or unstructured meshes.

# Arguments
- `grad_phi` — output, length `ncells`, overwritten in place.
- `phi` — scalar field (internal + boundary face values).
- `mesh` — `UnstructuredFVMMesh` (must have `cell_faces`).
- `bmap` — optional boundary-face → `phi.boundary` index map
  (pass in via `build_boundary_map` to avoid reconstructing it).
"""
function least_squares_gradient!(
        grad_phi::Vector{SVector{Dim, T}},
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        bmap::Union{Nothing, Vector{Int}} = nothing,
    ) where {Dim, T}
    mesh.cell_faces === nothing && error("cell_faces required for LSQ gradient")
    nc = length(mesh.cell_volumes)
    length(grad_phi) == nc ||
        error("grad_phi length $(length(grad_phi)) ≠ ncells $nc")
    bmap_eff = bmap === nothing ? build_boundary_map(phi, mesh) : bmap

    @inbounds for P in 1:nc
        phi_P = phi.internal[P]
        x_P = cell_center(mesh, P)

        # Normal-equation accumulators: symmetric matrix M (stored as
        # Dim×Dim) and RHS vector r.
        M = zero(StaticArrays.SMatrix{Dim, Dim, T})
        r = zero(SVector{Dim, T})

        for f in mesh.cell_faces[P]
            if is_internal_face(mesh, f)
                N = owner(mesh, f) == P ? neighbour(mesh, f) : owner(mesh, f)
                d = cell_center(mesh, N) - x_P
                dphi = phi.internal[N] - phi_P
            else
                # Boundary face: use face-centre and boundary value.
                d = face_center(mesh, f) - x_P
                dphi = phi.boundary[bmap_eff[f]] - phi_P
            end
            d2 = dot(d, d)
            d2 > zero(T) || continue
            w = one(T) / d2
            M = M + w * (d * d')
            r = r + (w * dphi) * d
        end

        grad_phi[P] = _lsq_solve(M, r)
    end

    return nothing
end

"""Solve the symmetric LSQ normal-equation `M g = r` in closed form for
`Dim ∈ {2, 3}`. Returns `zero(SVector)` if `M` is numerically singular
(e.g. a purely collinear one-dimensional stencil)."""
@inline function _lsq_solve(
        M::StaticArrays.SMatrix{2, 2, T}, r::SVector{2, T},
    ) where {T}
    detM = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
    scale = max(abs(M[1, 1]) + abs(M[2, 2]), one(T))
    if abs(detM) < eps(T) * scale * scale
        return zero(SVector{2, T})
    end
    inv_det = one(T) / detM
    gx = (M[2, 2] * r[1] - M[1, 2] * r[2]) * inv_det
    gy = (-M[2, 1] * r[1] + M[1, 1] * r[2]) * inv_det
    return SVector{2, T}(gx, gy)
end

@inline function _lsq_solve(
        M::StaticArrays.SMatrix{3, 3, T}, r::SVector{3, T},
    ) where {T}
    # Cofactor expansion along the first row.
    c11 = M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]
    c12 = M[2, 3] * M[3, 1] - M[2, 1] * M[3, 3]
    c13 = M[2, 1] * M[3, 2] - M[2, 2] * M[3, 1]
    detM = M[1, 1] * c11 + M[1, 2] * c12 + M[1, 3] * c13
    scale = max(abs(M[1, 1]) + abs(M[2, 2]) + abs(M[3, 3]), one(T))
    if abs(detM) < eps(T) * scale * scale * scale
        return zero(SVector{3, T})
    end
    c21 = M[1, 3] * M[3, 2] - M[1, 2] * M[3, 3]
    c22 = M[1, 1] * M[3, 3] - M[1, 3] * M[3, 1]
    c23 = M[1, 2] * M[3, 1] - M[1, 1] * M[3, 2]
    c31 = M[1, 2] * M[2, 3] - M[1, 3] * M[2, 2]
    c32 = M[1, 3] * M[2, 1] - M[1, 1] * M[2, 3]
    c33 = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
    inv_det = one(T) / detM
    gx = (c11 * r[1] + c21 * r[2] + c31 * r[3]) * inv_det
    gy = (c12 * r[1] + c22 * r[2] + c32 * r[3]) * inv_det
    gz = (c13 * r[1] + c23 * r[2] + c33 * r[3]) * inv_det
    return SVector{3, T}(gx, gy, gz)
end

"""
    least_squares_gradient(phi, mesh) -> Vector{SVector{Dim, T}}

Allocating wrapper around `least_squares_gradient!`.
"""
function least_squares_gradient(
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    grad_phi = Vector{SVector{Dim, T}}(undef, nc)
    least_squares_gradient!(grad_phi, phi, mesh)
    return grad_phi
end

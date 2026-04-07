# collocated/gradient.jl — Cell-centered gradient reconstruction on polyhedral meshes
#
# Implements Green-Gauss gradient reconstruction with optional iterative
# non-orthogonal correction, suitable for arbitrary unstructured meshes.
# Follows the OpenFOAM `Gauss linear` gradient scheme semantics.

# ── Green-Gauss gradient (single cell) ──────────────────────────────

"""
    green_gauss_gradient(
        phi::CollocatedScalarField, mesh::UnstructuredFVMMesh{Dim, T},
        cell::Int, bmap::Dict{Int, Int},
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
        bmap::Dict{Int, Int},
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
    ) where {Dim, T}
    mesh.cell_faces === nothing && error("cell_faces required for gradient reconstruction")
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    bmap = build_boundary_map(phi)

    # --- Initial Green-Gauss pass ---
    fill!(grad_phi, zero(SVector{Dim, T}))
    for f in 1:nf
        phi_f = face_value(phi, mesh, f, bmap)
        S_f = face_normal_area(mesh, f)

        P = owner(mesh, f)
        grad_phi[P] = grad_phi[P] + phi_f * S_f

        N = neighbour(mesh, f)
        if N != 0
            grad_phi[N] = grad_phi[N] - phi_f * S_f
        end
    end

    for c in 1:nc
        grad_phi[c] = grad_phi[c] / mesh.cell_volumes[c]
    end

    # --- Iterative non-orthogonal correction ---
    for _ in 1:n_corrections
        _corrected_gradient_pass!(grad_phi, phi, mesh, bmap)
    end

    return nothing
end

"""
Internal helper: one correction pass of the iterative Green-Gauss scheme.
Returns nothing; mutates `grad_phi` in-place.
"""
function _corrected_gradient_pass!(
        grad_phi::Vector{SVector{Dim, T}},
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bmap::Dict{Int, Int},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Temporary accumulator
    grad_new = fill(zero(SVector{Dim, T}), nc)

    for f in 1:nf
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

        grad_new[P] = grad_new[P] + phi_f * S_f

        N = neighbour(mesh, f)
        if N != 0
            grad_new[N] = grad_new[N] - phi_f * S_f
        end
    end

    for c in 1:nc
        grad_phi[c] = grad_new[c] / mesh.cell_volumes[c]
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

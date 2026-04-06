# multiphase/surface_tension.jl — Continuum Surface Force (CSF) model
#
# Computes interface curvature from the volume fraction gradient and
# produces a body force F_st = σ · κ · ∇α for the momentum equation.

using LinearAlgebra: norm, dot

"""
    compute_curvature(alpha, mesh) -> Vector{T}

Compute the interface curvature `κ = -div(∇α / |∇α|)` per cell.

Steps:
1. Compute `∇α` via Green-Gauss gradient
2. Normalize to get interface normal `n̂ = ∇α / |∇α|`
3. Compute `div(n̂)` via face summation
4. `κ = -div(n̂)`
"""
function compute_curvature(
        alpha::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Step 1: gradient of alpha
    grad_alpha = gradient(alpha, mesh)

    # Step 2: normalize to get interface normal per cell
    n_hat = Vector{SVector{Dim, T}}(undef, nc)
    for c in 1:nc
        g_mag = norm(grad_alpha[c])
        if g_mag > T(1.0e-12)
            n_hat[c] = grad_alpha[c] / g_mag
        else
            n_hat[c] = zero(SVector{Dim, T})
        end
    end

    # Step 3: div(n_hat) via face summation
    div_n = zeros(T, nc)
    for f in 1:nf
        P = owner(mesh, f)
        S_f = face_normal_area(mesh, f)

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            # Interpolate n_hat to face
            n_f = w * n_hat[P] + (one(T) - w) * n_hat[N]
            flux = dot(n_f, S_f)

            div_n[P] += flux
            div_n[N] -= flux
        else
            # Boundary: use owner value
            flux = dot(n_hat[P], S_f)
            div_n[P] += flux
        end
    end

    # Normalize by cell volume
    kappa = Vector{T}(undef, nc)
    for c in 1:nc
        div_n[c] /= mesh.cell_volumes[c]
        kappa[c] = -div_n[c]
    end

    return kappa
end

"""
    compute_surface_tension_force(alpha, props, mesh) -> Union{Nothing, Vector{SVector{Dim, T}}}

Compute the CSF surface tension body force: `F_st = σ · κ · ∇α`.

Returns `nothing` when `sigma == 0` (surface tension disabled).
"""
function compute_surface_tension_force(
        alpha::CollocatedScalarField{T},
        props::TwoPhaseProperties{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    if !has_surface_tension(props)
        return nothing
    end

    nc = length(mesh.cell_volumes)
    grad_alpha = gradient(alpha, mesh)
    kappa = compute_curvature(alpha, mesh)

    force = Vector{SVector{Dim, T}}(undef, nc)
    for c in 1:nc
        force[c] = props.sigma * kappa[c] * grad_alpha[c]
    end

    return force
end

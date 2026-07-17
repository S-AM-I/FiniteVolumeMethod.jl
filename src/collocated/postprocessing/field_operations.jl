# postprocessing/field_operations.jl — Derived field computations
#
# Computes vorticity, Q-criterion, enstrophy, and Courant number from
# velocity and flux fields on UnstructuredFVMMesh.

using LinearAlgebra: norm, dot, cross

# -- Velocity gradient helper -------------------------------------------------

"""
    _compute_velocity_gradients(U, mesh) -> Vector{Vector{SVector{Dim, T}}}

Compute the gradient of each velocity component. Returns `grad_U` where
`grad_U[d][c]` is the gradient of component `d` at cell `c`.
"""
function _compute_velocity_gradients(
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    grad_U = Vector{Vector{SVector{Dim, T}}}(undef, Dim)

    for d in 1:Dim
        u_d = CollocatedScalarField(Symbol(:U, d), mesh; value = zero(T))
        for c in 1:nc
            u_d.internal[c] = U.internal[c][d]
        end
        for (i, f) in enumerate(u_d.boundary_face_indices)
            bi = findfirst(==(f), U.boundary_face_indices)
            if bi !== nothing
                u_d.boundary[i] = U.boundary[bi][d]
            end
        end
        grad_U[d] = gradient(u_d, mesh)
    end

    return grad_U
end

# -- Vorticity ----------------------------------------------------------------

"""
    compute_vorticity(U, mesh) -> Vector{T}  (2D)

Compute the z-component of vorticity at each cell:
`omega_z = dv/dx - du/dy`
"""
function compute_vorticity(
        U::CollocatedVectorField{2, T},
        mesh::UnstructuredFVMMesh{2, T},
    ) where {T}
    grad_U = _compute_velocity_gradients(U, mesh)
    nc = length(mesh.cell_volumes)
    omega = Vector{T}(undef, nc)
    for c in 1:nc
        dvdx = grad_U[2][c][1]
        dudy = grad_U[1][c][2]
        omega[c] = dvdx - dudy
    end
    return omega
end

"""
    compute_vorticity(U, mesh) -> Vector{SVector{3, T}}  (3D)

Compute the vorticity vector at each cell:
`omega = curl(U) = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)`
"""
function compute_vorticity(
        U::CollocatedVectorField{3, T},
        mesh::UnstructuredFVMMesh{3, T},
    ) where {T}
    grad_U = _compute_velocity_gradients(U, mesh)
    nc = length(mesh.cell_volumes)
    omega = Vector{SVector{3, T}}(undef, nc)
    for c in 1:nc
        dudy = grad_U[1][c][2]; dudz = grad_U[1][c][3]
        dvdx = grad_U[2][c][1]; dvdz = grad_U[2][c][3]
        dwdx = grad_U[3][c][1]; dwdy = grad_U[3][c][2]
        omega[c] = SVector{3, T}(dwdy - dvdz, dudz - dwdx, dvdx - dudy)
    end
    return omega
end

# -- Q-criterion --------------------------------------------------------------

"""
    compute_q_criterion(U, mesh) -> Vector{T}

Compute the Q-criterion at each cell:
`Q = 0.5 * (|Omega|^2 - |S|^2)`

Positive Q identifies vortex cores.
"""
function compute_q_criterion(
        U::CollocatedVectorField{2, T},
        mesh::UnstructuredFVMMesh{2, T},
    ) where {T}
    grad_U = _compute_velocity_gradients(U, mesh)
    nc = length(mesh.cell_volumes)
    Q = Vector{T}(undef, nc)
    for c in 1:nc
        dudx = grad_U[1][c][1]; dudy = grad_U[1][c][2]
        dvdx = grad_U[2][c][1]; dvdy = grad_U[2][c][2]

        S_11 = dudx; S_22 = dvdy
        S_12 = T(0.5) * (dudy + dvdx)
        Omega_12 = T(0.5) * (dvdx - dudy)

        S_sq = S_11^2 + S_22^2 + T(2) * S_12^2
        Omega_sq = T(2) * Omega_12^2

        Q[c] = T(0.5) * (Omega_sq - S_sq)
    end
    return Q
end

function compute_q_criterion(
        U::CollocatedVectorField{3, T},
        mesh::UnstructuredFVMMesh{3, T},
    ) where {T}
    grad_U = _compute_velocity_gradients(U, mesh)
    nc = length(mesh.cell_volumes)
    Q = Vector{T}(undef, nc)
    for c in 1:nc
        dudx = grad_U[1][c][1]; dudy = grad_U[1][c][2]; dudz = grad_U[1][c][3]
        dvdx = grad_U[2][c][1]; dvdy = grad_U[2][c][2]; dvdz = grad_U[2][c][3]
        dwdx = grad_U[3][c][1]; dwdy = grad_U[3][c][2]; dwdz = grad_U[3][c][3]

        S_11 = dudx; S_22 = dvdy; S_33 = dwdz
        S_12 = T(0.5) * (dudy + dvdx)
        S_13 = T(0.5) * (dudz + dwdx)
        S_23 = T(0.5) * (dvdz + dwdy)
        S_sq = S_11^2 + S_22^2 + S_33^2 + T(2) * (S_12^2 + S_13^2 + S_23^2)

        O_12 = T(0.5) * (dvdx - dudy)
        O_13 = T(0.5) * (dwdx - dudz)
        O_23 = T(0.5) * (dwdy - dvdz)
        Omega_sq = T(2) * (O_12^2 + O_13^2 + O_23^2)

        Q[c] = T(0.5) * (Omega_sq - S_sq)
    end
    return Q
end

# -- Enstrophy ----------------------------------------------------------------

"""
    compute_enstrophy(U, mesh) -> Vector{T}

Compute enstrophy `|omega|^2` at each cell.
"""
function compute_enstrophy(
        U::CollocatedVectorField{2, T},
        mesh::UnstructuredFVMMesh{2, T},
    ) where {T}
    omega = compute_vorticity(U, mesh)
    return [w^2 for w in omega]
end

function compute_enstrophy(
        U::CollocatedVectorField{3, T},
        mesh::UnstructuredFVMMesh{3, T},
    ) where {T}
    omega = compute_vorticity(U, mesh)
    return [dot(w, w) for w in omega]
end

# -- Courant number ------------------------------------------------------------

"""
    compute_courant_number(phi, mesh, dt) -> Vector{T}

Compute the Courant number per cell:
`Co = dt * sum_f |phi_f| / (2 * V_c)`

Requires `mesh.cell_faces` to be populated.
"""
function compute_courant_number(
        phi::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    Co = zeros(T, nc)
    mesh.cell_faces === nothing && error("cell_faces required for Courant number")

    for c in 1:nc
        flux_sum = zero(T)
        for f in mesh.cell_faces[c]
            flux_sum += abs(phi.values[f])
        end
        Co[c] = dt * flux_sum / (T(2) * mesh.cell_volumes[c])
    end

    return Co
end

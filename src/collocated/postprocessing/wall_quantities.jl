# postprocessing/wall_quantities.jl — Wall surface metrics
#
# Computes wall shear stress, y+, heat flux, and Nusselt number at
# named boundary patches on UnstructuredFVMMesh.

using LinearAlgebra: norm, dot

# -- Patch face helper ---------------------------------------------------------

"""
    _patch_faces(mesh, patch::Symbol) -> Vector{Int}

Return face indices belonging to boundary patch `patch`.
"""
function _patch_faces(mesh::UnstructuredFVMMesh{Dim, T}, patch::Symbol) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    faces = Int[]
    for f in 1:nf
        if !is_internal_face(mesh, f)
            tag = _face_tag(mesh, f)
            tag == patch && push!(faces, f)
        end
    end
    return faces
end

# -- Wall shear stress ---------------------------------------------------------

"""
    compute_wall_shear_stress(U, nu, mesh, patch) -> Vector{SVector{Dim, T}}

Compute wall shear stress at each face of boundary `patch`.

Uses the linear near-wall approximation:
`tau_w = nu * U_tangential / d`

where `d` is the distance from the cell center to the face center and
`U_tangential` is the velocity component parallel to the wall.
"""
function compute_wall_shear_stress(
        U::CollocatedVectorField{Dim, T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    tau = Vector{SVector{Dim, T}}(undef, length(faces))

    for (i, f) in enumerate(faces)
        P = owner(mesh, f)
        U_P = U.internal[P]
        x_P = cell_center(mesh, P)
        x_f = face_center(mesh, f)
        d_vec = x_f - x_P
        d = norm(d_vec)

        if d > zero(T)
            n_hat = d_vec / d
            U_normal = dot(U_P, n_hat) * n_hat
            U_tan = U_P - U_normal
            tau[i] = nu * U_tan / d
        else
            tau[i] = zero(SVector{Dim, T})
        end
    end

    return tau
end

# -- y+ -----------------------------------------------------------------------

"""
    compute_y_plus(U, nu, mesh, patch) -> Vector{T}

Compute y+ at each face of boundary `patch`.

`y+ = y * u_tau / nu` where `u_tau = sqrt(|tau_w|)` (rho = 1 for incompressible).
"""
function compute_y_plus(
        U::CollocatedVectorField{Dim, T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    tau = compute_wall_shear_stress(U, nu, mesh, patch)
    yp = Vector{T}(undef, length(faces))

    for (i, f) in enumerate(faces)
        P = owner(mesh, f)
        x_P = cell_center(mesh, P)
        x_f = face_center(mesh, f)
        y = norm(x_f - x_P)

        tau_mag = norm(tau[i])
        u_tau = sqrt(tau_mag)
        yp[i] = nu > zero(T) ? y * u_tau / nu : zero(T)
    end

    return yp
end

# -- Wall heat flux ------------------------------------------------------------

"""
    compute_wall_heat_flux(T_field, k, mesh, patch) -> Vector{T}

Compute wall heat flux at each face of boundary `patch`:
`q_w = -k * (T_wall - T_cell) / d`

Positive q means heat flows out of the domain.
"""
function compute_wall_heat_flux(
        T_field::CollocatedScalarField{T},
        k::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    pbmap = build_boundary_map(T_field)
    q = Vector{T}(undef, length(faces))

    for (i, f) in enumerate(faces)
        P = owner(mesh, f)
        T_cell = T_field.internal[P]
        T_wall = T_field.boundary[pbmap[f]]
        x_P = cell_center(mesh, P)
        x_f = face_center(mesh, f)
        d = norm(x_f - x_P)

        q[i] = d > zero(T) ? -k * (T_wall - T_cell) / d : zero(T)
    end

    return q
end

# -- Nusselt number ------------------------------------------------------------

"""
    compute_nusselt_number(T_field, k, mesh, patch; T_ref, L_ref) -> Vector{T}

Compute Nusselt number at each face of boundary `patch`:
`Nu = q_w * L_ref / (k * (T_wall - T_ref))`
"""
function compute_nusselt_number(
        T_field::CollocatedScalarField{T},
        k::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol;
        T_ref::T,
        L_ref::T,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    q_w = compute_wall_heat_flux(T_field, k, mesh, patch)
    pbmap = build_boundary_map(T_field)
    Nu = Vector{T}(undef, length(faces))

    for (i, f) in enumerate(faces)
        T_wall = T_field.boundary[pbmap[f]]
        dT = T_wall - T_ref
        if abs(dT) > eps(T) && k > zero(T)
            Nu[i] = abs(q_w[i]) * L_ref / (k * abs(dT))
        else
            Nu[i] = zero(T)
        end
    end

    return Nu
end

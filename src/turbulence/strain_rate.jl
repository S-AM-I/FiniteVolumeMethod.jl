# turbulence/strain_rate.jl — Strain rate magnitude from velocity gradients
#
# Computes |S| = sqrt(2 * S_ij * S_ij) where S_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i).
# Used by all RANS models to compute turbulence production.

"""
    compute_strain_rate(
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) -> Vector{T}

Compute the strain rate magnitude `|S|` at each cell center from the
velocity field `U`.

Uses Green-Gauss gradient reconstruction to compute velocity gradients,
then assembles the symmetric strain rate tensor and returns its magnitude.

For 2D: `|S| = sqrt(2*(S_xx² + S_yy² + 2*S_xy²))`
For 3D: `|S| = sqrt(2*(S_xx² + S_yy² + S_zz² + 2*(S_xy² + S_xz² + S_yz²)))`
"""
function compute_strain_rate(
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    S_mag = Vector{T}(undef, nc)

    # Compute gradient of each velocity component.  The face → boundary
    # slot map is built once (O(n_b)) instead of a findfirst per face
    # (O(n_b²)).
    ubmap = build_boundary_map(U, mesh)
    grad_U = Vector{Vector{SVector{Dim, T}}}(undef, Dim)
    for d in 1:Dim
        u_d_field = CollocatedScalarField(
            Symbol(:U, d), mesh;
            value = zero(T),
        )
        # Copy component d into scalar field
        for c in 1:nc
            u_d_field.internal[c] = U.internal[c][d]
        end
        # Copy boundary values
        for (i, f) in enumerate(u_d_field.boundary_face_indices)
            bi = ubmap[f]
            if bi != 0
                u_d_field.boundary[i] = U.boundary[bi][d]
            end
        end
        grad_U[d] = gradient(u_d_field, mesh)
    end

    # Assemble strain rate magnitude per cell
    for c in 1:nc
        S_sq = _strain_rate_squared(Val(Dim), grad_U, c)
        S_mag[c] = sqrt(max(S_sq, zero(T)))
    end

    return S_mag
end

"""2D strain rate: 2*(S_xx² + S_yy² + 2*S_xy²)"""
function _strain_rate_squared(
        ::Val{2}, grad_U::Vector{Vector{SVector{2, T}}}, c::Int,
    ) where {T}
    dudx = grad_U[1][c][1]
    dudy = grad_U[1][c][2]
    dvdx = grad_U[2][c][1]
    dvdy = grad_U[2][c][2]

    S_xx = dudx
    S_yy = dvdy
    S_xy = T(0.5) * (dudy + dvdx)

    return T(2) * (S_xx^2 + S_yy^2 + T(2) * S_xy^2)
end

"""3D strain rate: 2*(S_xx² + S_yy² + S_zz² + 2*(S_xy² + S_xz² + S_yz²))"""
function _strain_rate_squared(
        ::Val{3}, grad_U::Vector{Vector{SVector{3, T}}}, c::Int,
    ) where {T}
    dudx = grad_U[1][c][1]; dudy = grad_U[1][c][2]; dudz = grad_U[1][c][3]
    dvdx = grad_U[2][c][1]; dvdy = grad_U[2][c][2]; dvdz = grad_U[2][c][3]
    dwdx = grad_U[3][c][1]; dwdy = grad_U[3][c][2]; dwdz = grad_U[3][c][3]

    S_xx = dudx; S_yy = dvdy; S_zz = dwdz
    S_xy = T(0.5) * (dudy + dvdx)
    S_xz = T(0.5) * (dudz + dwdx)
    S_yz = T(0.5) * (dvdz + dwdy)

    return T(2) * (S_xx^2 + S_yy^2 + S_zz^2 + T(2) * (S_xy^2 + S_xz^2 + S_yz^2))
end

# turbulence/wale.jl — WALE (Wall-Adapting Local Eddy-viscosity) SGS model
#
# Better near-wall behavior than Smagorinsky — ν_sgs vanishes at walls
# without explicit damping functions.
#
# ν_sgs = (Cw·Δ)² · (S_d:S_d)^(3/2) / ((S:S)^(5/2) + (S_d:S_d)^(5/4))
# where S_d is the traceless symmetric part of the squared velocity gradient.

"""
    WALE{T} <: AbstractLESModel

WALE (Wall-Adapting Local Eddy-viscosity) SGS model.

# Fields
- `Cw::T` — WALE constant (default 0.325)
- `delta::Vector{T}` — grid filter width per cell
"""
struct WALE{T} <: AbstractLESModel
    Cw::T
    delta::Vector{T}
end

"""
    WALE(mesh; Cw = 0.325)

Construct a WALE model, computing filter width from `mesh`.
"""
function WALE(mesh::UnstructuredFVMMesh{Dim, T}; Cw::Real = 0.325) where {Dim, T}
    delta = compute_filter_width(mesh)
    return WALE{T}(T(Cw), delta)
end

"""
    _compute_velocity_gradient_tensor(U, mesh) -> Vector{Vector{SVector{Dim, T}}}

Compute the full velocity gradient tensor ∂u_i/∂x_j per cell using
Green-Gauss gradient reconstruction. Returns `grad_U[i][c][j]` = ∂u_i/∂x_j
at cell `c`.

Uses the same pattern as `compute_strain_rate` — creates temporary scalar
fields per velocity component and calls `gradient()`.
"""
function _compute_velocity_gradient_tensor(
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    grad_U = Vector{Vector{SVector{Dim, T}}}(undef, Dim)
    for d in 1:Dim
        u_d_field = CollocatedScalarField(
            Symbol(:U, d), mesh;
            value = zero(T),
        )
        for c in 1:nc
            u_d_field.internal[c] = U.internal[c][d]
        end
        for (i, f) in enumerate(u_d_field.boundary_face_indices)
            bi = findfirst(==(f), U.boundary_face_indices)
            if bi !== nothing
                u_d_field.boundary[i] = U.boundary[bi][d]
            end
        end
        grad_U[d] = gradient(u_d_field, mesh)
    end
    return grad_U
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::WALE{T},
        U::CollocatedVectorField{2, T},
        mesh::UnstructuredFVMMesh{2, T},
    ) where {T}
    nc = length(mesh.cell_volumes)
    grad_U = _compute_velocity_gradient_tensor(U, mesh)

    for c in 1:nc
        g11 = grad_U[1][c][1]; g12 = grad_U[1][c][2]
        g21 = grad_U[2][c][1]; g22 = grad_U[2][c][2]

        # Squared velocity gradient g²_ij = g_ik * g_kj
        g2_11 = g11 * g11 + g12 * g21
        g2_12 = g11 * g12 + g12 * g22
        g2_21 = g21 * g11 + g22 * g21
        g2_22 = g21 * g12 + g22 * g22

        # Traceless symmetric part: S_d_ij = 0.5*(g²_ij + g²_ji) - (1/2)*δ_ij*g²_kk
        # For 2D, use (1/2)*δ_ij*trace instead of (1/3)
        trace_g2 = g2_11 + g2_22
        sd_11 = T(0.5) * (g2_11 + g2_11) - T(0.5) * trace_g2
        sd_22 = T(0.5) * (g2_22 + g2_22) - T(0.5) * trace_g2
        sd_12 = T(0.5) * (g2_12 + g2_21)

        # S_d:S_d = sd_ij * sd_ij
        sd_sq = sd_11^2 + sd_22^2 + T(2) * sd_12^2

        # S:S (strain rate)
        S_11 = g11; S_22 = g22
        S_12 = T(0.5) * (g12 + g21)
        s_sq = S_11^2 + S_22^2 + T(2) * S_12^2

        # WALE viscosity
        denom = s_sq^T(2.5) + sd_sq^T(1.25)
        if denom > eps(T)
            nu_t[c] = (model.Cw * model.delta[c])^2 * sd_sq^T(1.5) / denom
        else
            nu_t[c] = zero(T)
        end
    end

    return nothing
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::WALE{T},
        U::CollocatedVectorField{3, T},
        mesh::UnstructuredFVMMesh{3, T},
    ) where {T}
    nc = length(mesh.cell_volumes)
    grad_U = _compute_velocity_gradient_tensor(U, mesh)

    for c in 1:nc
        g = ntuple(i -> ntuple(j -> grad_U[i][c][j], Val(3)), Val(3))

        # g²_ij = g_ik * g_kj
        g2 = ntuple(Val(3)) do i
            ntuple(Val(3)) do j
                g[i][1] * g[1][j] + g[i][2] * g[2][j] + g[i][3] * g[3][j]
            end
        end

        trace_g2 = g2[1][1] + g2[2][2] + g2[3][3]

        # S_d_ij = 0.5*(g²_ij + g²_ji) - (1/3)*δ_ij*trace
        sd = ntuple(Val(3)) do i
            ntuple(Val(3)) do j
                sym = T(0.5) * (g2[i][j] + g2[j][i])
                diag_part = (i == j) ? trace_g2 / T(3) : zero(T)
                sym - diag_part
            end
        end

        sd_sq = zero(T)
        s_sq = zero(T)
        for i in 1:3, j in 1:3
            sd_sq += sd[i][j]^2
            S_ij = T(0.5) * (g[i][j] + g[j][i])
            s_sq += S_ij^2
        end

        denom = s_sq^T(2.5) + sd_sq^T(1.25)
        if denom > eps(T)
            nu_t[c] = (model.Cw * model.delta[c])^2 * sd_sq^T(1.5) / denom
        else
            nu_t[c] = zero(T)
        end
    end

    return nothing
end

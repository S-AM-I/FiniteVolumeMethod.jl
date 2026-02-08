# ============================================================
# Characteristic-wise WENO Reconstruction
# ============================================================
#
# Projects to characteristic variables before applying WENO
# reconstruction, then projects back. This avoids oscillations
# that arise from component-wise reconstruction near strong
# waves of different families.
#
# For Euler equations, we provide analytic left/right eigenvector
# matrices of the flux Jacobian. For other systems, a generic
# fallback does component-wise reconstruction (no projection).

"""
    CharacteristicWENO{R}

Wrapper that applies characteristic decomposition before WENO
reconstruction and projects back afterwards.

The inner reconstruction `recon` must be a WENO-type scheme
(e.g., `WENO3` or `WENO5`).

# Fields
- `recon::R`: Underlying WENO reconstruction scheme.
"""
struct CharacteristicWENO{R}
    recon::R
end

nghost(cw::CharacteristicWENO) = nghost(cw.recon)

# ============================================================
# Eigenvector interface
# ============================================================

"""
    left_eigenvectors(law, w, dir) -> L

Return the matrix of left eigenvectors (rows are left eigenvectors)
of the flux Jacobian ∂F/∂U in direction `dir`, evaluated at the
primitive state `w`.

The left eigenvector matrix satisfies `L * R = I` where `R` is the
right eigenvector matrix.
"""
function left_eigenvectors end

"""
    right_eigenvectors(law, w, dir) -> R

Return the matrix of right eigenvectors (columns are right eigenvectors)
of the flux Jacobian ∂F/∂U in direction `dir`, evaluated at the
primitive state `w`.
"""
function right_eigenvectors end

# ============================================================
# 1D Euler Eigenvectors
# ============================================================

"""
    right_eigenvectors(law::EulerEquations{1}, w::SVector{3}, dir::Int) -> SMatrix{3,3}

Right eigenvector matrix of the 1D Euler flux Jacobian.

Primitive state: `w = [ρ, v, P]`.
Eigenvalues: `λ₁ = v - c, λ₂ = v, λ₃ = v + c`.

Columns are the right eigenvectors `r₁, r₂, r₃`.
"""
@inline function right_eigenvectors(law::EulerEquations{1}, w::SVector{3}, dir::Int)
    ρ, v, P = w
    γ = law.eos.gamma
    ρ = max(ρ, 1e-12)
    P = max(P, 1e-12)
    c = sound_speed(law.eos, ρ, P)
    H = (total_energy(law.eos, ρ, v, P) + P) / ρ  # specific enthalpy

    # Right eigenvectors as columns:
    #   r1 = [1,  v-c,  H - v*c]
    #   r2 = [1,  v,    v²/2  ]
    #   r3 = [1,  v+c,  H + v*c]
    return StaticArrays.SMatrix{3, 3}(
        one(ρ), v - c, H - v * c,                  # column 1
        one(ρ), v, 0.5 * v^2,                      # column 2
        one(ρ), v + c, H + v * c                    # column 3
    )
end

"""
    left_eigenvectors(law::EulerEquations{1}, w::SVector{3}, dir::Int) -> SMatrix{3,3}

Left eigenvector matrix (rows are left eigenvectors) of the 1D Euler flux Jacobian.
"""
@inline function left_eigenvectors(law::EulerEquations{1}, w::SVector{3}, dir::Int)
    ρ, v, P = w
    γ = law.eos.gamma
    ρ = max(ρ, 1e-12)
    P = max(P, 1e-12)
    c = sound_speed(law.eos, ρ, P)
    b1 = (γ - 1) / c^2
    b2 = 0.5 * b1 * v^2

    # Left eigenvectors as rows:
    #   l1 = 0.5 * [b2 + v/c,  -(b1*v + 1/c),  b1]
    #   l2 =       [1 - b2,     b1*v,          -b1]
    #   l3 = 0.5 * [b2 - v/c,  -(b1*v - 1/c),  b1]
    return StaticArrays.SMatrix{3, 3}(
        0.5 * (b2 + v / c), one(ρ) - b2, 0.5 * (b2 - v / c),         # column 1 (= row elements col 1)
        0.5 * (-(b1 * v + one(ρ) / c)), b1 * v, 0.5 * (-(b1 * v - one(ρ) / c)),  # column 2
        0.5 * b1, -b1, 0.5 * b1                                        # column 3
    )
end

# ============================================================
# 2D Euler Eigenvectors (x-direction, dir=1)
# ============================================================

"""
    right_eigenvectors(law::EulerEquations{2}, w::SVector{4}, dir::Int) -> SMatrix{4,4}

Right eigenvector matrix of the 2D Euler flux Jacobian in direction `dir`.

Primitive state: `w = [ρ, vx, vy, P]`.
"""
@inline function right_eigenvectors(law::EulerEquations{2}, w::SVector{4}, dir::Int)
    ρ, vx, vy, P = w
    γ = law.eos.gamma
    ρ = max(ρ, 1e-12)
    P = max(P, 1e-12)
    c = sound_speed(law.eos, ρ, P)
    H = (total_energy(law.eos, ρ, vx, vy, P) + P) / ρ

    if dir == 1
        # Eigenvalues: vx-c, vx, vx, vx+c
        # Right eigenvectors as columns
        return StaticArrays.SMatrix{4, 4}(
            one(ρ), vx - c, vy, H - vx * c,          # r1
            one(ρ), vx, vy, 0.5 * (vx^2 + vy^2),     # r2
            zero(ρ), zero(ρ), one(ρ), vy,              # r3
            one(ρ), vx + c, vy, H + vx * c             # r4
        )
    else
        # dir == 2: eigenvalues vy-c, vy, vy, vy+c
        return StaticArrays.SMatrix{4, 4}(
            one(ρ), vx, vy - c, H - vy * c,          # r1
            one(ρ), vx, vy, 0.5 * (vx^2 + vy^2),     # r2
            zero(ρ), one(ρ), zero(ρ), vx,              # r3
            one(ρ), vx, vy + c, H + vy * c             # r4
        )
    end
end

"""
    left_eigenvectors(law::EulerEquations{2}, w::SVector{4}, dir::Int) -> SMatrix{4,4}

Left eigenvector matrix (rows are left eigenvectors) of the 2D Euler flux Jacobian.
"""
@inline function left_eigenvectors(law::EulerEquations{2}, w::SVector{4}, dir::Int)
    ρ, vx, vy, P = w
    γ = law.eos.gamma
    ρ = max(ρ, 1e-12)
    P = max(P, 1e-12)
    c = sound_speed(law.eos, ρ, P)
    b1 = (γ - 1) / c^2
    q2 = vx^2 + vy^2
    b2 = 0.5 * b1 * q2

    if dir == 1
        # Left eigenvectors as rows
        # l1 = 0.5 * [b2 + vx/c,  -(b1*vx + 1/c),  -b1*vy,  b1]
        # l2 =       [1 - b2,      b1*vx,            b1*vy,  -b1]
        # l3 =       [-vy,         0,                 1,       0]
        # l4 = 0.5 * [b2 - vx/c,  -(b1*vx - 1/c),  -b1*vy,  b1]
        return StaticArrays.SMatrix{4, 4}(
            # column 1 (first element of each row)
            0.5 * (b2 + vx / c),
            one(ρ) - b2,
            -vy,
            0.5 * (b2 - vx / c),
            # column 2
            0.5 * (-(b1 * vx + one(ρ) / c)),
            b1 * vx,
            zero(ρ),
            0.5 * (-(b1 * vx - one(ρ) / c)),
            # column 3
            -0.5 * b1 * vy,
            b1 * vy,
            one(ρ),
            -0.5 * b1 * vy,
            # column 4
            0.5 * b1,
            -b1,
            zero(ρ),
            0.5 * b1
        )
    else
        # dir == 2: swap roles of vx and vy
        return StaticArrays.SMatrix{4, 4}(
            0.5 * (b2 + vy / c),
            one(ρ) - b2,
            -vx,
            0.5 * (b2 - vy / c),
            -0.5 * b1 * vx,
            b1 * vx,
            one(ρ),
            -0.5 * b1 * vx,
            0.5 * (-(b1 * vy + one(ρ) / c)),
            b1 * vy,
            zero(ρ),
            0.5 * (-(b1 * vy - one(ρ) / c)),
            0.5 * b1,
            -b1,
            zero(ρ),
            0.5 * b1
        )
    end
end

# ============================================================
# Generic fallback: no characteristic projection
# ============================================================

# For conservation laws without explicit eigenvectors, fall back
# to component-wise reconstruction (equivalent to no projection).

"""
    left_eigenvectors(law::AbstractConservationLaw, w, dir) -> identity

Generic fallback returning the identity matrix (no projection).
"""
@inline function left_eigenvectors(law::AbstractConservationLaw, w::SVector{N}, dir::Int) where {N}
    return StaticArrays.SMatrix{N, N}(one(StaticArrays.SMatrix{N, N, eltype(w)}))
end

"""
    right_eigenvectors(law::AbstractConservationLaw, w, dir) -> identity

Generic fallback returning the identity matrix (no projection).
"""
@inline function right_eigenvectors(law::AbstractConservationLaw, w::SVector{N}, dir::Int) where {N}
    return StaticArrays.SMatrix{N, N}(one(StaticArrays.SMatrix{N, N, eltype(w)}))
end

# ============================================================
# Characteristic WENO-3 interface (4-point stencil)
# ============================================================

"""
    reconstruct_interface(cw::CharacteristicWENO{<:WENO3}, wLL, wL, wR, wRR) -> (wL_face, wR_face)

Component-wise fallback when called without law information.

The actual characteristic projection is performed in the 1D/2D dispatch
functions (`reconstruct_interface_1d`, `_reconstruct_face_2d`, etc.)
where the conservation law is available.
"""
@inline function reconstruct_interface(cw::CharacteristicWENO{<:WENO3}, wLL::SVector{N}, wL::SVector{N}, wR::SVector{N}, wRR::SVector{N}) where {N}
    return reconstruct_interface(cw.recon, wLL, wL, wR, wRR)
end

# ============================================================
# Characteristic WENO-5 interface (5-point stencil)
# ============================================================

"""
    reconstruct_interface_weno5_char(cw::CharacteristicWENO{<:WENO5}, law, dir, v1, v2, v3, v4, v5)

Characteristic-wise WENO-5 reconstruction at a cell interface.
"""
@inline function reconstruct_interface_weno5_char(cw::CharacteristicWENO{<:WENO5}, law, dir::Int, v1::SVector{N}, v2::SVector{N}, v3::SVector{N}, v4::SVector{N}, v5::SVector{N}, v6::SVector{N}) where {N}
    # Average state at the interface (between v3 and v4)
    w_avg = 0.5 * (v3 + v4)

    L = left_eigenvectors(law, w_avg, dir)
    R = right_eigenvectors(law, w_avg, dir)

    # Project to characteristic variables
    c1 = L * v1
    c2 = L * v2
    c3 = L * v3
    c4 = L * v4
    c5 = L * v5
    c6 = L * v6

    # Reconstruct in characteristic space
    cL_face, cR_face = reconstruct_interface_weno5(cw.recon, c1, c2, c3, c4, c5, c6)

    # Project back
    wL_face = R * cL_face
    wR_face = R * cR_face

    return wL_face, wR_face
end

# Without law: component-wise fallback
@inline function reconstruct_interface_weno5(cw::CharacteristicWENO{<:WENO5}, v1::SVector{N}, v2::SVector{N}, v3::SVector{N}, v4::SVector{N}, v5::SVector{N}, v6::SVector{N}) where {N}
    return reconstruct_interface_weno5(cw.recon, v1, v2, v3, v4, v5, v6)
end

# ============================================================
# 1D dispatch for CharacteristicWENO
# ============================================================

@inline function reconstruct_interface_1d(cw::CharacteristicWENO{<:WENO3}, law, U::AbstractVector, face_idx::Int, ncells::Int)
    iL = face_idx + 2
    iR = face_idx + 3

    wLL = conserved_to_primitive(law, U[iL - 1])
    wL = conserved_to_primitive(law, U[iL])
    wR = conserved_to_primitive(law, U[iR])
    wRR = conserved_to_primitive(law, U[iR + 1])

    # Characteristic projection with law
    w_avg = 0.5 * (wL + wR)
    L = left_eigenvectors(law, w_avg, 1)
    R = right_eigenvectors(law, w_avg, 1)

    cLL = L * wLL
    cL = L * wL
    cR = L * wR
    cRR = L * wRR

    cL_face, cR_face = reconstruct_interface(cw.recon, cLL, cL, cR, cRR)

    wL_face = R * cL_face
    wR_face = R * cR_face

    return wL_face, wR_face
end

@inline function reconstruct_interface_1d(cw::CharacteristicWENO{<:WENO5}, law, U::AbstractVector, face_idx::Int, ncells::Int)
    v1 = conserved_to_primitive(law, U[face_idx + 1])
    v2 = conserved_to_primitive(law, U[face_idx + 2])
    v3 = conserved_to_primitive(law, U[face_idx + 3])
    v4 = conserved_to_primitive(law, U[face_idx + 4])
    v5 = conserved_to_primitive(law, U[face_idx + 5])
    v6 = conserved_to_primitive(law, U[face_idx + 6])

    return reconstruct_interface_weno5_char(cw, law, 1, v1, v2, v3, v4, v5, v6)
end

# ============================================================
# 2D dispatch for CharacteristicWENO
# ============================================================

# WENO-3 characteristic, x-direction
@inline function _reconstruct_face_2d(
        cw::CharacteristicWENO{<:WENO3}, law, U::AbstractMatrix,
        iL::Int, iR::Int, jj::Int, dir::Int, nx::Int
    )
    wLL = conserved_to_primitive(law, U[iL - 1, jj])
    wL = conserved_to_primitive(law, U[iL, jj])
    wR = conserved_to_primitive(law, U[iR, jj])
    wRR = conserved_to_primitive(law, U[iR + 1, jj])

    w_avg = 0.5 * (wL + wR)
    L = left_eigenvectors(law, w_avg, dir)
    R = right_eigenvectors(law, w_avg, dir)

    cLL = L * wLL
    cL = L * wL
    cR = L * wR
    cRR = L * wRR

    cL_face, cR_face = reconstruct_interface(cw.recon, cLL, cL, cR, cRR)

    return R * cL_face, R * cR_face
end

# WENO-3 characteristic, y-direction
@inline function _reconstruct_face_2d_y(
        cw::CharacteristicWENO{<:WENO3}, law, U::AbstractMatrix,
        ii::Int, jL::Int, jR::Int, ny::Int
    )
    wLL = conserved_to_primitive(law, U[ii, jL - 1])
    wL = conserved_to_primitive(law, U[ii, jL])
    wR = conserved_to_primitive(law, U[ii, jR])
    wRR = conserved_to_primitive(law, U[ii, jR + 1])

    w_avg = 0.5 * (wL + wR)
    L = left_eigenvectors(law, w_avg, 2)
    R = right_eigenvectors(law, w_avg, 2)

    cLL = L * wLL
    cL = L * wL
    cR = L * wR
    cRR = L * wRR

    cL_face, cR_face = reconstruct_interface(cw.recon, cLL, cL, cR, cRR)

    return R * cL_face, R * cR_face
end

# WENO-5 characteristic, x-direction
@inline function _reconstruct_face_2d(
        cw::CharacteristicWENO{<:WENO5}, law, U::AbstractMatrix,
        iL::Int, iR::Int, jj::Int, dir::Int, nx::Int
    )
    v1 = conserved_to_primitive(law, U[iL - 2, jj])
    v2 = conserved_to_primitive(law, U[iL - 1, jj])
    v3 = conserved_to_primitive(law, U[iL, jj])
    v4 = conserved_to_primitive(law, U[iR, jj])
    v5 = conserved_to_primitive(law, U[iR + 1, jj])
    v6 = conserved_to_primitive(law, U[iR + 2, jj])

    return reconstruct_interface_weno5_char(cw, law, dir, v1, v2, v3, v4, v5, v6)
end

# WENO-5 characteristic, y-direction
@inline function _reconstruct_face_2d_y(
        cw::CharacteristicWENO{<:WENO5}, law, U::AbstractMatrix,
        ii::Int, jL::Int, jR::Int, ny::Int
    )
    v1 = conserved_to_primitive(law, U[ii, jL - 2])
    v2 = conserved_to_primitive(law, U[ii, jL - 1])
    v3 = conserved_to_primitive(law, U[ii, jL])
    v4 = conserved_to_primitive(law, U[ii, jR])
    v5 = conserved_to_primitive(law, U[ii, jR + 1])
    v6 = conserved_to_primitive(law, U[ii, jR + 2])

    return reconstruct_interface_weno5_char(cw, law, 2, v1, v2, v3, v4, v5, v6)
end

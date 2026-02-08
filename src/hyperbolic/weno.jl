# ============================================================
# WENO-5 / WENO-Z Reconstruction
# ============================================================
#
# Fifth-order WENO reconstruction using 3 candidate polynomials
# on a 5-point stencil {i-2, i-1, i, i+1, i+2}.
#
# Requires 3 ghost cells on each side.
# Uses WENO-Z weights (Borges et al. 2008) for improved accuracy
# near smooth extrema.

"""
    WENO5{FT}

Weighted Essentially Non-Oscillatory reconstruction of order 5 with
WENO-Z nonlinear weights.

Uses a 5-point stencil with 3 candidate quadratic polynomials.
Requires 3 ghost cells on each side of the domain.

# Fields
- `epsilon::FT`: Small parameter to avoid division by zero (default `1e-6`).
- `p::Int`: Exponent for weight computation (default `2`).
"""
struct WENO5{FT}
    epsilon::FT
    p::Int
end

WENO5() = WENO5(1e-6, 2)

# ============================================================
# nghost: number of ghost cells required by each reconstruction
# ============================================================

"""
    nghost(recon) -> Int

Return the number of ghost cells required on each side of the domain
for the given reconstruction scheme.
"""
function nghost end

nghost(::NoReconstruction) = 1
nghost(::CellCenteredMUSCL) = 2
nghost(::WENO3) = 2
nghost(::WENO5) = 3

# ============================================================
# Scalar WENO-5 reconstruction
# ============================================================

"""
    _weno5_reconstruct_left(v1, v2, v3, v4, v5, eps, p) -> value

WENO-5 (WENO-Z) reconstruction at the i+1/2 interface from the left.

Input stencil: `v1 = u_{i-2}, v2 = u_{i-1}, v3 = u_i, v4 = u_{i+1}, v5 = u_{i+2}`.

Returns the left-biased reconstructed value at the right face of cell i.
"""
@inline function _weno5_reconstruct_left(v1, v2, v3, v4, v5, eps, p)
    # Ideal weights
    d0 = one(v1) / 10
    d1 = 6 * one(v1) / 10
    d2 = 3 * one(v1) / 10

    # Candidate polynomial values at x_{i+1/2}
    p0 = (2 * v1 - 7 * v2 + 11 * v3) / 6
    p1 = (-v2 + 5 * v3 + 2 * v4) / 6
    p2 = (2 * v3 + 5 * v4 - v5) / 6

    # Smoothness indicators (Jiang-Shu)
    beta0 = (13 * one(v1) / 12) * (v1 - 2 * v2 + v3)^2 + (one(v1) / 4) * (v1 - 4 * v2 + 3 * v3)^2
    beta1 = (13 * one(v1) / 12) * (v2 - 2 * v3 + v4)^2 + (one(v1) / 4) * (v2 - v4)^2
    beta2 = (13 * one(v1) / 12) * (v3 - 2 * v4 + v5)^2 + (one(v1) / 4) * (3 * v3 - 4 * v4 + v5)^2

    # WENO-Z weights: use tau5 = |beta0 - beta2| for improved accuracy
    tau5 = abs(beta0 - beta2)

    alpha0 = d0 * (1 + (tau5 / (eps + beta0))^p)
    alpha1 = d1 * (1 + (tau5 / (eps + beta1))^p)
    alpha2 = d2 * (1 + (tau5 / (eps + beta2))^p)
    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    return w0 * p0 + w1 * p1 + w2 * p2
end

"""
    _weno5_reconstruct_right(v1, v2, v3, v4, v5, eps, p) -> value

WENO-5 (WENO-Z) reconstruction at the i-1/2 interface from the right.

Input stencil: `v1 = u_{i-2}, v2 = u_{i-1}, v3 = u_i, v4 = u_{i+1}, v5 = u_{i+2}`.

Returns the right-biased reconstructed value at the left face of cell i.
This is obtained by mirroring the left reconstruction.
"""
@inline function _weno5_reconstruct_right(v1, v2, v3, v4, v5, eps, p)
    # Mirror the stencil: reconstruct at i-1/2 from the right
    return _weno5_reconstruct_left(v5, v4, v3, v2, v1, eps, p)
end

# ============================================================
# 5-point interface reconstruction
# ============================================================

"""
    reconstruct_interface_weno5(recon::WENO5, v1, v2, v3, v4, v5, v6) -> (left_face, right_face)

Reconstruct left and right states at the face between cells 3 and 4
in the stencil `(v1, v2, v3, v4, v5, v6)`.

- `v1..v6`: six consecutive cell values
- Face is between v3 and v4
- `left_face`: WENO-5 reconstruction from the left using stencil {v1,v2,v3,v4,v5}
- `right_face`: WENO-5 reconstruction from the right using stencil {v2,v3,v4,v5,v6}
"""
@inline function reconstruct_interface_weno5(recon::WENO5, v1, v2, v3, v4, v5, v6)
    eps = recon.epsilon
    p = recon.p
    # Left state: 5-point stencil centred on v3
    left_face = _weno5_reconstruct_left(v1, v2, v3, v4, v5, eps, p)
    # Right state: 5-point stencil centred on v4, mirrored
    right_face = _weno5_reconstruct_right(v2, v3, v4, v5, v6, eps, p)
    return left_face, right_face
end

# SVector component-wise overload
@inline function reconstruct_interface_weno5(recon::WENO5, v1::SVector{N}, v2::SVector{N}, v3::SVector{N}, v4::SVector{N}, v5::SVector{N}, v6::SVector{N}) where {N}
    eps = recon.epsilon
    p = recon.p
    left_face = SVector{N}(ntuple(i -> _weno5_reconstruct_left(v1[i], v2[i], v3[i], v4[i], v5[i], eps, p), Val(N)))
    right_face = SVector{N}(ntuple(i -> _weno5_reconstruct_right(v2[i], v3[i], v4[i], v5[i], v6[i], eps, p), Val(N)))
    return left_face, right_face
end

# ============================================================
# 1D dispatch for WENO-5
# ============================================================

"""
    reconstruct_interface_1d(recon::WENO5, law, U, face_idx, ncells) -> (wL, wR)

Reconstruct primitive left/right states at a 1D face using WENO-5.

For a face at position `face_idx` (0-based), needs 3 cells on each side:
- Left stencil (for left state): cells face_idx-1, face_idx, face_idx+1 (=left of face), face_idx+2, face_idx+3
- In padded array with 3-ghost offset: U[face_idx+1] .. U[face_idx+5]

With 3-ghost padding, interior cells are at indices 4:ncells+3, and
the padded array has length ncells+6.
"""
@inline function reconstruct_interface_1d(recon::WENO5, law, U::AbstractVector, face_idx::Int, ncells::Int)
    # With 3 ghost cells, the padding offset is 3.
    # Face face_idx (0-based) is between cells face_idx and face_idx+1.
    # In padded array: cell i (1-based) is at U[i+3].
    # Face between cell face_idx and face_idx+1 needs 6 cells:
    #   v1 = U[face_idx+1], ..., v6 = U[face_idx+6]
    # where v3 = left cell, v4 = right cell of the face.
    v1 = conserved_to_primitive(law, U[face_idx + 1])
    v2 = conserved_to_primitive(law, U[face_idx + 2])
    v3 = conserved_to_primitive(law, U[face_idx + 3])
    v4 = conserved_to_primitive(law, U[face_idx + 4])
    v5 = conserved_to_primitive(law, U[face_idx + 5])
    v6 = conserved_to_primitive(law, U[face_idx + 6])

    return reconstruct_interface_weno5(recon, v1, v2, v3, v4, v5, v6)
end

# ============================================================
# 2D dispatch for WENO-5 (x-direction and y-direction)
# ============================================================

"""
Reconstruct interface states for an x-direction face using WENO-5.

With 3-ghost padding the padded matrix is (nx+6) x (ny+6), interior
cells start at index 4.  Face between columns iL and iR in row jj.
"""
@inline function _reconstruct_face_2d(
        recon::WENO5, law, U::AbstractMatrix,
        iL::Int, iR::Int, jj::Int, dir::Int, nx::Int
    )
    v1 = conserved_to_primitive(law, U[iL - 2, jj])
    v2 = conserved_to_primitive(law, U[iL - 1, jj])
    v3 = conserved_to_primitive(law, U[iL, jj])
    v4 = conserved_to_primitive(law, U[iR, jj])
    v5 = conserved_to_primitive(law, U[iR + 1, jj])
    v6 = conserved_to_primitive(law, U[iR + 2, jj])

    return reconstruct_interface_weno5(recon, v1, v2, v3, v4, v5, v6)
end

"""
Reconstruct interface states for a y-direction face using WENO-5.
"""
@inline function _reconstruct_face_2d_y(
        recon::WENO5, law, U::AbstractMatrix,
        ii::Int, jL::Int, jR::Int, ny::Int
    )
    v1 = conserved_to_primitive(law, U[ii, jL - 2])
    v2 = conserved_to_primitive(law, U[ii, jL - 1])
    v3 = conserved_to_primitive(law, U[ii, jL])
    v4 = conserved_to_primitive(law, U[ii, jR])
    v5 = conserved_to_primitive(law, U[ii, jR + 1])
    v6 = conserved_to_primitive(law, U[ii, jR + 2])

    return reconstruct_interface_weno5(recon, v1, v2, v3, v4, v5, v6)
end

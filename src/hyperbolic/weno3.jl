# ============================================================
# WENO-3 Reconstruction (2-ghost compatible)
# ============================================================
#
# Third-order WENO reconstruction using 2 candidate polynomials
# on a 3-point stencil {i-1, i, i+1}.
#
# Works with existing 2-ghost-cell padding.

"""
    WENO3{FT}

Weighted Essentially Non-Oscillatory reconstruction of order 3.

Uses a 3-point stencil with 2 candidate linear polynomials and
nonlinear weights that adapt to solution smoothness. Compatible
with the existing 2-ghost-cell padding.

# Fields
- `epsilon::FT`: Small parameter to avoid division by zero in weights (default `1e-6`).
"""
struct WENO3{FT}
    epsilon::FT
end

WENO3() = WENO3(1e-6)

# ============================================================
# Scalar WENO-3 reconstruction
# ============================================================

"""
    _weno3_reconstruct_left(v0, v1, v2, eps) -> value

WENO-3 reconstruction of the left-biased value at the right face
of cell with value `v1`, given neighbours `v0` (left) and `v2` (right).

Returns the reconstructed value at the i+1/2 interface from the left.
"""
@inline function _weno3_reconstruct_left(v0, v1, v2, eps)
    # Ideal (linear) weights for left-biased reconstruction
    d0 = one(v0) / 3
    d1 = 2 * one(v0) / 3

    # Candidate polynomials evaluated at x_{i+1/2}
    p0 = -0.5 * v0 + 1.5 * v1
    p1 = 0.5 * v1 + 0.5 * v2

    # Smoothness indicators
    beta0 = (v1 - v0)^2
    beta1 = (v2 - v1)^2

    # Nonlinear weights
    alpha0 = d0 / (eps + beta0)^2
    alpha1 = d1 / (eps + beta1)^2
    alpha_sum = alpha0 + alpha1

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum

    return w0 * p0 + w1 * p1
end

"""
    _weno3_reconstruct_right(v0, v1, v2, eps) -> value

WENO-3 reconstruction of the right-biased value at the left face
of cell with value `v1`, given neighbours `v0` (left) and `v2` (right).

Returns the reconstructed value at the i-1/2 interface from the right.
"""
@inline function _weno3_reconstruct_right(v0, v1, v2, eps)
    # Ideal (linear) weights for right-biased reconstruction
    d0 = 2 * one(v0) / 3
    d1 = one(v0) / 3

    # Candidate polynomials evaluated at x_{i-1/2}
    p0 = 0.5 * v0 + 0.5 * v1
    p1 = 1.5 * v1 - 0.5 * v2

    # Smoothness indicators (same as left)
    beta0 = (v1 - v0)^2
    beta1 = (v2 - v1)^2

    # Nonlinear weights
    alpha0 = d0 / (eps + beta0)^2
    alpha1 = d1 / (eps + beta1)^2
    alpha_sum = alpha0 + alpha1

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum

    return w0 * p0 + w1 * p1
end

# ============================================================
# Interface matching the CellCenteredMUSCL pattern
# ============================================================

"""
    reconstruct_interface(recon::WENO3, wLL, wL, wR, wRR) -> (wL_face, wR_face)

Reconstruct interface states at the face between cell L and cell R
using WENO-3 reconstruction on primitive variables.

- `wLL`: value two cells to the left
- `wL`: value one cell to the left (owner)
- `wR`: value one cell to the right (neighbor)
- `wRR`: value two cells to the right

This follows the same interface as `CellCenteredMUSCL`.
"""
@inline function reconstruct_interface(recon::WENO3, wLL, wL, wR, wRR)
    eps = recon.epsilon
    # Left state at face: reconstruct from {wLL, wL, wR} biased left
    wL_face = _weno3_reconstruct_left(wLL, wL, wR, eps)
    # Right state at face: reconstruct from {wL, wR, wRR} biased right
    wR_face = _weno3_reconstruct_right(wL, wR, wRR, eps)
    return wL_face, wR_face
end

# SVector component-wise overload
@inline function reconstruct_interface(recon::WENO3, wLL::SVector{N}, wL::SVector{N}, wR::SVector{N}, wRR::SVector{N}) where {N}
    eps = recon.epsilon
    wL_face = SVector{N}(ntuple(i -> _weno3_reconstruct_left(wLL[i], wL[i], wR[i], eps), Val(N)))
    wR_face = SVector{N}(ntuple(i -> _weno3_reconstruct_right(wL[i], wR[i], wRR[i], eps), Val(N)))
    return wL_face, wR_face
end

# ============================================================
# 1D dispatch
# ============================================================

"""
    reconstruct_interface_1d(recon::WENO3, law, U, face_idx, ncells) -> (wL, wR)

Reconstruct primitive left/right states at a 1D face using WENO-3.
Uses the same stencil as `CellCenteredMUSCL` (4 cells: iL-1, iL, iR, iR+1).
"""
@inline function reconstruct_interface_1d(recon::WENO3, law, U::AbstractVector, face_idx::Int, ncells::Int)
    iL = face_idx + 2
    iR = face_idx + 3

    uLL = U[iL - 1]
    uL = U[iL]
    uR = U[iR]
    uRR = U[iR + 1]

    wLL = conserved_to_primitive(law, uLL)
    wL = conserved_to_primitive(law, uL)
    wR = conserved_to_primitive(law, uR)
    wRR = conserved_to_primitive(law, uRR)

    wL_face, wR_face = reconstruct_interface(recon, wLL, wL, wR, wRR)
    return wL_face, wR_face
end

# ============================================================
# 2D dispatch (x-direction and y-direction)
# ============================================================

"""
Reconstruct interface states for an x-direction face at column boundary (iL, iR) in row jj using WENO-3.
"""
@inline function _reconstruct_face_2d(
        recon::WENO3, law, U::AbstractMatrix,
        iL::Int, iR::Int, jj::Int, dir::Int, nx::Int
    )
    uLL = U[iL - 1, jj]
    uL = U[iL, jj]
    uR = U[iR, jj]
    uRR = U[iR + 1, jj]

    wLL = conserved_to_primitive(law, uLL)
    wL = conserved_to_primitive(law, uL)
    wR = conserved_to_primitive(law, uR)
    wRR = conserved_to_primitive(law, uRR)

    return reconstruct_interface(recon, wLL, wL, wR, wRR)
end

"""
Reconstruct interface states for a y-direction face at row boundary (jL, jR) in column ii using WENO-3.
"""
@inline function _reconstruct_face_2d_y(
        recon::WENO3, law, U::AbstractMatrix,
        ii::Int, jL::Int, jR::Int, ny::Int
    )
    uLL = U[ii, jL - 1]
    uL = U[ii, jL]
    uR = U[ii, jR]
    uRR = U[ii, jR + 1]

    wLL = conserved_to_primitive(law, uLL)
    wL = conserved_to_primitive(law, uL)
    wR = conserved_to_primitive(law, uR)
    wRR = conserved_to_primitive(law, uRR)

    return reconstruct_interface(recon, wLL, wL, wR, wRR)
end

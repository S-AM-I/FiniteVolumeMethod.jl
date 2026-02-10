# ============================================================
# PPM (Piecewise Parabolic Method) Reconstruction
# ============================================================
# Colella & Woodward (1984) third-order reconstruction.
# Uses a 4-cell stencil {i-1, i, i+1, i+2} — 2 ghost cells.

"""
    PPMReconstruction

Piecewise Parabolic Method reconstruction (Colella & Woodward 1984).
Third-order spatial reconstruction using parabolic interpolation
with monotonicity constraints.

Uses a 4-cell stencil and 2 ghost cells per side (same as MUSCL/WENO3).
"""
struct PPMReconstruction end

nghost(::PPMReconstruction) = 2

# ============================================================
# Scalar PPM helper functions
# ============================================================

"""
    _ppm_face_value(wL, wR, wLL, wRR) -> a

Fourth-order interpolation of the face value between cells L and R:

    a = (7*(wL + wR) - (wLL + wRR)) / 12

This is the unlimited parabolic interface value from C&W84 eq. 1.6.
"""
@inline function _ppm_face_value(wL, wR, wLL, wRR)
    return (7 * (wL + wR) - (wLL + wRR)) / 12
end

"""
    _ppm_monotonize(w, aL, aR) -> (aL_m, aR_m)

Apply the PPM monotonicity constraints (Colella & Woodward 1984, eqs. 1.10-1.11)
to the left and right face values `aL`, `aR` bracketing a cell with value `w`.

1. If `(aR - w)(w - aL) <= 0`, the cell is a local extremum: flatten to constant `w`.
2. Otherwise, apply parabola overshoot constraints:
   - If `delta_m * (w - 0.5*(aL + aR)) > delta_m^2 / 6`, set `aL = 3w - 2aR`.
   - If `-delta_m^2 / 6 > delta_m * (w - 0.5*(aL + aR))`, set `aR = 3w - 2aL`.
"""
@inline function _ppm_monotonize(w, aL, aR)
    # Check for local extremum
    if (aR - w) * (w - aL) <= zero(w)
        return w, w
    end

    delta_m = aR - aL
    curv = w - (aL + aR) / 2

    if delta_m * curv > delta_m^2 / 6
        # Overshoot on left side
        aL = 3 * w - 2 * aR
    elseif -delta_m^2 / 6 > delta_m * curv
        # Overshoot on right side
        aR = 3 * w - 2 * aL
    end

    return aL, aR
end

"""
    _ppm_interface_scalar(wLL, wL, wR, wRR) -> (wL_face, wR_face)

Scalar PPM reconstruction at the interface between cells L and R.

- Left state: reconstruct from cell L using face values at (LL|L) and (L|R),
  apply monotonicity, return right face value of cell L.
- Right state: reconstruct from cell R using face values at (L|R) and (R|RR),
  apply monotonicity, return left face value of cell R.
"""
@inline function _ppm_interface_scalar(wLL, wL, wR, wRR)
    # Face value between LL and L (left face of cell L)
    # Need value further left, but we only have 4 cells.
    # Use 2nd-order at the boundary of the stencil:
    #   a_{L,left} = (7*(wLL + wL) - (extrapolated + wR)) / 12
    # Instead, compute the two faces that bracket each cell from the available data.
    #
    # For cell L: left face uses stencil {wLL-1, wLL, wL, wR} — but wLL-1 is unavailable.
    # For cell R: right face uses stencil {wL, wR, wRR, wRR+1} — but wRR+1 is unavailable.
    #
    # Standard PPM approach for a 4-cell interface stencil:
    # - Compute the shared face value a_mid at the L|R interface
    # - Compute the left face of cell L as a_left = (wLL + wL) / 2   (2nd order)
    # - Compute the right face of cell R as a_right = (wR + wRR) / 2 (2nd order)
    # - Apply monotonicity to each cell, then extract interface values.

    # Shared face at L|R (4th-order)
    a_mid = _ppm_face_value(wL, wR, wLL, wRR)

    # Left face of cell L (2nd-order, since we lack wLL-1)
    a_L_left = (wLL + wL) / 2

    # Right face of cell R (2nd-order, since we lack wRR+1)
    a_R_right = (wR + wRR) / 2

    # Monotonize cell L with face values (a_L_left, a_mid)
    aL_left_m, aL_right_m = _ppm_monotonize(wL, a_L_left, a_mid)

    # Monotonize cell R with face values (a_mid, a_R_right)
    aR_left_m, aR_right_m = _ppm_monotonize(wR, a_mid, a_R_right)

    # Left interface state = right face of cell L
    wL_face = aL_right_m
    # Right interface state = left face of cell R
    wR_face = aR_left_m

    return wL_face, wR_face
end

# ============================================================
# Interface matching the CellCenteredMUSCL / WENO3 pattern
# ============================================================

"""
    reconstruct_interface(recon::PPMReconstruction, wLL, wL, wR, wRR) -> (wL_face, wR_face)

Reconstruct interface states at the face between cell L and cell R
using PPM reconstruction on primitive variables.

- `wLL`: value two cells to the left
- `wL`: value one cell to the left (owner)
- `wR`: value one cell to the right (neighbor)
- `wRR`: value two cells to the right

Returns `(wL_face, wR_face)` — the left and right states at the interface.
"""
@inline function reconstruct_interface(recon::PPMReconstruction, wLL, wL, wR, wRR)
    return _ppm_interface_scalar(wLL, wL, wR, wRR)
end

# SVector component-wise overload
@inline function reconstruct_interface(recon::PPMReconstruction, wLL::SVector{N}, wL::SVector{N}, wR::SVector{N}, wRR::SVector{N}) where {N}
    pairs = ntuple(Val(N)) do i
        _ppm_interface_scalar(wLL[i], wL[i], wR[i], wRR[i])
    end
    wL_face = SVector{N}(ntuple(i -> pairs[i][1], Val(N)))
    wR_face = SVector{N}(ntuple(i -> pairs[i][2], Val(N)))
    return wL_face, wR_face
end

# ============================================================
# 1D dispatch
# ============================================================

"""
    reconstruct_interface_1d(recon::PPMReconstruction, law, U, face_idx, ncells) -> (wL, wR)

Reconstruct primitive left/right states at a 1D face using PPM.
Uses the same stencil as `CellCenteredMUSCL` (4 cells: iL-1, iL, iR, iR+1).
"""
@inline function reconstruct_interface_1d(recon::PPMReconstruction, law, U::AbstractVector, face_idx::Int, ncells::Int)
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
Reconstruct interface states for an x-direction face at column boundary (iL, iR) in row jj using PPM.
"""
@inline function _reconstruct_face_2d(
        recon::PPMReconstruction, law, U::AbstractMatrix,
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
Reconstruct interface states for a y-direction face at row boundary (jL, jR) in column ii using PPM.
"""
@inline function _reconstruct_face_2d_y(
        recon::PPMReconstruction, law, U::AbstractMatrix,
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

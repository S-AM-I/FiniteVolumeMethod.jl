using StaticArrays: SVector

"""
    CellCenteredMUSCL{L <: AbstractLimiter}

MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws) reconstruction
for cell-centered data on structured meshes.

Reconstructs left and right interface states from cell-averaged values using
slope limiting from the existing `AbstractLimiter` hierarchy.

# Fields
- `limiter::L`: Slope limiter to use for reconstruction.
"""
struct CellCenteredMUSCL{L <: AbstractLimiter}
    limiter::L
end

CellCenteredMUSCL() = CellCenteredMUSCL(MinmodLimiter())

"""
    reconstruct_states(recon::CellCenteredMUSCL, uL, uC, uR) -> (u_left, u_right)

Given three consecutive cell-average values `uL`, `uC`, `uR`, reconstruct
the left and right interface states at the right face of cell C.

Returns `(uC_right, uR_left)` — the reconstructed value at the right face
from the perspective of cell C, and the reconstructed value at the same face
from the perspective of cell R.

For the left face of cell C, call with shifted data.
"""
@inline function reconstruct_states(recon::CellCenteredMUSCL, uL, uC, uR)
    # Slope in cell C
    slope_C = _limited_slope(recon.limiter, uL, uC, uR)
    # Reconstructed value at right face of cell C
    uC_right = uC + 0.5 * slope_C
    return uC_right
end

"""
    reconstruct_interface(recon::CellCenteredMUSCL, uLL, uL, uR, uRR) -> (uL_face, uR_face)

Reconstruct interface states at the face between cell L and cell R.
- `uLL`: value two cells to the left
- `uL`: value one cell to the left (owner)
- `uR`: value one cell to the right (neighbor)
- `uRR`: value two cells to the right

Returns `(uL_face, uR_face)` — the left and right states at the interface.
"""
@inline function reconstruct_interface(recon::CellCenteredMUSCL, uLL, uL, uR, uRR)
    slope_L = _limited_slope(recon.limiter, uLL, uL, uR)
    slope_R = _limited_slope(recon.limiter, uL, uR, uRR)
    uL_face = uL + 0.5 * slope_L
    uR_face = uR - 0.5 * slope_R
    return uL_face, uR_face
end

# ============================================================
# Limited slope computation using existing limiter hierarchy
# ============================================================

"""
    _limited_slope(limiter, uL, uC, uR) -> slope

Compute a limited slope for MUSCL reconstruction given three consecutive values.
Uses the existing limiter functions from `src/schemes/limiters.jl`.
"""
@inline function _limited_slope(limiter::AbstractLimiter, uL, uC, uR)
    return _limited_slope_dispatch(limiter, uL, uC, uR)
end

# For slope-ratio based limiters: MinmodLimiter, SuperbeeLimiter, VanLeerLimiter
@inline function _limited_slope_dispatch(::MinmodLimiter, uL, uC, uR)
    ΔL = uC - uL
    ΔR = uR - uC
    return minmod(ΔL, ΔR)
end

@inline function _limited_slope_dispatch(::SuperbeeLimiter, uL, uC, uR)
    ΔL = uC - uL
    ΔR = uR - uC
    return superbee(ΔL, ΔR)
end

@inline function _limited_slope_dispatch(::VanLeerLimiter, uL, uC, uR)
    ΔL = uC - uL
    ΔR = uR - uC
    return van_leer(ΔL, ΔR)
end

# For ratio-based limiters: KorenLimiter, OspreLimiter, VenkatakrishnanLimiter
@inline function _limited_slope_dispatch(limiter::KorenLimiter, uL, uC, uR)
    ΔR = uR - uC
    r = compute_slope_ratio(uL, uC, uR)
    φ = koren(r)
    return φ * ΔR
end

@inline function _limited_slope_dispatch(limiter::OspreLimiter, uL, uC, uR)
    ΔR = uR - uC
    r = compute_slope_ratio(uL, uC, uR)
    φ = ospre(r)
    return φ * ΔR
end

@inline function _limited_slope_dispatch(limiter::VenkatakrishnanLimiter, uL, uC, uR)
    ΔR = uR - uC
    r = compute_slope_ratio(uL, uC, uR)
    φ = venkatakrishnan(r)
    return φ * ΔR
end

# For SVector: apply component-wise
@inline function _limited_slope(limiter::AbstractLimiter, uL::SVector{N}, uC::SVector{N}, uR::SVector{N}) where {N}
    return SVector{N}(ntuple(i -> _limited_slope_dispatch(limiter, uL[i], uC[i], uR[i]), Val(N)))
end

# No reconstruction (first-order)
"""
    NoReconstruction

First-order (piecewise constant) scheme with no reconstruction.
"""
struct NoReconstruction end

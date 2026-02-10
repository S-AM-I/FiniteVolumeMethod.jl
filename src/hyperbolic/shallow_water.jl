# ============================================================
# Shallow Water Equations
# ============================================================
#
# Conservation law implementation for the shallow water equations
# in 1D and 2D. Models free-surface gravity waves over variable
# bottom topography.
#
# 1D: U = [h, hu],  F = [hu, hu^2 + 1/2 g h^2]
# 2D: U = [h, hu, hv],
#     Fx = [hu, hu^2 + 1/2 g h^2, huv]
#     Fy = [hv, huv, hv^2 + 1/2 g h^2]

"""
    ShallowWaterEquations{Dim} <: AbstractConservationLaw{Dim}

The shallow water equations in `Dim` spatial dimensions.

## 1D (Dim=1)
Conserved variables: `U = [h, hu]`
Primitive variables: `W = [h, u]`
Flux: `F = [hu, hu^2 + 1/2 g h^2]`

## 2D (Dim=2)
Conserved variables: `U = [h, hu, hv]`
Primitive variables: `W = [h, u, v]`
Fluxes:
  `Fx = [hu, hu^2 + 1/2 g h^2, huv]`
  `Fy = [hv, huv, hv^2 + 1/2 g h^2]`

# Fields
- `g::Float64`: Gravitational acceleration (default 9.81).
"""
struct ShallowWaterEquations{Dim} <: AbstractConservationLaw{Dim}
    g::Float64
end

ShallowWaterEquations{Dim}(; g = 9.81) where {Dim} = ShallowWaterEquations{Dim}(g)

nvariables(::ShallowWaterEquations{1}) = 2
nvariables(::ShallowWaterEquations{2}) = 3

# ============================================================
# 1D Shallow Water
# ============================================================

"""
    conserved_to_primitive(law::ShallowWaterEquations{1}, u::SVector{2}) -> SVector{2}

Convert 1D conserved `[h, hu]` to primitive `[h, u]`.
"""
@inline function conserved_to_primitive(law::ShallowWaterEquations{1}, u::SVector{2})
    h = u[1]
    v = u[2] / h
    return SVector(h, v)
end

"""
    primitive_to_conserved(law::ShallowWaterEquations{1}, w::SVector{2}) -> SVector{2}

Convert 1D primitive `[h, u]` to conserved `[h, hu]`.
"""
@inline function primitive_to_conserved(law::ShallowWaterEquations{1}, w::SVector{2})
    h, v = w
    return SVector(h, h * v)
end

"""
    physical_flux(law::ShallowWaterEquations{1}, w::SVector{2}, ::Int) -> SVector{2}

Compute the 1D shallow water flux from primitive variables `[h, u]`.
"""
@inline function physical_flux(law::ShallowWaterEquations{1}, w::SVector{2}, ::Int)
    h, v = w
    return SVector(h * v, h * v^2 + 0.5 * law.g * h^2)
end

"""
    max_wave_speed(law::ShallowWaterEquations{1}, w::SVector{2}, ::Int) -> Real

Maximum wave speed `|u| + sqrt(gh)` from primitive variables.
"""
@inline function max_wave_speed(law::ShallowWaterEquations{1}, w::SVector{2}, ::Int)
    h, v = w
    c = sqrt(law.g * h)
    return abs(v) + c
end

"""
    wave_speeds(law::ShallowWaterEquations{1}, w::SVector{2}, ::Int) -> (lambda_min, lambda_max)

Return the minimum and maximum wave speeds from primitive variables.
"""
@inline function wave_speeds(law::ShallowWaterEquations{1}, w::SVector{2}, ::Int)
    h, v = w
    c = sqrt(law.g * h)
    return v - c, v + c
end

# ============================================================
# 2D Shallow Water
# ============================================================

"""
    conserved_to_primitive(law::ShallowWaterEquations{2}, u::SVector{3}) -> SVector{3}

Convert 2D conserved `[h, hu, hv]` to primitive `[h, u, v]`.
"""
@inline function conserved_to_primitive(law::ShallowWaterEquations{2}, u::SVector{3})
    h = u[1]
    vx = u[2] / h
    vy = u[3] / h
    return SVector(h, vx, vy)
end

"""
    primitive_to_conserved(law::ShallowWaterEquations{2}, w::SVector{3}) -> SVector{3}

Convert 2D primitive `[h, u, v]` to conserved `[h, hu, hv]`.
"""
@inline function primitive_to_conserved(law::ShallowWaterEquations{2}, w::SVector{3})
    h, vx, vy = w
    return SVector(h, h * vx, h * vy)
end

"""
    physical_flux(law::ShallowWaterEquations{2}, w::SVector{3}, dir::Int) -> SVector{3}

Compute the 2D shallow water flux in direction `dir` (1=x, 2=y) from primitive variables `[h, u, v]`.
"""
@inline function physical_flux(law::ShallowWaterEquations{2}, w::SVector{3}, dir::Int)
    h, vx, vy = w
    if dir == 1
        return SVector(h * vx, h * vx^2 + 0.5 * law.g * h^2, h * vx * vy)
    else
        return SVector(h * vy, h * vx * vy, h * vy^2 + 0.5 * law.g * h^2)
    end
end

"""
    max_wave_speed(law::ShallowWaterEquations{2}, w::SVector{3}, dir::Int) -> Real

Maximum wave speed in direction `dir` from primitive variables.
"""
@inline function max_wave_speed(law::ShallowWaterEquations{2}, w::SVector{3}, dir::Int)
    h, vx, vy = w
    c = sqrt(law.g * h)
    v_n = dir == 1 ? vx : vy
    return abs(v_n) + c
end

"""
    wave_speeds(law::ShallowWaterEquations{2}, w::SVector{3}, dir::Int) -> (lambda_min, lambda_max)

Return the minimum and maximum wave speeds in direction `dir` from primitive variables.
"""
@inline function wave_speeds(law::ShallowWaterEquations{2}, w::SVector{3}, dir::Int)
    h, vx, vy = w
    c = sqrt(law.g * h)
    v_n = dir == 1 ? vx : vy
    return v_n - c, v_n + c
end

# ============================================================
# HLLC Riemann Solver — 1D Shallow Water
# ============================================================

"""
    _hllc_star_state_swe_1d(h, v, S_K, S_star) -> SVector{2}

Compute the HLLC star-region conserved state for 1D shallow water.
"""
@inline function _hllc_star_state_swe_1d(h, v, S_K, S_star)
    h_star = h * (S_K - v) / (S_K - S_star)
    return SVector(h_star, h_star * S_star)
end

function solve_riemann(::HLLCSolver, law::ShallowWaterEquations{1}, wL::SVector{2}, wR::SVector{2}, dir::Int)
    hL, vL = wL
    hR, vR = wR
    g = law.g

    # Conserved states
    uL = primitive_to_conserved(law, wL)
    uR = primitive_to_conserved(law, wR)

    # Gravity wave speeds
    cL = sqrt(g * hL)
    cR = sqrt(g * hR)

    # Wave speed estimates (two-rarefaction approximation)
    h_star_est = 0.5 * (hL + hR) - 0.25 * (vR - vL) * (hL + hR) / (cL + cR)
    h_star_est = max(h_star_est, zero(h_star_est))

    SL = vL - cL * (h_star_est > hL ? sqrt(0.5 * h_star_est * (h_star_est + hL)) / hL : one(hL))
    SR = vR + cR * (h_star_est > hR ? sqrt(0.5 * h_star_est * (h_star_est + hR)) / hR : one(hR))

    # Contact wave speed
    S_star = (SL * hR * (vR - SR) - SR * hL * (vL - SL)) /
        (hR * (vR - SR) - hL * (vL - SL))

    if SL >= zero(SL)
        # Left of all waves
        return physical_flux(law, wL, dir)
    elseif SR <= zero(SR)
        # Right of all waves
        return physical_flux(law, wR, dir)
    elseif S_star >= zero(S_star)
        # Star-left region
        fL = physical_flux(law, wL, dir)
        u_star_L = _hllc_star_state_swe_1d(hL, vL, SL, S_star)
        return fL + SL * (u_star_L - uL)
    else
        # Star-right region
        fR = physical_flux(law, wR, dir)
        u_star_R = _hllc_star_state_swe_1d(hR, vR, SR, S_star)
        return fR + SR * (u_star_R - uR)
    end
end

# ============================================================
# HLLC Riemann Solver — 2D Shallow Water
# ============================================================

"""
    _hllc_star_state_swe_2d(h, vn, vt, S_K, S_star, dir) -> SVector{3}

Compute the HLLC star-region conserved state for 2D shallow water.
The tangential velocity is preserved across the contact, only the
normal velocity jumps to `S_star`.
"""
@inline function _hllc_star_state_swe_2d(h, vn, vt, S_K, S_star, dir)
    h_star = h * (S_K - vn) / (S_K - S_star)
    if dir == 1
        return SVector(
            h_star,
            h_star * S_star,   # h* u* (normal is x)
            h_star * vt        # h* v* (tangential preserved)
        )
    else
        return SVector(
            h_star,
            h_star * vt,       # h* u* (tangential preserved)
            h_star * S_star    # h* v* (normal is y)
        )
    end
end

function solve_riemann(::HLLCSolver, law::ShallowWaterEquations{2}, wL::SVector{3}, wR::SVector{3}, dir::Int)
    hL, vxL, vyL = wL
    hR, vxR, vyR = wR
    g = law.g

    # Normal and tangential velocities
    if dir == 1
        vnL, vtL = vxL, vyL
        vnR, vtR = vxR, vyR
    else
        vnL, vtL = vyL, vxL
        vnR, vtR = vyR, vxR
    end

    # Conserved states
    uL = primitive_to_conserved(law, wL)
    uR = primitive_to_conserved(law, wR)

    # Gravity wave speeds
    cL = sqrt(g * hL)
    cR = sqrt(g * hR)

    # Wave speed estimates (two-rarefaction approximation)
    h_star_est = 0.5 * (hL + hR) - 0.25 * (vnR - vnL) * (hL + hR) / (cL + cR)
    h_star_est = max(h_star_est, zero(h_star_est))

    SL = vnL - cL * (h_star_est > hL ? sqrt(0.5 * h_star_est * (h_star_est + hL)) / hL : one(hL))
    SR = vnR + cR * (h_star_est > hR ? sqrt(0.5 * h_star_est * (h_star_est + hR)) / hR : one(hR))

    # Contact wave speed
    S_star = (SL * hR * (vnR - SR) - SR * hL * (vnL - SL)) /
        (hR * (vnR - SR) - hL * (vnL - SL))

    if SL >= zero(SL)
        return physical_flux(law, wL, dir)
    elseif SR <= zero(SR)
        return physical_flux(law, wR, dir)
    elseif S_star >= zero(S_star)
        # Star-left region
        fL = physical_flux(law, wL, dir)
        u_star_L = _hllc_star_state_swe_2d(hL, vnL, vtL, SL, S_star, dir)
        return fL + SL * (u_star_L - uL)
    else
        # Star-right region
        fR = physical_flux(law, wR, dir)
        u_star_R = _hllc_star_state_swe_2d(hR, vnR, vtR, SR, S_star, dir)
        return fR + SR * (u_star_R - uR)
    end
end

# ============================================================
# Bottom Topography Source Term
# ============================================================

"""
    BottomTopography{F}

Represents the bottom elevation function for shallow water equations.

For 1D problems, `b` is a function `b(x)` returning the bottom elevation.
For 2D problems, `b` is a function `b(x, y)` returning the bottom elevation.

# Fields
- `b::F`: Bottom elevation function.
"""
struct BottomTopography{F}
    b::F
end

"""
    topography_source_1d(law::ShallowWaterEquations, h, b_L, b_R, dx) -> SVector{2}

Compute the well-balanced topography source term for 1D shallow water.

Uses a simple centred difference approximation of the bed slope:
  `S = [0, -g h (b_R - b_L) / dx]`

This source balances the hydrostatic flux gradient at rest (lake-at-rest),
providing a well-balanced discretisation when combined with appropriate
flux differencing.

# Arguments
- `law`: The shallow water conservation law (provides `g`).
- `h`: Water depth at the cell centre.
- `b_L`: Bottom elevation at the left cell interface.
- `b_R`: Bottom elevation at the right cell interface.
- `dx`: Cell width.

# Returns
Source vector `SVector{2}`.
"""
@inline function topography_source_1d(law::ShallowWaterEquations, h, b_L, b_R, dx)
    return SVector(zero(h), -law.g * h * (b_R - b_L) / dx)
end

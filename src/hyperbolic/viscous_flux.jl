# ============================================================
# Viscous flux computation for Navier-Stokes equations
# ============================================================

"""
    viscous_flux_1d(ns::NavierStokesEquations{1}, wL::SVector{3}, wR::SVector{3}, dx) -> SVector{3}

Compute the viscous flux at a 1D face between left and right primitive states.

Stress tensor: `τ_xx = (4/3) μ ∂v/∂x`
Heat flux: `q_x = -κ ∂T/∂x` where `T = P/ρ` (non-dimensional temperature)

Returns `SVector(0, τ_xx, v_face * τ_xx - q_x)`.
"""
@inline function viscous_flux_1d(ns::NavierStokesEquations{1}, wL::SVector{3}, wR::SVector{3}, dx)
    ρL, vL, PL = wL
    ρR, vR, PR = wR
    μ = ns.mu
    κ = thermal_conductivity(ns)

    # Gradients via central difference
    dv_dx = (vR - vL) / dx

    # Temperature T = P/ρ (non-dimensional)
    TL = PL / ρL
    TR = PR / ρR
    dT_dx = (TR - TL) / dx

    # Stress
    τ_xx = (4.0 / 3.0) * μ * dv_dx

    # Heat flux
    q_x = -κ * dT_dx

    # Face velocity
    v_face = 0.5 * (vL + vR)

    return SVector(zero(τ_xx), τ_xx, v_face * τ_xx - q_x)
end

"""
    viscous_flux_x_2d(ns::NavierStokesEquations{2}, wL::SVector{4}, wR::SVector{4},
                      dvx_dy, dvy_dy, dx) -> SVector{4}

Compute the viscous flux at a 2D x-direction face.

Normal gradients are computed from the direct neighbors;
cross-derivatives `∂vx/∂y` and `∂vy/∂y` must be provided (4-cell average).

Returns `SVector(0, τ_xx, τ_xy, vx*τ_xx + vy*τ_xy - q_x)`.
"""
@inline function viscous_flux_x_2d(
        ns::NavierStokesEquations{2}, wL::SVector{4}, wR::SVector{4},
        dvx_dy, dvy_dy, dx
    )
    ρL, vxL, vyL, PL = wL
    ρR, vxR, vyR, PR = wR
    μ = ns.mu
    κ = thermal_conductivity(ns)

    # Normal gradients
    dvx_dx = (vxR - vxL) / dx
    dvy_dx = (vyR - vyL) / dx

    # Temperature
    TL = PL / ρL
    TR = PR / ρR
    dT_dx = (TR - TL) / dx

    # Stress components
    τ_xx = μ * ((4.0 / 3.0) * dvx_dx - (2.0 / 3.0) * dvy_dy)
    τ_xy = μ * (dvx_dy + dvy_dx)

    # Heat flux
    q_x = -κ * dT_dx

    # Face velocities
    vx_face = 0.5 * (vxL + vxR)
    vy_face = 0.5 * (vyL + vyR)

    return SVector(zero(τ_xx), τ_xx, τ_xy, vx_face * τ_xx + vy_face * τ_xy - q_x)
end

"""
    viscous_flux_y_2d(ns::NavierStokesEquations{2}, wB::SVector{4}, wT::SVector{4},
                      dvx_dx, dvy_dx, dy) -> SVector{4}

Compute the viscous flux at a 2D y-direction face.

Normal gradients are computed from the direct neighbors;
cross-derivatives `∂vx/∂x` and `∂vy/∂x` must be provided (4-cell average).

Returns `SVector(0, τ_yx, τ_yy, vx*τ_yx + vy*τ_yy - q_y)`.
"""
@inline function viscous_flux_y_2d(
        ns::NavierStokesEquations{2}, wB::SVector{4}, wT::SVector{4},
        dvx_dx, dvy_dx, dy
    )
    ρB, vxB, vyB, PB = wB
    ρT, vxT, vyT, PT = wT
    μ = ns.mu
    κ = thermal_conductivity(ns)

    # Normal gradients
    dvx_dy = (vxT - vxB) / dy
    dvy_dy = (vyT - vyB) / dy

    # Temperature
    TB = PB / ρB
    TT = PT / ρT
    dT_dy = (TT - TB) / dy

    # Stress components
    τ_yy = μ * ((4.0 / 3.0) * dvy_dy - (2.0 / 3.0) * dvx_dx)
    τ_yx = μ * (dvx_dy + dvy_dx)

    # Heat flux
    q_y = -κ * dT_dy

    # Face velocities
    vx_face = 0.5 * (vxB + vxT)
    vy_face = 0.5 * (vyB + vyT)

    return SVector(zero(τ_yy), τ_yx, τ_yy, vx_face * τ_yx + vy_face * τ_yy - q_y)
end

# ============================================================
# Kerr Metric in Kerr-Schild Coordinates
# ============================================================
#
# The Kerr black hole metric for a spinning BH with mass M and
# spin parameter a, in horizon-penetrating Kerr-Schild coordinates.
#
# We work in the equatorial plane (theta = pi/2, z = 0) where
# the Boyer-Lindquist-like radius r satisfies:
#   x^2 + y^2 = r^2 + a^2
#
# In Kerr-Schild form, the 4-metric is:
#   g_mu_nu = eta_mu_nu + 2H * l_mu * l_nu
#
# with H = Mr / Sigma, and l^mu is the principal null congruence.
#
# In the equatorial plane (Sigma = r^2):
#   H = M / r
#
# The Kerr-Schild null vector in Cartesian coords (equatorial):
#   l^x = (r*x + a*y) / (r^2 + a^2)
#   l^y = (r*y - a*x) / (r^2 + a^2)
#   l^t = 1
#
# 3+1 decomposition:
#   alpha = 1 / sqrt(1 + 2H)
#   beta^i = 2H / (1 + 2H) * l^i
#   gamma_ij = delta_ij + 2H * l_i * l_j
#   gamma^ij = delta_ij - 2H / (1 + 2H) * l^i * l^j
#   sqrt(gamma) = sqrt(1 + 2H)
#
# Reference: Kerr (1963), Font et al. (2000), McKinney & Gammie (2004).

"""
    KerrMetric{FT} <: AbstractMetric{2}

Kerr black hole spacetime in Kerr-Schild (horizon-penetrating) coordinates,
restricted to the equatorial plane.

# Fields
- `M::FT`: Black hole mass.
- `a::FT`: Spin parameter (dimensionless, |a| <= M).
- `r_min::FT`: Minimum radius floor.
"""
struct KerrMetric{FT} <: AbstractMetric{2}
    M::FT
    a::FT
    r_min::FT
end

function KerrMetric(M::FT, a::FT; r_min::FT = FT(0.1) * M) where {FT}
    @assert abs(a) <= M "Spin parameter |a| must be <= M"
    return KerrMetric{FT}(M, a, r_min)
end

"""
    _kerr_r(m::KerrMetric, x, y) -> r

Compute the Boyer-Lindquist-like radius in the equatorial plane from
x^2 + y^2 = r^2 + a^2, i.e., r = sqrt(x^2 + y^2 - a^2), clamped to r_min.
"""
@inline function _kerr_r(m::KerrMetric, x, y)
    R_sq = x^2 + y^2
    r_sq = R_sq - m.a^2
    r = sqrt(max(r_sq, m.r_min^2))
    return max(r, m.r_min)
end

"""
    _kerr_H_and_l(m::KerrMetric, x, y) -> (H, lx, ly)

Compute the Kerr-Schild scalar H = M/r and the spatial components
of the null vector l^i in Cartesian coordinates (equatorial plane).
"""
@inline function _kerr_H_and_l(m::KerrMetric, x, y)
    r = _kerr_r(m, x, y)
    a = m.a
    H = m.M / r

    # Kerr-Schild null vector spatial components in Cartesian (equatorial)
    denom = r^2 + a^2
    lx = (r * x + a * y) / denom
    ly = (r * y - a * x) / denom

    return H, lx, ly
end

@inline function lapse(m::KerrMetric, x, y)
    H, _, _ = _kerr_H_and_l(m, x, y)
    return 1 / sqrt(1 + 2 * H)
end

@inline function shift(m::KerrMetric, x, y)
    H, lx, ly = _kerr_H_and_l(m, x, y)
    fac = 2 * H / (1 + 2 * H)
    return SVector(fac * lx, fac * ly)
end

@inline function spatial_metric(m::KerrMetric, x, y)
    H, lx, ly = _kerr_H_and_l(m, x, y)
    f = 2 * H
    gxx = 1 + f * lx * lx
    gxy = f * lx * ly
    gyy = 1 + f * ly * ly
    return StaticArrays.SMatrix{2, 2}(gxx, gxy, gxy, gyy)
end

@inline function sqrt_gamma(m::KerrMetric, x, y)
    H, _, _ = _kerr_H_and_l(m, x, y)
    return sqrt(1 + 2 * H)
end

@inline function inv_spatial_metric(m::KerrMetric, x, y)
    H, lx, ly = _kerr_H_and_l(m, x, y)
    fac = 2 * H / (1 + 2 * H)
    gixx = 1 - fac * lx * lx
    gixy = -fac * lx * ly
    giyy = 1 - fac * ly * ly
    return StaticArrays.SMatrix{2, 2}(gixx, gixy, gixy, giyy)
end

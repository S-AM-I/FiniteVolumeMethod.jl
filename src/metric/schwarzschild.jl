# ============================================================
# Schwarzschild Metric in Kerr-Schild Coordinates
# ============================================================
#
# The Schwarzschild black hole metric in horizon-penetrating
# Kerr-Schild coordinates (a.k.a. Eddington-Finkelstein-like).
#
# In these coordinates the metric is regular at the horizon,
# making them suitable for numerical simulations that cross
# or approach r = 2M.
#
# The 3+1 decomposition in Kerr-Schild form (Cartesian):
#
#   H = M / r,  l^i = x^i / r  (radial unit vector)
#
#   alpha = 1 / sqrt(1 + 2H)
#   beta^i = 2H / (1 + 2H) * l^i
#   gamma_ij = delta_ij + 2H * l_i * l_j
#   gamma^ij = delta_ij - 2H / (1 + 2H) * l^i * l^j
#   sqrt(gamma) = sqrt(1 + 2H)
#
# Reference: Font et al. (2000), Baumgarte & Shapiro (2010).

"""
    SchwarzschildMetric{FT} <: AbstractMetric{2}

Schwarzschild black hole spacetime in Kerr-Schild (horizon-penetrating) coordinates.

The metric is parameterized by the black hole mass `M`. Coordinates are Cartesian
`(x, y)` with `r = sqrt(x^2 + y^2)`.

A floor radius `r_min` prevents singularities at `r = 0` in numerical simulations.

# Fields
- `M::FT`: Black hole mass.
- `r_min::FT`: Minimum radius floor (default `0.1 * M`).
"""
struct SchwarzschildMetric{FT} <: AbstractMetric{2}
    M::FT
    r_min::FT
end

function SchwarzschildMetric(M::FT; r_min::FT = FT(0.1) * M) where {FT}
    return SchwarzschildMetric{FT}(M, r_min)
end

"""
    _ks_radius(m::SchwarzschildMetric, x, y) -> r

Compute the clamped coordinate radius r = max(sqrt(x^2 + y^2), r_min).
"""
@inline function _ks_radius(m::SchwarzschildMetric, x, y)
    r = sqrt(x^2 + y^2)
    return max(r, m.r_min)
end

"""
    _ks_H_and_l(m::SchwarzschildMetric, x, y) -> (H, lx, ly)

Compute the Kerr-Schild scalar H = M/r and the radial unit vector l^i = x^i/r.
"""
@inline function _ks_H_and_l(m::SchwarzschildMetric, x, y)
    r = _ks_radius(m, x, y)
    H = m.M / r
    lx = x / r
    ly = y / r
    return H, lx, ly
end

@inline function lapse(m::SchwarzschildMetric, x, y)
    H, _, _ = _ks_H_and_l(m, x, y)
    return 1 / sqrt(1 + 2 * H)
end

@inline function shift(m::SchwarzschildMetric, x, y)
    H, lx, ly = _ks_H_and_l(m, x, y)
    fac = 2 * H / (1 + 2 * H)
    return SVector(fac * lx, fac * ly)
end

@inline function spatial_metric(m::SchwarzschildMetric, x, y)
    H, lx, ly = _ks_H_and_l(m, x, y)
    f = 2 * H
    gxx = 1 + f * lx * lx
    gxy = f * lx * ly
    gyy = 1 + f * ly * ly
    return StaticArrays.SMatrix{2, 2}(gxx, gxy, gxy, gyy)
end

@inline function sqrt_gamma(m::SchwarzschildMetric, x, y)
    H, _, _ = _ks_H_and_l(m, x, y)
    return sqrt(1 + 2 * H)
end

@inline function inv_spatial_metric(m::SchwarzschildMetric, x, y)
    H, lx, ly = _ks_H_and_l(m, x, y)
    fac = 2 * H / (1 + 2 * H)
    gixx = 1 - fac * lx * lx
    gixy = -fac * lx * ly
    giyy = 1 - fac * ly * ly
    return StaticArrays.SMatrix{2, 2}(gixx, gixy, gixy, giyy)
end

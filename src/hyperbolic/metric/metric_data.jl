# ============================================================
# Precomputed Metric Data for 2D GRMHD
# ============================================================
#
# Evaluating transcendental functions (sqrt, division) for the
# metric at every cell, every RK stage, every time step is
# expensive. Instead, we precompute and cache all metric
# quantities at cell centers once before the time integration.
#
# This is valid for stationary spacetimes (Kerr, Schwarzschild,
# Minkowski) where the metric does not depend on time.

"""
    MetricData2D{FT}

Precomputed metric quantities at cell centers for a 2D structured mesh.

All arrays are `nx x ny` and correspond to interior cells `(ix, iy)`
for `ix = 1:nx, iy = 1:ny`.

# Fields
- `alpha::Matrix{FT}`: Lapse function.
- `beta_x::Matrix{FT}`: x-component of shift vector.
- `beta_y::Matrix{FT}`: y-component of shift vector.
- `gamma_xx::Matrix{FT}`: (1,1) component of spatial metric.
- `gamma_xy::Matrix{FT}`: (1,2) component of spatial metric.
- `gamma_yy::Matrix{FT}`: (2,2) component of spatial metric.
- `gammaI_xx::Matrix{FT}`: (1,1) component of inverse spatial metric.
- `gammaI_xy::Matrix{FT}`: (1,2) component of inverse spatial metric.
- `gammaI_yy::Matrix{FT}`: (2,2) component of inverse spatial metric.
- `sqrtg::Matrix{FT}`: Square root of spatial metric determinant.
"""
struct MetricData2D{FT}
    alpha::Matrix{FT}
    beta_x::Matrix{FT}
    beta_y::Matrix{FT}
    gamma_xx::Matrix{FT}
    gamma_xy::Matrix{FT}
    gamma_yy::Matrix{FT}
    gammaI_xx::Matrix{FT}
    gammaI_xy::Matrix{FT}
    gammaI_yy::Matrix{FT}
    sqrtg::Matrix{FT}
end

"""
    MetricData2D(nx::Int, ny::Int, ::Type{FT}=Float64) -> MetricData2D{FT}

Allocate zero-initialized metric data for an `nx x ny` mesh.
"""
function MetricData2D(nx::Int, ny::Int, ::Type{FT} = Float64) where {FT}
    return MetricData2D{FT}(
        zeros(FT, nx, ny),
        zeros(FT, nx, ny),
        zeros(FT, nx, ny),
        zeros(FT, nx, ny),
        zeros(FT, nx, ny),
        zeros(FT, nx, ny),
        zeros(FT, nx, ny),
        zeros(FT, nx, ny),
        zeros(FT, nx, ny),
        zeros(FT, nx, ny)
    )
end

"""
    precompute_metric(metric::AbstractMetric{2}, mesh::StructuredMesh2D) -> MetricData2D

Precompute all metric quantities at cell centers for a 2D structured mesh.

This should be called once before the time integration loop for stationary spacetimes.
"""
function precompute_metric(metric::AbstractMetric{2}, mesh::StructuredMesh2D)
    nx, ny = mesh.nx, mesh.ny
    x0, _ = cell_center(mesh, 1)
    FT = typeof(x0)

    md = MetricData2D(nx, ny, FT)

    for iy in 1:ny, ix in 1:nx
        x, y = cell_center(mesh, cell_idx(mesh, ix, iy))

        md.alpha[ix, iy] = lapse(metric, x, y)

        beta = shift(metric, x, y)
        md.beta_x[ix, iy] = beta[1]
        md.beta_y[ix, iy] = beta[2]

        g = spatial_metric(metric, x, y)
        md.gamma_xx[ix, iy] = g[1, 1]
        md.gamma_xy[ix, iy] = g[1, 2]
        md.gamma_yy[ix, iy] = g[2, 2]

        gi = inv_spatial_metric(metric, x, y)
        md.gammaI_xx[ix, iy] = gi[1, 1]
        md.gammaI_xy[ix, iy] = gi[1, 2]
        md.gammaI_yy[ix, iy] = gi[2, 2]

        md.sqrtg[ix, iy] = sqrt_gamma(metric, x, y)
    end

    return md
end

"""
    precompute_metric_at_faces(metric::AbstractMetric{2}, mesh::StructuredMesh2D)
        -> (alpha_xf, alpha_yf, betax_xf, betay_xf, betax_yf, betay_yf, sqrtg_xf, sqrtg_yf)

Precompute metric quantities at face centers for the flux correction.

Returns arrays at x-faces (size `(nx+1) x ny`) and y-faces (size `nx x (ny+1)`).
"""
function precompute_metric_at_faces(metric::AbstractMetric{2}, mesh::StructuredMesh2D)
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    x0, _ = cell_center(mesh, 1)
    FT = typeof(x0)

    # x-face arrays: (nx+1) x ny
    alpha_xf = Matrix{FT}(undef, nx + 1, ny)
    betax_xf = Matrix{FT}(undef, nx + 1, ny)
    betay_xf = Matrix{FT}(undef, nx + 1, ny)
    sqrtg_xf = Matrix{FT}(undef, nx + 1, ny)

    for j in 1:ny, i in 1:(nx + 1)
        xf = mesh.xmin + (i - 1) * dx
        yf = mesh.ymin + (j - FT(0.5)) * dy
        alpha_xf[i, j] = lapse(metric, xf, yf)
        b = shift(metric, xf, yf)
        betax_xf[i, j] = b[1]
        betay_xf[i, j] = b[2]
        sqrtg_xf[i, j] = sqrt_gamma(metric, xf, yf)
    end

    # y-face arrays: nx x (ny+1)
    alpha_yf = Matrix{FT}(undef, nx, ny + 1)
    betax_yf = Matrix{FT}(undef, nx, ny + 1)
    betay_yf = Matrix{FT}(undef, nx, ny + 1)
    sqrtg_yf = Matrix{FT}(undef, nx, ny + 1)

    for j in 1:(ny + 1), i in 1:nx
        xf = mesh.xmin + (i - FT(0.5)) * dx
        yf = mesh.ymin + (j - 1) * dy
        alpha_yf[i, j] = lapse(metric, xf, yf)
        b = shift(metric, xf, yf)
        betax_yf[i, j] = b[1]
        betay_yf[i, j] = b[2]
        sqrtg_yf[i, j] = sqrt_gamma(metric, xf, yf)
    end

    return (
        alpha_xf, alpha_yf, betax_xf, betay_xf, betax_yf, betay_yf,
        sqrtg_xf, sqrtg_yf,
    )
end

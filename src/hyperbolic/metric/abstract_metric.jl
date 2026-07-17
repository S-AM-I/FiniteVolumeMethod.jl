# ============================================================
# Abstract Metric Interface for General Relativistic MHD
# ============================================================
#
# Defines the interface for spacetime metrics used in GRMHD
# simulations. All metrics are characterized by the 3+1
# decomposition of the spacetime line element:
#
#   ds^2 = -alpha^2 dt^2 + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt)
#
# where:
#   alpha   = lapse function (relates coordinate time to proper time)
#   beta^i  = shift vector (relates spatial coordinates between slices)
#   gamma_ij = spatial metric on the hypersurface
#
# The determinant sqrt(gamma) is needed for densitizing conserved
# variables in the Valencia formulation.

"""
    AbstractMetric{Dim}

Abstract supertype for spacetime metrics in `Dim` spatial dimensions.

All subtypes must implement:
- `lapse(m, x, y)`: Lapse function alpha at position `(x, y)`.
- `shift(m, x, y)`: Shift vector `SVector{Dim}` at position `(x, y)`.
- `spatial_metric(m, x, y)`: Spatial metric tensor `SMatrix{Dim,Dim}`.
- `sqrt_gamma(m, x, y)`: Square root of the determinant of the spatial metric.
- `inv_spatial_metric(m, x, y)`: Inverse spatial metric tensor `SMatrix{Dim,Dim}`.
"""
abstract type AbstractMetric{Dim} end

"""
    lapse(m::AbstractMetric, x, y) -> Real

Compute the lapse function alpha at position `(x, y)`.
The lapse measures the rate of proper time flow relative to coordinate time.
"""
function lapse end

"""
    shift(m::AbstractMetric{Dim}, x, y) -> SVector{Dim}

Compute the shift vector beta^i at position `(x, y)`.
The shift relates spatial coordinates between neighboring time slices.
"""
function shift end

"""
    spatial_metric(m::AbstractMetric{Dim}, x, y) -> SMatrix{Dim,Dim}

Compute the spatial metric tensor gamma_ij at position `(x, y)`.
This is the induced metric on the spatial hypersurface.
"""
function spatial_metric end

"""
    sqrt_gamma(m::AbstractMetric, x, y) -> Real

Compute sqrt(det(gamma_ij)) at position `(x, y)`.
This factor is used to densitize conserved variables in the Valencia formulation.
"""
function sqrt_gamma end

"""
    inv_spatial_metric(m::AbstractMetric{Dim}, x, y) -> SMatrix{Dim,Dim}

Compute the inverse spatial metric tensor gamma^ij at position `(x, y)`.
Satisfies gamma^ik gamma_kj = delta^i_j.
"""
function inv_spatial_metric end

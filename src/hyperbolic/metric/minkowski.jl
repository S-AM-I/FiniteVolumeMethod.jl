# ============================================================
# Minkowski Metric (Flat Spacetime)
# ============================================================
#
# The Minkowski metric represents flat spacetime in Cartesian
# coordinates. In 3+1 form:
#   alpha = 1, beta^i = 0, gamma_ij = delta_ij
#
# This metric should reduce GRMHD to SRMHD exactly, serving
# as both a sanity check and the trivial limit of GR.

"""
    MinkowskiMetric{Dim} <: AbstractMetric{Dim}

Flat Minkowski spacetime in `Dim` spatial dimensions.

Lapse alpha = 1, shift beta = 0, spatial metric gamma_ij = delta_ij.
Using this metric in GRMHDEquations should recover SRMHDEquations exactly.
"""
struct MinkowskiMetric{Dim} <: AbstractMetric{Dim} end

"""
    MinkowskiMetric(dim::Int) -> MinkowskiMetric{dim}

Construct a Minkowski metric in `dim` spatial dimensions.
"""
MinkowskiMetric(dim::Int) = MinkowskiMetric{dim}()

@inline lapse(::MinkowskiMetric, x, y) = one(typeof(x))

@inline function shift(::MinkowskiMetric{2}, x, y)
    T = typeof(x)
    return SVector{2, T}(zero(T), zero(T))
end

@inline function spatial_metric(::MinkowskiMetric{2}, x, y)
    T = typeof(x)
    return StaticArrays.SMatrix{2, 2, T}(one(T), zero(T), zero(T), one(T))
end

@inline sqrt_gamma(::MinkowskiMetric, x, y) = one(typeof(x))

@inline function inv_spatial_metric(::MinkowskiMetric{2}, x, y)
    T = typeof(x)
    return StaticArrays.SMatrix{2, 2, T}(one(T), zero(T), zero(T), one(T))
end

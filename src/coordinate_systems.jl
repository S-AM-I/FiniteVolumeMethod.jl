# Coordinate system types for non-Cartesian FVM
abstract type AbstractCoordinateSystem end
struct Cartesian <: AbstractCoordinateSystem end
struct Cylindrical <: AbstractCoordinateSystem end   # Axisymmetric (r,z) - x=r, y=z
struct Spherical <: AbstractCoordinateSystem end     # Radially symmetric (r,θ) - x=r, y=θ

# Geometric weight for volume integration
geometric_volume_weight(::Cartesian, x, y) = 1.0
geometric_volume_weight(::Cylindrical, r, z) = r     # r is the x-coordinate
geometric_volume_weight(::Spherical, r, θ) = r^2 * sin(θ)

# Geometric weight for flux integration (face weighting)
geometric_flux_weight(::Cartesian, x, y) = 1.0
geometric_flux_weight(::Cylindrical, r, z) = r       # r at the face midpoint
geometric_flux_weight(::Spherical, r, θ) = r^2 * sin(θ)

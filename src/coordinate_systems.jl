"""Abstract supertype for coordinate system representations (Cartesian, cylindrical, spherical)."""
abstract type AbstractCoordinateSystem end

"""Cartesian coordinate system (default, unit geometric weights)."""
struct Cartesian <: AbstractCoordinateSystem end

"""Cylindrical (axisymmetric) coordinate system where `x = r`, `y = z`."""
struct Cylindrical <: AbstractCoordinateSystem end

"""Spherical coordinate system where `x = r`, `y = θ`."""
struct Spherical <: AbstractCoordinateSystem end

"""Return the geometric volume integration weight for the given coordinate system."""
geometric_volume_weight(::Cartesian, x, y) = 1.0
geometric_volume_weight(::Cylindrical, r, z) = r     # r is the x-coordinate
geometric_volume_weight(::Spherical, r, θ) = r^2 * sin(θ)

"""Return the geometric flux integration weight (face weighting) for the given coordinate system."""
geometric_flux_weight(::Cartesian, x, y) = 1.0
geometric_flux_weight(::Cylindrical, r, z) = r       # r at the face midpoint
geometric_flux_weight(::Spherical, r, θ) = r^2 * sin(θ)

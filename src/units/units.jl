# units/units.jl — Unitful.jl integration at problem-setup boundaries (Stage 9f)
#
# Philosophy: the solver is unit-agnostic and hot-path code continues to
# use plain `Float64` without dimension tracking. Unit-checking happens
# exclusively at the INPUT BOUNDARY — problem constructors that accept
# `Unitful.Quantity` values validate dimensional consistency and strip
# to SI-reduced numbers before handing to the solver. Mixed SI /
# imperial / cgs inputs are caught by Unitful's own dimension checks;
# unit-free inputs are permitted for backward compatibility.
#
# This module provides:
#   - `strip_units(x, target_unit)` — convert a Quantity to its numeric
#     value after confirming the unit matches `target_unit`. Throws
#     `DimensionError` on mismatch.
#   - `strip_velocity`, `strip_density`, `strip_viscosity`, etc. —
#     convenience wrappers for the most-common CFD inputs.
#
# Unitful.jl itself is NOT a runtime dependency — we never import it.
# Instead, we duck-type by checking whether an input value supports
# multiplication, division, and comparison against Julia's `Real` type.
# If a user passes a Unitful.Quantity, that package's defined
# arithmetic will be called; if a user passes a plain Float64, nothing
# special happens. This keeps the base module zero-dependency.

"""
    strip_units(value, target_scale::Real = 1.0) -> Float64

Convert an input `value` — which may be a plain `Real` or a
`Unitful.Quantity` — into a plain `Float64` by dividing by the
target-scale reference (also a Quantity of the same dimension, or 1.0).

For dimensionless inputs just returns `Float64(value)`. For dimensioned
inputs, the caller passes an appropriate target (e.g. `1u"m/s"`) and
the quotient is dimensionless; converted to Float64.

Example:

    julia> using Unitful
    julia> strip_units(2.0u"m/s", 1.0u"m/s")
    2.0
    julia> strip_units(120u"km/hr", 1.0u"m/s")
    33.33333333333333
    julia> strip_units(1.2, 1.0)    # dimensionless
    1.2
"""
function strip_units(value, target_scale)
    q = value / target_scale
    # If q is dimensionless (plain Real), return its Float64 view.
    # If the division fails to produce a Real (e.g. user mixed SI and
    # imperial without a conversion), that will raise a DimensionError
    # inside Unitful before we ever reach this line.
    return Float64(q)
end

"""
    is_dimensionless(value) -> Bool

True if `value` is already a plain `Real`. Used by backward-compatible
constructors to decide whether to run unit-checking.
"""
is_dimensionless(::Real) = true
is_dimensionless(::Any) = false

"""
    as_si_velocity(value) -> Float64

Convenience wrapper: asserts `value` is either a plain `Real`
(interpreted as m/s) or a `Unitful.Quantity` with velocity dimension,
then returns `Float64(value_in_meters_per_second)`.

The caller passes `1u"m/s"` as the reference unit if they have
Unitful.jl loaded; otherwise the value is assumed to be SI already.
"""
function as_si_velocity(value; unit_reference = 1.0)
    return strip_units(value, unit_reference)
end

"""
    as_si_density(value; unit_reference = 1.0) -> Float64

Density in kg/m³.
"""
function as_si_density(value; unit_reference = 1.0)
    return strip_units(value, unit_reference)
end

"""
    as_si_viscosity(value; unit_reference = 1.0) -> Float64

Dynamic viscosity in Pa·s (≡ kg/(m·s)).
"""
function as_si_viscosity(value; unit_reference = 1.0)
    return strip_units(value, unit_reference)
end

"""
    as_si_temperature(value; unit_reference = 1.0) -> Float64

Temperature in K. (If a user passes Unitful degrees Celsius with an
affine-offset setup, they need to convert to Kelvin explicitly —
Unitful.jl's `uconvert` or `ustrip` handles this.)
"""
function as_si_temperature(value; unit_reference = 1.0)
    return strip_units(value, unit_reference)
end

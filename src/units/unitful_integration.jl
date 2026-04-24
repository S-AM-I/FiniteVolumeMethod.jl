# units/unitful_integration.jl — Minimal Unitful.jl hook.
#
# We don't require Unitful as a hard dep; the helpers work on any type
# that supports `ustrip` and `unit`. When Unitful.jl is loaded, a weak-
# dep extension can override `strip_units` + `annotate_units` to use
# the package's API directly.

"""
    strip_units(x)

Return the bare numerical value of `x`. For plain `Number`s this is
the identity. Overridden by `FVMUnitfulExt` (if present) for `Unitful.Quantity`.
"""
strip_units(x::Number) = x

"""
    annotate_units(value, unit)

Return a unit-annotated value. In the pure-Julia path (no Unitful.jl
loaded) this returns the bare value — the annotation is a no-op
surface. With the `FVMUnitfulExt` loaded the extension overrides to
produce a true `Unitful.Quantity`.
"""
annotate_units(value, unit) = value

"""
    is_unitful(x) -> Bool

Heuristic: `false` for plain `Number`, `true` otherwise (any type with
a unit-aware shadow). Extensions override as needed.
"""
is_unitful(::Number) = false
is_unitful(::Any) = false

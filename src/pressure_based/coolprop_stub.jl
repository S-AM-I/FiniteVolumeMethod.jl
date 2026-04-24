# pressure_based/coolprop_stub.jl — CoolProp.jl weak-dep stub.
#
# The real implementation lives in `ext/FVMCoolPropExt.jl`; these stubs
# fire when CoolProp.jl is not loaded so callers fail loudly instead of
# silently returning zero.

"""
    CoolPropFluid(name::AbstractString)

Marker type naming a fluid in the CoolProp library. Accessor methods
(density, viscosity, specific_heat, thermal_conductivity) error unless
`CoolProp.jl` is loaded.
"""
struct CoolPropFluid
    name::String
end

_coolprop_required() = error(
    "CoolProp.jl required — add `using CoolProp` to activate FVMCoolPropExt",
)

coolprop_density(::CoolPropFluid, p, T) = _coolprop_required()
coolprop_viscosity(::CoolPropFluid, T) = _coolprop_required()
coolprop_specific_heat(::CoolPropFluid, T) = _coolprop_required()
coolprop_thermal_conductivity(::CoolPropFluid, T) = _coolprop_required()

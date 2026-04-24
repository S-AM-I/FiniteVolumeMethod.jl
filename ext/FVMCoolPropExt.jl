module FVMCoolPropExt

using FiniteVolumeMethod
using CoolProp

function FiniteVolumeMethod.coolprop_density(f::FiniteVolumeMethod.CoolPropFluid, p, T)
    return CoolProp.PropsSI("D", "P", p, "T", T, f.name)
end

function FiniteVolumeMethod.coolprop_viscosity(f::FiniteVolumeMethod.CoolPropFluid, T)
    return CoolProp.PropsSI("V", "T", T, "P", 101325.0, f.name)
end

function FiniteVolumeMethod.coolprop_specific_heat(f::FiniteVolumeMethod.CoolPropFluid, T)
    return CoolProp.PropsSI("C", "T", T, "P", 101325.0, f.name)
end

function FiniteVolumeMethod.coolprop_thermal_conductivity(f::FiniteVolumeMethod.CoolPropFluid, T)
    return CoolProp.PropsSI("L", "T", T, "P", 101325.0, f.name)
end

end

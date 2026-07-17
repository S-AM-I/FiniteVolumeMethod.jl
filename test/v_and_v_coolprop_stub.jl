# test/v_and_v_coolprop_stub.jl — CoolProp weak-dep stub ergonomics.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: CoolPropFluid
using Test

@testset "V&V: CoolPropFluid — constructor round-trip" begin
    f = FiniteVolumeMethod.CoolPropFluid("Water")
    @test f.name == "Water"
end

@testset "V&V: CoolProp accessors — error without CoolProp.jl loaded" begin
    f = FiniteVolumeMethod.CoolPropFluid("Water")
    @test_throws ErrorException FiniteVolumeMethod.coolprop_density(f, 101325.0, 300.0)
    @test_throws ErrorException FiniteVolumeMethod.coolprop_viscosity(f, 300.0)
    @test_throws ErrorException FiniteVolumeMethod.coolprop_specific_heat(f, 300.0)
    @test_throws ErrorException FiniteVolumeMethod.coolprop_thermal_conductivity(f, 300.0)
end

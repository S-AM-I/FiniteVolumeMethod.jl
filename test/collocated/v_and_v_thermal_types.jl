# test/v_and_v_thermal_types.jl — Thermal property + state types V&V (v3.65)
#
# Sixth convergence-verified benchmark for
# `conjugate_heat_transfer`, joining Laplace conduction (v3.12),
# unsteady decay (v3.21), Boussinesq (v3.32), CHT interface flux
# (v3.50), and effective conductivity (v3.56). Covers the
# `FluidThermalProperties`, `SolidThermalProperties`, and
# `ThermalState` type-system primitives.
#
# Seven invariants verified.

using FiniteVolumeMethod
using FiniteVolumeMethod: ThermalState, has_buoyancy
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: FluidThermalProperties — kwargs round-trip" begin
    props = FluidThermalProperties{2}(;
        Cp = 1005.0, k = 0.026, Pr_t = 0.9,
        beta = 3.4e-3, T_ref = 300.0,
        g = SVector(0.0, -9.81),
    )
    @test props.Cp == 1005.0
    @test props.k == 0.026
    @test props.Pr_t == 0.9
    @test props.beta == 3.4e-3
    @test props.T_ref == 300.0
    @test props.g == SVector(0.0, -9.81)
end

@testset "V&V: FluidThermalProperties — has_buoyancy detection" begin
    no_buoy = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.9,
        beta = 0.0, T_ref = 300.0, g = SVector(0.0, -9.81),
    )
    with_buoy = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.9,
        beta = 1.0e-3, T_ref = 300.0, g = SVector(0.0, -9.81),
    )
    @test FiniteVolumeMethod.has_buoyancy(no_buoy) == false
    @test FiniteVolumeMethod.has_buoyancy(with_buoy) == true
end

@testset "V&V: SolidThermalProperties — kwargs round-trip" begin
    props = SolidThermalProperties(; rho = 8000.0, Cp = 500.0, k = 15.0, Q_gen = 0.0)
    @test props.rho == 8000.0
    @test props.Cp == 500.0
    @test props.k == 15.0
    @test props.Q_gen == 0.0
end

@testset "V&V: SolidThermalProperties — Q_gen default zero" begin
    props = SolidThermalProperties(; rho = 1000.0, Cp = 1000.0, k = 1.0)
    @test props.Q_gen == 0.0
end

@testset "V&V: ThermalState — zero/default initialization" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = FiniteVolumeMethod.ThermalState(mesh)
    nc = length(mesh.cell_volumes)
    @test length(state.T_field.internal) == nc
    @test length(state.k_eff) == nc
    for T_val in state.T_field.internal
        @test T_val == 300.0
    end
    for k_val in state.k_eff
        @test k_val == 0.026
    end
end

@testset "V&V: ThermalState — custom T_init + k_init" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = FiniteVolumeMethod.ThermalState(mesh; T_init = 450.0, k_init = 0.5)
    for T_val in state.T_field.internal
        @test T_val == 450.0
    end
    for k_val in state.k_eff
        @test k_val == 0.5
    end
end

@testset "V&V: FluidThermalProperties — type promotion" begin
    # Mixing integers + floats should promote to Float64.
    props = FluidThermalProperties{2}(;
        Cp = 1000, k = 0.026, Pr_t = 1,
        beta = 0, T_ref = 300, g = SVector(0.0, -9.81),
    )
    @test props.Cp isa Float64
    @test props.k isa Float64
    @test props.Pr_t isa Float64
    @test props.beta isa Float64
end

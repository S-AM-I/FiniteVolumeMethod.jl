# test/v_and_v_combustion_props.jl — CombustionProperties + SpeciesState V&V (v3.61)
#
# Fifth convergence-verified benchmark for `combustion`, joining
# species AD (v3.17), EDM algebra (v3.27), Arrhenius kinetics
# (v3.37), and FR/ED blending (v3.47). Covers the
# `CombustionProperties` + `SpeciesState` constructor primitives
# — the type-system contracts consumed by every combustion solve.
#
# Six invariants verified.

using FiniteVolumeMethod
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: CombustionProperties — defaults match documented values" begin
    props = CombustionProperties()
    @test props.species_names == (:fuel, :oxidizer, :product)
    @test props.molecular_weights == (16.0, 32.0, 44.0)
    @test all(d == 2.0e-5 for d in props.diffusivities)
    @test props.Sc_t == 0.7
    @test props.stoich_ratio == 4.0
    @test props.heat_of_combustion == 5.0e7
end

@testset "V&V: CombustionProperties — custom kwargs round-trip" begin
    props = CombustionProperties(;
        species_names = (:H2, :O2, :H2O),
        molecular_weights = (2.016, 31.998, 18.015),
        diffusivities = (7.5e-5, 2.0e-5, 2.5e-5),
        Sc_t = 0.85,
        stoich_ratio = 7.936,
        heat_of_combustion = 120.0e6,
    )
    @test props.species_names == (:H2, :O2, :H2O)
    @test props.molecular_weights == (2.016, 31.998, 18.015)
    @test props.diffusivities == (7.5e-5, 2.0e-5, 2.5e-5)
    @test props.Sc_t == 0.85
    @test props.stoich_ratio == 7.936
    @test props.heat_of_combustion == 120.0e6
end

@testset "V&V: CombustionProperties — type promotion" begin
    # Mixing integers and floats should promote to Float64.
    props = CombustionProperties(;
        molecular_weights = (16, 32, 44),
        diffusivities = (2.0e-5, 2.0e-5, 2.0e-5),
        Sc_t = 1,
        stoich_ratio = 4,
        heat_of_combustion = 5.0e7,
    )
    @test eltype(props.molecular_weights) == Float64
    @test eltype(props.diffusivities) == Float64
    @test props.Sc_t isa Float64
end

@testset "V&V: SpeciesState — default zero initialization" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    props = CombustionProperties()
    state = SpeciesState(mesh, props)

    # Without Y_init kwargs, all species start at 0.
    for i in 1:3
        for val in state.Y[i].internal
            @test val == 0.0
        end
    end
end

@testset "V&V: SpeciesState — kwarg Y_init round-trip" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    props = CombustionProperties()
    state = SpeciesState(mesh, props; fuel = 0.05, oxidizer = 0.233, product = 0.0)

    for val in state.Y[1].internal   # fuel
        @test val == 0.05
    end
    for val in state.Y[2].internal   # oxidizer
        @test val == 0.233
    end
    for val in state.Y[3].internal   # product
        @test val == 0.0
    end
end

@testset "V&V: CombustionProperties — stoichiometry accessible" begin
    # The stoich_ratio field is consumed by EDM, Arrhenius,
    # and FR/ED as the s coefficient in:
    #    ω_ox = s · ω_fuel,   ω_prod = −(1+s) · ω_fuel
    # Verify round-trip and sign consistency.
    props = CombustionProperties(; stoich_ratio = 4.0)
    @test props.stoich_ratio > 0.0

    # The products' stoichiometric coefficient (1+s) is consistent.
    @test (1 + props.stoich_ratio) == 5.0
end

@testset "V&V: Species state — field mesh consistency" begin
    mesh = build_cartesian_unstructured_mesh(5, 5, 1.0, 1.0)
    props = CombustionProperties()
    state = SpeciesState(mesh, props; fuel = 0.1)

    nc = length(mesh.cell_volumes)
    for i in 1:3
        @test length(state.Y[i].internal) == nc
    end
end

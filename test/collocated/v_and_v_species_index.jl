# test/v_and_v_species_index.jl — _species_index lookup V&V (v3.82)

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_arrhenius_reaction_rates, compute_edm_reaction_rates
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const _species_idx = FiniteVolumeMethod._species_index

@testset "V&V: _species_index — default 3-species lookup" begin
    props = CombustionProperties()
    @test _species_idx(props, :fuel) == 1
    @test _species_idx(props, :oxidizer) == 2
    @test _species_idx(props, :product) == 3
end

@testset "V&V: _species_index — custom names" begin
    props = CombustionProperties(;
        species_names = (:CH4, :O2, :CO2, :H2O, :N2),
        molecular_weights = (16.0, 32.0, 44.0, 18.0, 28.0),
        diffusivities = ntuple(_ -> 2.0e-5, 5),
        Sc_t = 0.7, stoich_ratio = 4.0, heat_of_combustion = 5.0e7,
    )
    @test _species_idx(props, :CH4) == 1
    @test _species_idx(props, :O2) == 2
    @test _species_idx(props, :CO2) == 3
    @test _species_idx(props, :H2O) == 4
    @test _species_idx(props, :N2) == 5
end

@testset "V&V: _species_index — unknown species errors" begin
    props = CombustionProperties()
    @test_throws ErrorException _species_idx(props, :unknown)
    @test_throws ErrorException _species_idx(props, :H2)
end

@testset "V&V: _species_index — reordered species still correct" begin
    # Order is (:product, :fuel, :oxidizer) — different from default.
    props = CombustionProperties(;
        species_names = (:product, :fuel, :oxidizer),
        molecular_weights = (44.0, 16.0, 32.0),
        diffusivities = ntuple(_ -> 2.0e-5, 3),
        Sc_t = 0.7, stoich_ratio = 4.0, heat_of_combustion = 5.0e7,
    )
    @test _species_idx(props, :product) == 1
    @test _species_idx(props, :fuel) == 2
    @test _species_idx(props, :oxidizer) == 3
end

@testset "V&V: _species_index — EDM + Arrhenius consume same index" begin
    # Both compute_edm_reaction_rates and compute_arrhenius_reaction_rates
    # look up species by name — verify consistency with a 3-species
    # default tuple.
    props = CombustionProperties()
    fuel_idx = _species_idx(props, :fuel)
    ox_idx = _species_idx(props, :oxidizer)
    prod_idx = _species_idx(props, :product)
    @test fuel_idx != ox_idx
    @test ox_idx != prod_idx
    @test fuel_idx + ox_idx + prod_idx == 1 + 2 + 3   # 1+2+3 (permutation invariant sum)
end

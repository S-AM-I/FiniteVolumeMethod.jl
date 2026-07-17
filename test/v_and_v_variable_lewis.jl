# test/v_and_v_variable_lewis.jl — Variable Lewis number V&V
#
# Validates that `VariableLewis` correctly modifies the effective
# species diffusivity `α_species_i = α_thermal / Le_i` and preserves
# the unity-Le limit exactly.

using FiniteVolumeMethod
using FiniteVolumeMethod: VariableLewis, lewis_number, species_diffusivity
using Test

include("TestHelpers.jl")

@testset "V&V: Variable Lewis — Le ≡ 1 recovers unity-Le diffusivity" begin
    props = CombustionProperties(; stoich_ratio = 4.0)
    Le = VariableLewis(props)  # defaults to Le_i = 1 per species
    alpha_thermal = 2.5e-5
    for i in 1:3
        @test isapprox(
            species_diffusivity(Le, alpha_thermal, i),
            alpha_thermal; rtol = 1.0e-12,
        )
    end
end

@testset "V&V: Variable Lewis — Le = 2 halves diffusion coefficient" begin
    Le = VariableLewis((2.0, 2.0, 2.0))
    alpha_thermal = 3.0e-5
    for i in 1:3
        @test species_diffusivity(Le, alpha_thermal, i) == alpha_thermal / 2.0
    end
end

@testset "V&V: Variable Lewis — species index round-trip" begin
    props = CombustionProperties(; stoich_ratio = 4.0)
    Le = VariableLewis(props; fuel = 0.3, oxidizer = 1.1, product = 1.0)
    @test lewis_number(Le, 1) == 0.3
    @test lewis_number(Le, 2) == 1.1
    @test lewis_number(Le, 3) == 1.0
end

@testset "V&V: Variable Lewis — per-species Le preserved across constructor" begin
    Le_tuple = (0.3, 1.1, 1.0)
    Le = VariableLewis(Le_tuple)
    for i in 1:3
        @test Le.Le[i] == Le_tuple[i]
    end
end

@testset "V&V: Variable Lewis — monotone Le ⇒ monotone α_species" begin
    # Increasing Le_i strictly decreases α_species_i.
    Le = VariableLewis((0.5, 1.0, 2.0))
    alpha_thermal = 1.0e-5
    alpha = [species_diffusivity(Le, alpha_thermal, i) for i in 1:3]
    @test alpha[1] > alpha[2] > alpha[3]
end

@testset "V&V: Variable Lewis — per-cell α_thermal vector" begin
    Le = VariableLewis((2.0, 1.0, 0.5))
    alpha_vec = [1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5]

    alpha_1 = species_diffusivity(Le, alpha_vec, 1)
    alpha_2 = species_diffusivity(Le, alpha_vec, 2)
    alpha_3 = species_diffusivity(Le, alpha_vec, 3)

    @test length(alpha_1) == length(alpha_vec)
    for (i, a) in enumerate(alpha_vec)
        @test isapprox(alpha_1[i], a / 2.0; rtol = 1.0e-14)
        @test isapprox(alpha_2[i], a; rtol = 1.0e-14)
        @test isapprox(alpha_3[i], a / 0.5; rtol = 1.0e-14)
    end
end

@testset "V&V: Variable Lewis — positivity check" begin
    # Non-positive Lewis numbers must be rejected at construction.
    @test_throws ErrorException VariableLewis((1.0, -0.5, 1.0))
    @test_throws ErrorException VariableLewis((0.0, 1.0, 1.0))
end

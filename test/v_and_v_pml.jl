# test/v_and_v_pml.jl — PML sponge-zone primitive V&V (Wave 3 Agent D)
#
# Verifies the polynomial σ(x) profile, the damping-source algebra, and
# the outside-zone zero-behaviour of `PMLZone`. All tests are primitive:
# they probe `pml_sigma` at hand-picked positions and assert closed-form
# ratios, then drive `add_pml_source!` on a tiny sample and verify the
# source equals −σ(x) · (φ − φ_∞) to machine precision.
#
# Invariants:
#
# 1. σ(inner) = 0, σ(outer) = σ_max.
# 2. Quadratic profile: σ at the midpoint equals 0.25 · σ_max.
# 3. Quartic profile:   σ at the midpoint equals 0.0625 · σ_max.
# 4. σ is monotone non-decreasing across the layer.
# 5. Outside the PML slab (either CFD side or past the far wall): σ = 0.
# 6. `add_pml_source!` pulls φ toward φ_∞ with weight exactly −σ.

using StaticArrays
using Test

_experimental_warn(::Symbol) = nothing # no-op shim: source included standalone, outside module Experimental
include(joinpath(@__DIR__, "..", "src", "experimental", "aeroacoustics", "pml.jl"))

@testset "V&V: PML σ endpoints — σ(inner) = 0, σ(outer) = σ_max" begin
    inner = SVector(1.0, 0.0)
    outer = SVector(2.0, 0.0)
    sigma_max = 7.5
    pml = PMLZone(inner, outer, sigma_max)

    @test pml_sigma(pml, inner) == 0.0
    @test isapprox(pml_sigma(pml, outer), sigma_max; rtol = 1.0e-14)
end

@testset "V&V: quadratic profile σ(mid) = 0.25 · σ_max" begin
    inner = SVector(0.0, 0.0)
    outer = SVector(1.0, 0.0)
    sigma_max = 3.0
    pml = PMLZone(inner, outer, sigma_max; profile = Quadratic)

    mid = SVector(0.5, 0.0)
    @test isapprox(pml_sigma(pml, mid), 0.25 * sigma_max; rtol = 1.0e-14)

    # Quarter-point: σ(0.25) = σ_max · 0.25² = σ_max / 16.
    quarter = SVector(0.25, 0.0)
    @test isapprox(pml_sigma(pml, quarter), sigma_max / 16.0; rtol = 1.0e-14)
end

@testset "V&V: quartic profile σ(mid) = 0.0625 · σ_max" begin
    inner = SVector(0.0, 0.0)
    outer = SVector(1.0, 0.0)
    sigma_max = 3.0
    pml = PMLZone(inner, outer, sigma_max; profile = Quartic)

    mid = SVector(0.5, 0.0)
    @test isapprox(pml_sigma(pml, mid), 0.0625 * sigma_max; rtol = 1.0e-14)

    # Quarter-point: σ(0.25) = σ_max · 0.25⁴ = σ_max / 256.
    quarter = SVector(0.25, 0.0)
    @test isapprox(pml_sigma(pml, quarter), sigma_max / 256.0; rtol = 1.0e-14)
end

@testset "V&V: monotone non-decreasing σ across layer" begin
    inner = SVector(0.0, 0.0)
    outer = SVector(1.0, 0.0)
    sigma_max = 5.0
    for prof in (Quadratic, Quartic)
        pml = PMLZone(inner, outer, sigma_max; profile = prof)
        prev = -Inf
        for s in range(0.0, 1.0; length = 21)
            val = pml_sigma(pml, SVector(s, 0.0))
            @test val >= prev - 1.0e-14
            prev = val
        end
    end
end

@testset "V&V: σ = 0 outside the PML slab" begin
    inner = SVector(1.0, 0.0)
    outer = SVector(2.0, 0.0)
    sigma_max = 4.0
    pml = PMLZone(inner, outer, sigma_max)

    # CFD side (x < 1).
    @test pml_sigma(pml, SVector(0.5, 0.0)) == 0.0
    @test pml_sigma(pml, SVector(-10.0, 0.0)) == 0.0
    # Far side (x > 2).
    @test pml_sigma(pml, SVector(2.5, 0.0)) == 0.0
    @test pml_sigma(pml, SVector(100.0, 0.0)) == 0.0
end

@testset "V&V: reversed-direction slab (outer < inner) still works" begin
    # Sponge at the left boundary: inner = 1.0, outer = 0.0.
    inner = SVector(1.0, 0.0)
    outer = SVector(0.0, 0.0)
    sigma_max = 2.0
    pml = PMLZone(inner, outer, sigma_max)

    @test pml_sigma(pml, inner) == 0.0
    @test isapprox(pml_sigma(pml, outer), sigma_max; rtol = 1.0e-14)
    @test isapprox(pml_sigma(pml, SVector(0.5, 0.0)), 0.25 * sigma_max; rtol = 1.0e-14)
    # Outside the CFD-facing side (x > 1).
    @test pml_sigma(pml, SVector(1.5, 0.0)) == 0.0
end

@testset "V&V: add_pml_source! pulls φ toward φ_∞ with weight −σ" begin
    inner = SVector(0.0, 0.0)
    outer = SVector(1.0, 0.0)
    sigma_max = 2.0
    pml = PMLZone(inner, outer, sigma_max; profile = Quadratic)

    points = [
        SVector(-0.5, 0.0),   # outside (CFD side)  → σ = 0
        SVector(0.0, 0.0),    # inner               → σ = 0
        SVector(0.5, 0.0),    # mid                 → σ = 0.25 · σ_max
        SVector(1.0, 0.0),    # outer               → σ = σ_max
        SVector(1.5, 0.0),    # outside (far side)  → σ = 0
    ]
    phi = [2.0, 3.0, 4.0, 5.0, 6.0]
    phi_far = 1.0
    source = zeros(Float64, length(points))

    add_pml_source!(source, phi, phi_far, pml, points)

    expected = [
        0.0,
        0.0,
        -0.25 * sigma_max * (4.0 - 1.0),
        -sigma_max * (5.0 - 1.0),
        0.0,
    ]
    for i in eachindex(source)
        @test isapprox(source[i], expected[i]; rtol = 1.0e-14, atol = 1.0e-14)
    end
end

@testset "V&V: add_pml_source! length-mismatch errors" begin
    inner = SVector(0.0, 0.0)
    outer = SVector(1.0, 0.0)
    pml = PMLZone(inner, outer, 1.0)

    points = [SVector(0.5, 0.0), SVector(0.75, 0.0)]
    phi = [1.0, 2.0]
    source_bad = zeros(Float64, 1)
    @test_throws ErrorException add_pml_source!(source_bad, phi, 0.0, pml, points)
    source_ok = zeros(Float64, 2)
    phi_bad = [1.0]
    @test_throws ErrorException add_pml_source!(source_ok, phi_bad, 0.0, pml, points)
end

@testset "V&V: 2-D PML slab picks dominant axis" begin
    # Thin slab at top of a 2-D box: inner at y = 9, outer at y = 10.
    inner = SVector(0.0, 9.0)
    outer = SVector(0.0, 10.0)
    sigma_max = 3.0
    pml = PMLZone(inner, outer, sigma_max; profile = Quadratic)

    # x should not matter — active axis is y.
    @test isapprox(pml_sigma(pml, SVector(5.0, 9.5)), 0.25 * sigma_max; rtol = 1.0e-14)
    @test isapprox(pml_sigma(pml, SVector(-100.0, 10.0)), sigma_max; rtol = 1.0e-14)
    # Below the layer.
    @test pml_sigma(pml, SVector(5.0, 8.0)) == 0.0
end

@testset "V&V: sigma_max = 0 ⇒ source everywhere zero" begin
    pml = PMLZone(SVector(0.0, 0.0), SVector(1.0, 0.0), 0.0)
    points = [SVector(s, 0.0) for s in range(-1.0, 2.0; length = 7)]
    for x in points
        @test pml_sigma(pml, x) == 0.0
    end
end

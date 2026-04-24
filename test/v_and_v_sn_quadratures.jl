# test/v_and_v_sn_quadratures.jl — Higher-order SN quadrature V&V
#
# Verifies that the level-symmetric S6, S8, and S12 quadratures added
# for Wave 2 satisfy the standard unit-sphere moment identities:
#   1. All direction vectors are unit: |Ω̂| = 1
#   2. Σ w = 4π (3D) or 2π (2D)
#   3. Σ w · Ω̂ = 0 (isotropy)
#   4. Direction counts match Lewis & Miller (1984), Table 4-1:
#        S6  → 3D 48,  2D 24
#        S8  → 3D 80,  2D 40
#        S12 → 3D 168, 2D 84
#   5. Direction counts strictly increase with order (S6 < S8 < S12)

using FiniteVolumeMethod
using LinearAlgebra: norm
using StaticArrays
using Test

include("TestHelpers.jl")

const _s6_q = FiniteVolumeMethod._s6_quadrature
const _s8_q = FiniteVolumeMethod._s8_quadrature
const _s12_q = FiniteVolumeMethod._s12_quadrature

# --------------------------------------------------------------------
# 3D tests
# --------------------------------------------------------------------

@testset "V&V: SN — 3D S6 has 48 unit directions" begin
    dirs, w = _s6_q(Val(3), Float64)
    @test length(dirs) == 48
    @test length(w) == 48
    for d in dirs
        @test isapprox(norm(d), 1.0; rtol = 1.0e-14)
    end
end

@testset "V&V: SN — 3D S8 has 80 unit directions" begin
    dirs, w = _s8_q(Val(3), Float64)
    @test length(dirs) == 80
    @test length(w) == 80
    for d in dirs
        @test isapprox(norm(d), 1.0; rtol = 1.0e-14)
    end
end

@testset "V&V: SN — 3D S12 has 168 unit directions" begin
    dirs, w = _s12_q(Val(3), Float64)
    @test length(dirs) == 168
    @test length(w) == 168
    for d in dirs
        @test isapprox(norm(d), 1.0; rtol = 1.0e-14)
    end
end

@testset "V&V: SN — 3D Σ w = 4π for every order" begin
    for (q, name) in ((_s6_q, "S6"), (_s8_q, "S8"), (_s12_q, "S12"))
        _, w = q(Val(3), Float64)
        @test isapprox(sum(w), 4 * pi; rtol = 1.0e-12)
    end
end

@testset "V&V: SN — 3D isotropy Σ w·Ω̂ = 0" begin
    for q in (_s6_q, _s8_q, _s12_q)
        dirs, w = q(Val(3), Float64)
        s = sum(w[i] .* dirs[i] for i in 1:length(dirs))
        @test norm(s) < 1.0e-12
    end
end

@testset "V&V: SN — 3D direction counts strictly increasing" begin
    n6 = length(_s6_q(Val(3), Float64)[1])
    n8 = length(_s8_q(Val(3), Float64)[1])
    n12 = length(_s12_q(Val(3), Float64)[1])
    @test n6 < n8 < n12
end

@testset "V&V: SN — 3D all weights non-negative" begin
    for q in (_s6_q, _s8_q, _s12_q)
        _, w = q(Val(3), Float64)
        @test all(wi -> wi >= 0.0, w)
    end
end

# --------------------------------------------------------------------
# 2D tests (half-space projection of 3D)
# --------------------------------------------------------------------

@testset "V&V: SN — 2D S6 has 24 unit directions" begin
    dirs, w = _s6_q(Val(2), Float64)
    @test length(dirs) == 24
    @test length(w) == 24
    for d in dirs
        @test isapprox(norm(d), 1.0; rtol = 1.0e-12)
    end
end

@testset "V&V: SN — 2D S8 has 40 unit directions" begin
    dirs, w = _s8_q(Val(2), Float64)
    @test length(dirs) == 40
    @test length(w) == 40
    for d in dirs
        @test isapprox(norm(d), 1.0; rtol = 1.0e-12)
    end
end

@testset "V&V: SN — 2D S12 has 84 unit directions" begin
    dirs, w = _s12_q(Val(2), Float64)
    @test length(dirs) == 84
    @test length(w) == 84
    for d in dirs
        @test isapprox(norm(d), 1.0; rtol = 1.0e-12)
    end
end

@testset "V&V: SN — 2D Σ w = 2π for every order" begin
    for q in (_s6_q, _s8_q, _s12_q)
        _, w = q(Val(2), Float64)
        @test isapprox(sum(w), 2 * pi; rtol = 1.0e-12)
    end
end

@testset "V&V: SN — 2D isotropy Σ w·Ω̂ = 0" begin
    for q in (_s6_q, _s8_q, _s12_q)
        dirs, w = q(Val(2), Float64)
        s = sum(w[i] .* dirs[i] for i in 1:length(dirs))
        @test norm(s) < 1.0e-12
    end
end

@testset "V&V: SN — 2D direction counts strictly increasing" begin
    n6 = length(_s6_q(Val(2), Float64)[1])
    n8 = length(_s8_q(Val(2), Float64)[1])
    n12 = length(_s12_q(Val(2), Float64)[1])
    @test n6 < n8 < n12
end

# --------------------------------------------------------------------
# High-level FvDOMModel dispatch
# --------------------------------------------------------------------

@testset "V&V: SN — FvDOMModel dispatches :S6, :S8, :S12 in 2D" begin
    m6 = FvDOMModel(; Dim = 2, order = :S6)
    m8 = FvDOMModel(; Dim = 2, order = :S8)
    m12 = FvDOMModel(; Dim = 2, order = :S12)
    @test length(m6.directions) == 24
    @test length(m8.directions) == 40
    @test length(m12.directions) == 84
end

@testset "V&V: SN — FvDOMModel dispatches :S6, :S8, :S12 in 3D" begin
    m6 = FvDOMModel(; Dim = 3, order = :S6)
    m8 = FvDOMModel(; Dim = 3, order = :S8)
    m12 = FvDOMModel(; Dim = 3, order = :S12)
    @test length(m6.directions) == 48
    @test length(m8.directions) == 80
    @test length(m12.directions) == 168
end

@testset "V&V: SN — unknown order errors" begin
    @test_throws ErrorException FvDOMModel(; order = :S42)
end

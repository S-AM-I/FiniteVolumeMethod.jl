# test/v_and_v_fgm.jl — Flamelet-Generated Manifold (FGM) V&V
#
# Validates the bilinear-interpolation invariants of the runtime FGM
# lookup:
#   - constant callbacks round-trip exactly,
#   - table-corner samples match the source,
#   - linear callbacks reproduce the closed-form at interior points,
#   - out-of-range inputs clamp to `[0, 1]`.

using FiniteVolumeMethod
using Test

@testset "V&V: FGM — constant callback round-trip" begin
    f_const = (C, Z) -> (0.1, 0.2, 0.7)
    table = build_fgm_table_from_callback(f_const, 8, 8)

    for C in (0.0, 0.25, 0.5, 0.75, 1.0), Z in (0.0, 0.33, 0.66, 1.0)
        Y = lookup_fgm(table, C, Z)
        @test length(Y) == 3
        @test isapprox(Y[1], 0.1; rtol = 1.0e-14)
        @test isapprox(Y[2], 0.2; rtol = 1.0e-14)
        @test isapprox(Y[3], 0.7; rtol = 1.0e-14)
    end
end

@testset "V&V: FGM — bilinear hits table corners exactly" begin
    f = (C, Z) -> (C + Z, C * Z, 1.0 - C - Z)
    NC, NZ = 5, 5
    table = build_fgm_table_from_callback(f, NC, NZ)

    for iC in 1:NC, iZ in 1:NZ
        C = table.C_grid[iC]
        Z = table.Z_grid[iZ]
        Y = lookup_fgm(table, C, Z)
        for s in 1:3
            @test isapprox(Y[s], table.Y[iC, iZ, s]; rtol = 1.0e-14)
        end
    end
end

@testset "V&V: FGM — linear callback matches closed form" begin
    # A callback that is linear in C and Z is reproduced exactly by
    # bilinear interpolation (up to rounding).
    f = (C, Z) -> (C, Z, 1.0 - C - Z)
    NC, NZ = 11, 11
    table = build_fgm_table_from_callback(f, NC, NZ)

    Cs = range(0.05, 0.95; length = 5)
    Zs = range(0.05, 0.95; length = 5)
    for C in Cs, Z in Zs
        Y = lookup_fgm(table, C, Z)
        @test isapprox(Y[1], C; atol = 1.0e-12)
        @test isapprox(Y[2], Z; atol = 1.0e-12)
        @test isapprox(Y[3], 1.0 - C - Z; atol = 1.0e-12)
    end
end

@testset "V&V: FGM — out-of-range inputs clamp to [0, 1]" begin
    f = (C, Z) -> (C, Z, C + Z)
    table = build_fgm_table_from_callback(f, 9, 9)

    # C below 0 / above 1 should clamp.
    Y_low = lookup_fgm(table, -0.5, 0.3)
    Y_zero = lookup_fgm(table, 0.0, 0.3)
    @test isapprox(Y_low[1], Y_zero[1]; rtol = 1.0e-14)
    @test isapprox(Y_low[2], Y_zero[2]; rtol = 1.0e-14)

    Y_high = lookup_fgm(table, 1.5, 0.7)
    Y_one = lookup_fgm(table, 1.0, 0.7)
    @test isapprox(Y_high[1], Y_one[1]; rtol = 1.0e-14)
    @test isapprox(Y_high[2], Y_one[2]; rtol = 1.0e-14)

    # Z clamps too.
    Y_Z_low = lookup_fgm(table, 0.4, -2.0)
    Y_Z_zero = lookup_fgm(table, 0.4, 0.0)
    for s in 1:3
        @test isapprox(Y_Z_low[s], Y_Z_zero[s]; rtol = 1.0e-14)
    end
end

@testset "V&V: FGM — in-place lookup matches allocating variant" begin
    f = (C, Z) -> (C^2, Z^2, 1.0 - C * Z)
    table = build_fgm_table_from_callback(f, 7, 7)

    Y_buffer = zeros(3)
    for (C, Z) in ((0.2, 0.4), (0.55, 0.1), (0.9, 0.9))
        Y_alloc = lookup_fgm(table, C, Z)
        lookup_fgm!(Y_buffer, table, C, Z)
        for s in 1:3
            @test isapprox(Y_buffer[s], Y_alloc[s]; rtol = 1.0e-14)
        end
    end
end

@testset "V&V: FGM — Cantera stub errors without extension" begin
    @test_throws ErrorException compute_fgm_table_from_cantera(
        nothing, 5, 5, "CH4", "O2:1,N2:3.76",
    )
end

@testset "V&V: FGM — bad arguments are rejected" begin
    # NC or NZ < 2 are rejected.
    @test_throws ErrorException build_fgm_table_from_callback(
        (C, Z) -> (1.0,), 1, 5,
    )
    @test_throws ErrorException build_fgm_table_from_callback(
        (C, Z) -> (1.0,), 5, 1,
    )
    # Callback must return a Tuple.
    @test_throws ErrorException build_fgm_table_from_callback(
        (C, Z) -> 1.0, 5, 5,
    )
end

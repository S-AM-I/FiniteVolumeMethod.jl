# test/v_and_v_unitful.jl — Unitful hook round-trip without Unitful.jl.

using FiniteVolumeMethod
using Test

@testset "V&V: strip_units — plain Number passthrough" begin
    @test strip_units(3.14) === 3.14
    @test strip_units(42) === 42
end

@testset "V&V: annotate_units — no-op in pure-Julia path" begin
    # Without Unitful loaded, annotate_units returns the bare value
    # (the FVMUnitfulExt extension will override this).
    v = annotate_units(5.0, :meter)
    @test v == 5.0
end

@testset "V&V: is_unitful — false on plain numbers" begin
    @test !is_unitful(1.0)
    @test !is_unitful(42)
end

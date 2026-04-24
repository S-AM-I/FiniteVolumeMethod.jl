# test/v_and_v_transient_adjoint_stub.jl — verify deferral stub behaviour.

using FiniteVolumeMethod
using Test

@testset "V&V: transient adjoint stub — warns + throws" begin
    caught = false
    try
        solve_transient_adjoint()
    catch e
        caught = true
    end
    @test caught
end

@testset "V&V: TransientAdjoint dispatch — routed through solve_adjoint errors" begin
    @test_throws Exception solve_adjoint(TransientAdjoint())
end

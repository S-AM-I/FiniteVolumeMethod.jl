# test/v_and_v_enzyme_stub.jl — Enzyme full-solver AD stub ergonomics.

using FiniteVolumeMethod
using Test

@testset "V&V: autodiff_forward_step — errors without Enzyme" begin
    step = x -> x
    state = [1.0, 2.0]
    dstate = [0.0, 0.0]
    caught = false
    try
        FiniteVolumeMethod.autodiff_forward_step(step, state, dstate)
    catch e
        caught = true
    end
    @test caught
end

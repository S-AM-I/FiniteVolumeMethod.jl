# test/v_and_v_pv_coupling.jl — SIMPLE/PISO/PIMPLE constructor V&V (v3.86)

using FiniteVolumeMethod
using FiniteVolumeMethod: AbstractPVCoupling
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: SIMPLE — kwargs defaults" begin
    algo = SIMPLE()
    @test algo isa FiniteVolumeMethod.AbstractPVCoupling
    @test algo.alpha_U > 0.0
    @test algo.alpha_p > 0.0
    @test algo.max_iterations >= 1
    @test algo.tolerance > 0.0
end

@testset "V&V: SIMPLE — positional kwargs round-trip" begin
    algo = SIMPLE(0.7, 0.3, 500, 1.0e-5)
    @test algo.alpha_U == 0.7
    @test algo.alpha_p == 0.3
    @test algo.max_iterations == 500
    @test algo.tolerance == 1.0e-5
end

@testset "V&V: PISO — default n_correctors" begin
    algo = PISO()
    @test algo isa FiniteVolumeMethod.AbstractPVCoupling
    @test algo.n_correctors == 2
end

@testset "V&V: PISO — custom n_correctors" begin
    algo = PISO(; n_correctors = 5)
    @test algo.n_correctors == 5
end

@testset "V&V: PIMPLE — default kwargs" begin
    algo = PIMPLE()
    @test algo isa FiniteVolumeMethod.AbstractPVCoupling
    @test algo.n_outer >= 1
    @test algo.n_correctors >= 1
    @test 0.0 < algo.alpha_U <= 1.0
    @test 0.0 < algo.alpha_p <= 1.0
end

@testset "V&V: AbstractPVCoupling hierarchy" begin
    @test SIMPLE() isa FiniteVolumeMethod.AbstractPVCoupling
    @test PISO() isa FiniteVolumeMethod.AbstractPVCoupling
    @test PIMPLE() isa FiniteVolumeMethod.AbstractPVCoupling
end

@testset "V&V: SIMPLE — alpha_U < 1 under-relaxation expected" begin
    algo = SIMPLE()
    # Under-relaxation: alpha_U < 1 typically.
    @test 0.0 < algo.alpha_U <= 1.0
    @test 0.0 < algo.alpha_p <= 1.0
end

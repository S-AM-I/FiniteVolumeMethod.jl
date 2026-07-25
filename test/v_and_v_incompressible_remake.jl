# test/v_and_v_incompressible_remake.jl — remake parameter-update V&V (v3.64)
#
# Sixth convergence-verified benchmark for `incompressible_ns`,
# joining Ghia Re=100 (v3.1), Poiseuille (v3.10), Couette (v3.22),
# transient PISO (v3.41), and SciML interface (v3.55). Covers the
# `SciMLBase.remake` parameter-update path — the contract
# consumed by sensitivity + optimization workflows.
#
# Six invariants verified.

using FiniteVolumeMethod
using SciMLBase: remake
using StaticArrays: SVector
using Test

include("TestHelpers.jl")

function build_prob(; nu = 0.1, density = 1.0)
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(0.1, 0.0)),
    )
    return SteadyIncompressibleProblem(
        mesh, bcs, SIMPLE(0.5, 0.2, 10, 1.0e-5);
        nu = nu, density = density
    )
end

@testset "V&V: remake — nu keyword updates nu only" begin
    prob = build_prob(; nu = 0.1, density = 1.0)
    prob2 = remake(prob; nu = 0.05)

    @test prob2.nu == 0.05
    @test prob2.density == 1.0          # unchanged
    @test prob2.mesh === prob.mesh      # shared
    @test prob2.algorithm === prob.algorithm
end

@testset "V&V: remake — density keyword updates density only" begin
    prob = build_prob(; nu = 0.1, density = 1.0)
    prob2 = remake(prob; density = 2.0)

    @test prob2.density == 2.0
    @test prob2.nu == 0.1                # unchanged
end

@testset "V&V: remake — both nu and density" begin
    prob = build_prob()
    prob2 = remake(prob; nu = 0.02, density = 3.0)

    @test prob2.nu == 0.02
    @test prob2.density == 3.0
end

@testset "V&V: remake — original problem unchanged (immutability)" begin
    prob = build_prob(; nu = 0.1, density = 1.0)
    prob2 = remake(prob; nu = 0.5, density = 5.0)

    # Original must retain its values.
    @test prob.nu == 0.1
    @test prob.density == 1.0
end

@testset "V&V: remake — algorithm swap" begin
    prob = build_prob()
    new_algo = SIMPLE(0.3, 0.1, 20, 1.0e-6)
    prob2 = remake(prob; algorithm = new_algo)

    @test prob2.algorithm === new_algo
    @test prob2.algorithm !== prob.algorithm
    @test prob2.nu == prob.nu
end

@testset "V&V: remake — chained remakes compose" begin
    prob = build_prob()
    prob2 = remake(prob; nu = 0.05)
    prob3 = remake(prob2; density = 4.0)

    @test prob3.nu == 0.05
    @test prob3.density == 4.0
end

@testset "V&V: remake — type preserved" begin
    prob = build_prob(; nu = 0.1, density = 1.0)
    prob2 = remake(prob; nu = 0.05)
    @test typeof(prob2) === typeof(prob)
    @test prob2 isa SteadyIncompressibleProblem  # build_prob uses SIMPLE (steady)
end

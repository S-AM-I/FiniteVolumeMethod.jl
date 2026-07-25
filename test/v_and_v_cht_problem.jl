# test/v_and_v_cht_problem.jl — ConjugateHeatTransferProblem V&V (v3.76)

using FiniteVolumeMethod
using FiniteVolumeMethod: ConjugateHeatTransferProblem
using FiniteVolumeMethod.Parabolic: DirichletBC
using StaticArrays
using Test

include("TestHelpers.jl")

function build_cht(; max_iter = 50, tol = 1.0e-4)
    fluid_mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    fluid_bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(), :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(0.1, 0.0)),
    )
    fluid_prob = SteadyIncompressibleProblem(
        fluid_mesh, fluid_bcs, SIMPLE(0.5, 0.2, 10, 1.0e-5);
        nu = 0.1, density = 1.0,
    )
    fluid_thermal = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.9,
        beta = 0.0, T_ref = 300.0, g = SVector(0.0, -9.81),
    )
    fluid_bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(300.0),
        :right => DirichletBC(300.0),
        :bottom => DirichletBC(400.0),
        :top => DirichletBC(300.0),
    )

    solid_mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    solid_thermal = SolidThermalProperties(; rho = 8000.0, Cp = 500.0, k = 15.0)
    solid_bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(300.0),
        :right => DirichletBC(300.0),
        :bottom => DirichletBC(300.0),
        :top => DirichletBC(400.0),
    )

    return ConjugateHeatTransferProblem(
        fluid_prob, fluid_thermal, fluid_bcs_T,
        solid_mesh, solid_thermal, solid_bcs_T;
        interface_fluid_patch = :bottom,
        interface_solid_patch = :top,
        max_coupling_iterations = max_iter,
        coupling_tolerance = tol,
    )
end

@testset "V&V: CHT problem — kwargs round-trip" begin
    cht = build_cht(; max_iter = 42, tol = 5.0e-5)
    @test cht.max_coupling_iterations == 42
    @test cht.coupling_tolerance == 5.0e-5
    @test cht.interface_fluid_patch === :bottom
    @test cht.interface_solid_patch === :top
end

@testset "V&V: CHT problem — references preserved" begin
    cht = build_cht()
    # Cross-checks: fields are populated and non-empty.
    @test cht.fluid_prob.nu == 0.1
    @test cht.fluid_thermal.k == 0.026
    @test cht.solid_thermal.k == 15.0
    @test length(cht.fluid_bcs_T) == 4
    @test length(cht.solid_bcs_T) == 4
end

@testset "V&V: CHT problem — default coupling params" begin
    # Default: max_iter = 50, tol = 1e-4.
    cht = build_cht()
    @test cht.max_coupling_iterations == 50
    @test cht.coupling_tolerance == 1.0e-4
end

# test/v_and_v_turbulence_inlet.jl — turbulence_inlet_bc V&V (v3.77)

using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using Test

include("TestHelpers.jl")

@testset "V&V: turbulence_inlet_bc — k-ε closed form" begin
    # k_inlet = 1.5 (U·I)², ε_inlet = C_μ^0.75 · k^1.5 / L
    model = StandardKEpsilon()
    U = 10.0
    I = 0.05
    L = 0.01

    bcs = turbulence_inlet_bc(model, U, I, L)
    k_expected = 1.5 * (U * I)^2
    eps_expected = model.C_mu^0.75 * k_expected^1.5 / L

    @test bcs[:k] isa DirichletBC
    @test bcs[:epsilon] isa DirichletBC
    @test isapprox(bcs[:k].value, k_expected; rtol = 1.0e-14)
    @test isapprox(bcs[:epsilon].value, eps_expected; rtol = 1.0e-14)
end

@testset "V&V: turbulence_inlet_bc — k-ω closed form" begin
    model = KOmega()
    U = 10.0
    I = 0.05
    L = 0.01

    bcs = turbulence_inlet_bc(model, U, I, L)
    k_expected = 1.5 * (U * I)^2
    omega_expected = sqrt(k_expected) / (0.09^0.25 * L)

    @test isapprox(bcs[:k].value, k_expected; rtol = 1.0e-14)
    @test isapprox(bcs[:omega].value, omega_expected; rtol = 1.0e-14)
end

@testset "V&V: turbulence_inlet_bc — SA closed form" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    model = SpalartAllmaras(mesh, Symbol[])
    U = 10.0
    I = 0.05
    L = 0.01

    bcs = turbulence_inlet_bc(model, U, I, L)
    nt_expected = 3 * I * U * L

    @test bcs[:nu_tilde] isa DirichletBC
    @test isapprox(bcs[:nu_tilde].value, nt_expected; rtol = 1.0e-14)
end

@testset "V&V: turbulence_inlet_bc — k-ε intensity² scaling" begin
    model = StandardKEpsilon()
    bcs_1 = turbulence_inlet_bc(model, 10.0, 0.05, 0.01)
    bcs_2 = turbulence_inlet_bc(model, 10.0, 0.1, 0.01)
    @test isapprox(bcs_2[:k].value / bcs_1[:k].value, 4.0; rtol = 1.0e-14)
end

@testset "V&V: turbulence_inlet_bc — wall BC returns Neumann(0) for k-ε" begin
    bcs = turbulence_wall_bc(StandardKEpsilon())
    @test bcs[:k] isa NeumannBC
    @test bcs[:k].value == 0.0
    @test bcs[:epsilon] isa NeumannBC
    @test bcs[:epsilon].value == 0.0
end

@testset "V&V: turbulence_inlet_bc — SA wall BC Dirichlet(0)" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    bcs = turbulence_wall_bc(SpalartAllmaras(mesh, Symbol[]))
    @test bcs[:nu_tilde] isa DirichletBC
    @test bcs[:nu_tilde].value == 0.0
end

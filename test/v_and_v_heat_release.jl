# test/v_and_v_heat_release.jl — compute_heat_release V&V (v3.71)

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

@testset "V&V: heat release — exothermic sign with ω_fuel < 0" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; heat_of_combustion = 5.0e7)

    # ω_fuel negative (consumption) ⇒ S_h = -ω_fuel·ΔH positive.
    omega = ntuple(i -> fill(i == 1 ? -0.02 : 0.0, nc), 3)
    S_h = compute_heat_release(omega, props)

    for c in 1:nc
        @test isapprox(S_h[c], 0.02 * 5.0e7; rtol = 1.0e-14)
        @test S_h[c] > 0.0
    end
end

@testset "V&V: heat release — zero rate ⇒ zero release" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties()
    omega = ntuple(_ -> zeros(nc), 3)
    S_h = compute_heat_release(omega, props)
    for c in 1:nc
        @test S_h[c] == 0.0
    end
end

@testset "V&V: heat release — linear in ω_fuel" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; heat_of_combustion = 4.0e7)

    omega_a = ntuple(i -> fill(i == 1 ? -0.01 : 0.0, nc), 3)
    omega_b = ntuple(i -> fill(i == 1 ? -0.02 : 0.0, nc), 3)
    S_a = compute_heat_release(omega_a, props)
    S_b = compute_heat_release(omega_b, props)

    for c in 1:nc
        @test isapprox(S_b[c] / S_a[c], 2.0; rtol = 1.0e-14)
    end
end

@testset "V&V: heat release — ΔH linear scaling" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    omega = ntuple(i -> fill(i == 1 ? -0.01 : 0.0, nc), 3)

    props_a = CombustionProperties(; heat_of_combustion = 5.0e7)
    props_b = CombustionProperties(; heat_of_combustion = 1.0e8)

    S_a = compute_heat_release(omega, props_a)
    S_b = compute_heat_release(omega, props_b)

    for c in 1:nc
        @test isapprox(S_b[c] / S_a[c], 2.0; rtol = 1.0e-14)
    end
end

@testset "V&V: heat release — spatially-varying ω_fuel" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = CombustionProperties(; heat_of_combustion = 5.0e7)

    omega_fuel = -[0.001 * c for c in 1:nc]
    omega = (omega_fuel, 4 .* omega_fuel, -5 .* omega_fuel)
    S_h = compute_heat_release(omega, props)

    for c in 1:nc
        @test isapprox(S_h[c], -omega_fuel[c] * 5.0e7; rtol = 1.0e-14)
        @test S_h[c] > 0.0
    end
end

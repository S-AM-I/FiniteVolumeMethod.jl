# test/experimental/fsi_coupling.jl — Aitken-accelerated partitioned FSI coupling.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: AitkenRelaxation, FSIInterface, interface_residual_norm, update_aitken!
using Test
using LinearAlgebra: norm
using StaticArrays: SVector

@testset "Aitken relaxation adapts toward optimal ω" begin
    relax = AitkenRelaxation(; omega0 = 0.5)

    # Feed a declining residual sequence: ω should increase toward 1.
    r1 = [1.0, 1.0]
    r2 = [0.3, 0.3]
    r3 = [0.05, 0.05]

    ω1 = update_aitken!(relax, r1)
    ω2 = update_aitken!(relax, r2)
    ω3 = update_aitken!(relax, r3)

    @test ω1 == 0.5                 # first call keeps initial
    @test relax.omega_min <= ω2 <= relax.omega_max
    @test relax.omega_min <= ω3 <= relax.omega_max
end

@testset "FSIInterface has matching fluid/solid face lists" begin
    iface = FSIInterface{2, Float64}([10, 20, 30], [15, 25, 35])
    @test length(iface.fluid_face_indices) == 3
    @test length(iface.solid_face_indices) == 3
    @test all(d -> d == zero(SVector{2, Float64}), iface.displacement)
    @test all(t -> t == zero(SVector{2, Float64}), iface.traction)
end

@testset "interface_residual_norm matches L2 update" begin
    d_old = [SVector(0.0, 0.0), SVector(0.0, 0.0)]
    d_new = [SVector(0.3, 0.4), SVector(0.0, 0.0)]
    r = interface_residual_norm(d_new, d_old)
    @test r ≈ 0.5 atol = 1.0e-12      # sqrt(0.09 + 0.16) = 0.5
end

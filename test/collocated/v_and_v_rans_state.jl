# test/v_and_v_rans_state.jl — RANSTurbulenceState container V&V (v3.66)
#
# Sixth convergence-verified benchmark for `turbulence_rans`,
# joining k-ε DHIT (v3.18), k-ε log-layer (v3.23), k-ω (v3.38),
# Spalart-Allmaras (v3.44), and wall functions (v3.54). Covers
# the `RANSTurbulenceState` container and dispatch across all
# RANS models.
#
# Six invariants verified.

using FiniteVolumeMethod
using FiniteVolumeMethod: AbstractRANSModel, RANSTurbulenceState
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: RANSState — k-ε field names and count" begin
    model = StandardKEpsilon()
    @test FiniteVolumeMethod.n_turbulence_fields(model) == 2
    @test FiniteVolumeMethod.turbulence_field_names(model) == (:k, :epsilon)
end

@testset "V&V: RANSState — k-ω field names and count" begin
    model = KOmega()
    @test FiniteVolumeMethod.n_turbulence_fields(model) == 2
    @test FiniteVolumeMethod.turbulence_field_names(model) == (:k, :omega)
end

@testset "V&V: RANSState — Spalart-Allmaras field count" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    model = SpalartAllmaras(mesh, Symbol[])
    @test FiniteVolumeMethod.n_turbulence_fields(model) == 1
    @test FiniteVolumeMethod.turbulence_field_names(model) == (:nu_tilde,)
end

@testset "V&V: RANSState — default zero + size match" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    model = StandardKEpsilon()
    state = RANSTurbulenceState(model, mesh)   # no initial values

    @test length(state.nu_t) == nc
    @test length(state.fields[:k].internal) == nc
    @test length(state.fields[:epsilon].internal) == nc

    # Default init values are 1e-6 (not exactly zero — avoid div-by-0).
    for v in state.fields[:k].internal
        @test v == 1.0e-6
    end
    for v in state.fields[:epsilon].internal
        @test v == 1.0e-6
    end
end

@testset "V&V: RANSState — custom init kwargs round-trip" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    model = StandardKEpsilon()
    state = RANSTurbulenceState(model, mesh; k = 0.5, epsilon = 0.05)

    for v in state.fields[:k].internal
        @test v == 0.5
    end
    for v in state.fields[:epsilon].internal
        @test v == 0.05
    end
end

@testset "V&V: RANSState — AbstractRANSModel dispatch (KOmega, SA)" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    ko = KOmega()
    sa = SpalartAllmaras(mesh, Symbol[])

    @test ko isa FiniteVolumeMethod.AbstractRANSModel
    @test sa isa FiniteVolumeMethod.AbstractRANSModel
end

@testset "V&V: RANSState — k-ω state has omega not epsilon" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    model = KOmega()
    state = RANSTurbulenceState(model, mesh; k = 1.0, omega = 10.0)

    @test haskey(state.fields, :k)
    @test haskey(state.fields, :omega)
    @test !haskey(state.fields, :epsilon)

    for v in state.fields[:omega].internal
        @test v == 10.0
    end
end

# test/v_and_v_residual_indicator.jl — residual-based error indicator V&V.

using FiniteVolumeMethod
using FiniteVolumeMethod: CollocatedEquation, residual_error_indicator
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: residual indicator — exact solution ⇒ r ≈ 0" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    for c in 1:nc
        FiniteVolumeMethod.add_diag!(eq, c, 1.0)
        eq.b[c] = Float64(c)
    end
    u = copy(eq.b)
    r = residual_error_indicator(eq, u, mesh)
    @test length(r) == nc
    for v in r
        @test v < 1.0e-12
    end
end

@testset "V&V: residual indicator — ‖r‖ ≥ 0 everywhere" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    for c in 1:nc
        FiniteVolumeMethod.add_diag!(eq, c, 2.0)
        eq.b[c] = Float64(c)
    end
    u = zeros(nc)
    r = residual_error_indicator(eq, u, mesh)
    for v in r
        @test v >= 0.0
    end
end

@testset "V&V: residual indicator — perturbation spikes at the perturbed cell" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    for c in 1:nc
        FiniteVolumeMethod.add_diag!(eq, c, 1.0)
        eq.b[c] = 1.0
    end
    u = ones(nc)
    u[7] += 1.0
    r = residual_error_indicator(eq, u, mesh)
    @test r[7] > 0.0
    for c in 1:nc
        if c != 7
            @test r[c] < 1.0e-10
        end
    end
end

# test/v_and_v_clip_alpha.jl — clip_alpha! V&V (v3.79)

using FiniteVolumeMethod
using FiniteVolumeMethod: clip_alpha!
using Test

include("TestHelpers.jl")

@testset "V&V: clip_alpha! — values in [0,1] unchanged" begin
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        alpha.internal[c] = 0.5
    end
    clip_alpha!(alpha, mesh)
    for c in 1:nc
        @test alpha.internal[c] == 0.5
    end
end

@testset "V&V: clip_alpha! — negative values clipped to 0" begin
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    alpha = CollocatedScalarField(:alpha, mesh; value = -0.1)
    clip_alpha!(alpha, mesh)
    for c in 1:length(mesh.cell_volumes)
        @test alpha.internal[c] == 0.0
    end
end

@testset "V&V: clip_alpha! — values > 1 clipped to 1" begin
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    alpha = CollocatedScalarField(:alpha, mesh; value = 1.5)
    clip_alpha!(alpha, mesh)
    for c in 1:length(mesh.cell_volumes)
        @test alpha.internal[c] == 1.0
    end
end

@testset "V&V: clip_alpha! — boundedness + total-α·V conservation" begin
    # clip_alpha! clamps AND redistributes to preserve total α·V.
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        alpha.internal[c] = c % 3 == 0 ? 1.3 : (c % 3 == 1 ? -0.2 : 0.4)
    end
    total_before = sum(alpha.internal[c] * mesh.cell_volumes[c] for c in 1:nc)

    clip_alpha!(alpha, mesh)
    total_after = sum(alpha.internal[c] * mesh.cell_volumes[c] for c in 1:nc)

    for c in 1:nc
        @test -1.0e-12 <= alpha.internal[c] <= 1.0 + 1.0e-12
    end
    @test isapprox(total_after, total_before; rtol = 1.0e-10)
end

@testset "V&V: clip_alpha! — boundary endpoints (0 and 1) pass through" begin
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    alpha_0 = CollocatedScalarField(:alpha, mesh; value = 0.0)
    alpha_1 = CollocatedScalarField(:alpha, mesh; value = 1.0)
    clip_alpha!(alpha_0, mesh)
    clip_alpha!(alpha_1, mesh)
    for c in 1:length(mesh.cell_volumes)
        @test alpha_0.internal[c] == 0.0
        @test alpha_1.internal[c] == 1.0
    end
end

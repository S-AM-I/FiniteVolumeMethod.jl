# test/v_and_v_solid_body_motion.jl — SolidBodyMotion V&V (v3.67)

using FiniteVolumeMethod
using FiniteVolumeMethod: MeshMotionState, compute_displacement!
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: SolidBody — constant displacement-function applied uniformly" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    d_fixed = SVector(0.1, -0.05)
    motion = SolidBodyMotion{2, Float64}(_ -> d_fixed)

    compute_displacement!(ms, motion, mesh, 0.0)

    for c in 1:length(mesh.cell_volumes)
        @test ms.displacement[c] == d_fixed
    end
end

@testset "V&V: SolidBody — time-dependent displacement function" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    motion = SolidBodyMotion{2, Float64}(t -> SVector(0.1 * t, -0.05 * t))

    compute_displacement!(ms, motion, mesh, 2.0)

    expected = SVector(0.2, -0.1)
    for c in 1:length(mesh.cell_volumes)
        @test ms.displacement[c] == expected
    end
end

@testset "V&V: SolidBody — zero displacement function ⇒ no motion" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    motion = SolidBodyMotion{2, Float64}(_ -> SVector(0.0, 0.0))

    compute_displacement!(ms, motion, mesh, 1.0)

    for c in 1:length(mesh.cell_volumes)
        @test ms.displacement[c] == SVector(0.0, 0.0)
    end
end

@testset "V&V: SolidBody — t linear scaling" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    ms_1 = MeshMotionState(mesh)
    ms_2 = MeshMotionState(mesh)
    motion = SolidBodyMotion{2, Float64}(t -> SVector(0.05 * t, 0.0))

    compute_displacement!(ms_1, motion, mesh, 1.0)
    compute_displacement!(ms_2, motion, mesh, 2.0)

    for c in 1:length(mesh.cell_volumes)
        @test ms_2.displacement[c][1] ≈ 2 * ms_1.displacement[c][1] atol = 1.0e-14
    end
end

@testset "V&V: SolidBody — successive calls overwrite prior displacement" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    motion1 = SolidBodyMotion{2, Float64}(_ -> SVector(0.1, 0.0))
    motion2 = SolidBodyMotion{2, Float64}(_ -> SVector(0.0, 0.2))

    compute_displacement!(ms, motion1, mesh, 0.0)
    compute_displacement!(ms, motion2, mesh, 0.0)

    # Second call must overwrite, not accumulate.
    for c in 1:length(mesh.cell_volumes)
        @test ms.displacement[c] == SVector(0.0, 0.2)
    end
end

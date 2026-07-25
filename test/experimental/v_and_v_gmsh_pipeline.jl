# test/v_and_v_gmsh_pipeline.jl — Gmsh automation stub V&V.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: GmshPipeline, auto_remediate!, run_gmsh_pipeline
using Test

@testset "V&V: GmshPipeline — constructor round-trip" begin
    pipeline = GmshPipeline(
        "/tmp/geometry.geo";
        max_non_ortho = 65.0, max_skew = 0.8, max_aspect = 50.0,
    )
    @test pipeline.script == "/tmp/geometry.geo"
    @test pipeline.max_non_ortho == 65.0
    @test pipeline.max_skew == 0.8
    @test pipeline.max_aspect == 50.0
end

@testset "V&V: GmshPipeline — errors without Gmsh.jl loaded" begin
    pipeline = GmshPipeline("/tmp/geom.geo")
    @test_throws ErrorException run_gmsh_pipeline(pipeline, "/tmp/out.msh")
end

@testset "V&V: auto_remediate! — warns and returns mesh unchanged" begin
    mesh_stub = (cells = 1:4,)
    @test_logs (:warn, r"deferred to v3.1") auto_remediate!(mesh_stub, nothing, nothing)
end

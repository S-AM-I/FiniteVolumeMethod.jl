# test/v_and_v_metis_partition_stub.jl
#
# V&V: `partition_mesh_metis` must error helpfully when Metis.jl is not
# loaded. The real implementation lives in `ext/FVMMetisExt.jl` and is
# only activated by `using Metis`.
#
# Invariants:
# 1. Calling `partition_mesh_metis` without Metis.jl errors.
# 2. The error message mentions "Metis.jl" so users can diagnose the
#    missing extension without reading source.
# 3. `partition_rcb` remains available as a dependency-free fallback on
#    the same mesh (sanity check — we don't regress the non-Metis path).

using Test
using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: partition_mesh_metis, partition_rcb

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

# If the main thread hasn't yet wired the metis_stub.jl include into the
# layer graph, load it directly so this file can exercise the stub in
# isolation. Once main lands the include, `partition_mesh_metis` is
# visible on `FiniteVolumeMethod` and this fallback is a no-op.
if !isdefined(FiniteVolumeMethod, :partition_mesh_metis)
    stub_path = joinpath(dirname(pathof(FiniteVolumeMethod)), "parallel", "metis_stub.jl")
    @eval FiniteVolumeMethod Base.include($FiniteVolumeMethod, $stub_path)
end

@testset "V&V Metis partition stub: errors without Metis.jl" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)

    # The stub must raise an ErrorException (the concrete type of
    # `error(msg)`). We guard against the case where Metis.jl *is*
    # loaded in this Julia session — if it is, the ext has already
    # overridden the stub and this test is a no-op.
    metis_loaded = any(nameof(m) === :Metis for m in Base.loaded_modules_array())

    if metis_loaded
        @test_skip "Metis.jl loaded in this session — stub path cannot be exercised"
    else
        err = try
            FiniteVolumeMethod.partition_mesh_metis(mesh, 2)
            nothing
        catch e
            e
        end

        @test err !== nothing
        @test err isa ErrorException
        @test occursin("Metis.jl", err.msg)
    end
end

@testset "V&V Metis partition stub: dep-free fallback remains available" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    # Even without Metis.jl, the RCB fallback partitioner should work.
    parts = FiniteVolumeMethod.partition_rcb(mesh, 2)
    @test length(parts) == length(mesh.cell_volumes)
    @test all(0 .<= parts .<= 1)
    @test count(==(0), parts) + count(==(1), parts) == length(parts)
end

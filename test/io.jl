using Test
using Dates
using FiniteVolumeMethod
using FiniteVolumeMethod: CheckpointManager, Diagnostic, IntegralMonitor, Mesh1D, OutputManager, OutputSchedule, OutputTarget, Probe, Provenance, SimulationConfig, compute_integral, conservation_summary, ensure_extension, ensure_output_dirs, flux_inout, generate_mesh_1d, load_checkpoint, load_model_package, print_scientific, read_solution_hdf5, safe_filename, sample_probe, save_checkpoint, save_model_package, stringify_keys, validate_schedule, volume_integral, write_line_vtk, write_solution_hdf5, write_structured_vtk_3d
using HDF5
using JLD2
using StaticArrays

@testset "I/O Module" begin
    @testset "OutputSchedule" begin
        sched = OutputSchedule(0.0, 1.0, 0.1)
        @test sched isa OutputSchedule
        @test length(sched.write_times) == 11  # 0.0:0.1:1.0
        @test sched.write_every == 0
    end

    @testset "OutputManager" begin
        sched = OutputSchedule(0.0, 1.0, 0.1)
        target = OutputTarget(:file, "/tmp/test", :vtk)
        prov = Provenance("sim1", Dates.now(), nothing, "1.0", "user", "host")
        mgr = OutputManager(sched, [target], Diagnostic[], 0.0, prov)
        @test mgr isa OutputManager
        @test validate_schedule(sched) === sched
    end

    @testset "VTK Output" begin
        mesh = generate_mesh_1d(10, 1.0)
        data = collect(1.0:10.0)
        tmpfile = tempname() * ".vtk"
        write_line_vtk(tmpfile, mesh, data, "test_field")
        @test isfile(tmpfile)
        rm(tmpfile)
    end

    @testset "VTK Output (raw coords)" begin
        xcoords = collect(0.0:0.1:1.0)
        scalars = sin.(xcoords)
        tmpfile = tempname() * ".vtk"
        write_line_vtk(tmpfile, xcoords, scalars; label = "sin_x")
        @test isfile(tmpfile)
        content = read(tmpfile, String)
        @test occursin("SCALARS sin_x float 1", content)
        rm(tmpfile)
    end

    @testset "VTK Output is deterministic" begin
        mesh = generate_mesh_1d(10, 1.0)
        data = collect(1.0:10.0)
        tmpfile_a = tempname() * ".vtk"
        tmpfile_b = tempname() * ".vtk"
        try
            write_line_vtk(tmpfile_a, mesh, data, "test_field")
            write_line_vtk(tmpfile_b, mesh, data, "test_field")
            @test read(tmpfile_a, String) == read(tmpfile_b, String)
        finally
            rm(tmpfile_a; force = true)
            rm(tmpfile_b; force = true)
        end
    end

    @testset "VTK 3D structured (WriteVTK extension)" begin
        using WriteVTK
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 4, 5, 6)
        data = rand(4, 5, 6)
        dir = mktempdir()
        base = joinpath(dir, "smoke3d")
        out = write_structured_vtk_3d(base, mesh, data; label = "rho")
        @test out == base
        # WriteVTK picks the concrete flavor (.vti for uniform axes); assert
        # exactly one smoke3d.vt* file was produced.
        written = filter(startswith("smoke3d.vt"), readdir(dir))
        @test length(written) == 1
        @test_throws Exception write_structured_vtk_3d(base, mesh, rand(2, 2, 2))
    end

    @testset "Volume Integral" begin
        mesh = generate_mesh_1d(10, 1.0)
        field = ones(10)
        integral = volume_integral(mesh, field)
        @test integral ≈ 1.0
    end

    @testset "Conservation Summary" begin
        mesh = generate_mesh_1d(10, 1.0)
        field = collect(1.0:10.0)
        summary = conservation_summary(mesh, field)
        @test summary.min == 1.0
        @test summary.max == 10.0
        @test summary.total > 0
    end

    @testset "Utils" begin
        @test safe_filename("test/file:name") != "test/file:name"
        @test safe_filename("test/file:name") == "test_file_name"
        @test ensure_extension("file", ".vtk") == "file.vtk"
        @test ensure_extension("file.vtk", ".vtk") == "file.vtk"
        @test ensure_extension("file", "vtk") == "file.vtk"
    end

    @testset "Output Dirs" begin
        tmpdir = mktempdir()
        dirs = ensure_output_dirs(tmpdir)
        @test isdir(dirs.data)
        @test isdir(dirs.plots)
        @test isdir(dirs.reports)
        @test isdir(dirs.vtk)
    end

    @testset "Stringify Keys" begin
        d = Dict(:a => 1, :b => Dict(:c => 2))
        s = stringify_keys(d)
        @test s isa Dict{String, Any}
        @test s["a"] == 1
        @test s["b"]["c"] == 2
    end

    @testset "CheckpointManager" begin
        cm = CheckpointManager(; interval = 50, dir = "cp", keep_recent = 5)
        @test cm isa CheckpointManager
        @test cm.interval == 50
        @test cm.dir == "cp"
        @test cm.keep_recent == 5
    end

    @testset "Checkpoint round-trip (JLD2 extension)" begin
        tmpdir = mktempdir()
        cm = CheckpointManager(; interval = 10, dir = tmpdir, keep_recent = 2)
        state = [SVector(1.0, 0.0, 2.5), SVector(0.9, 0.1, 2.4)]
        metadata = Dict("case" => "unit-test", "step_kind" => "restartable")

        first_path = save_checkpoint(cm, state, 10; metadata)
        loaded_first = load_checkpoint(cm, 10)

        @test isfile(first_path)
        @test loaded_first["state"] == state
        @test loaded_first["step"] == 10
        @test loaded_first["metadata"]["case"] == "unit-test"
        @test loaded_first["metadata"]["step_kind"] == "restartable"
        @test loaded_first["metadata"]["checkpoint_format_version"] == 1
        @test loaded_first["metadata"]["checkpoint_writer"] == "FiniteVolumeMethod.FVMCheckpointExt"

        second_path = save_checkpoint(cm, state, 20; metadata)
        third_path = save_checkpoint(cm, state, 30; metadata)

        @test !isfile(first_path)
        @test isfile(second_path)
        @test isfile(third_path)
    end

    @testset "HDF5 solution round-trip" begin
        mesh = generate_mesh_1d(5, 1.0)
        solution = collect(range(1.0, 2.0; length = 5))
        fields = Dict("temperature" => solution .^ 2, "density" => fill(1.0, 5))
        metadata = Dict("case" => "roundtrip", "schema_version" => 3)
        tmpfile = tempname() * ".h5"

        try
            write_solution_hdf5(tmpfile, solution, mesh; fields, metadata)
            @test isfile(tmpfile)

            loaded = read_solution_hdf5(tmpfile)
            loaded_solution_only = read_solution_hdf5(tmpfile; load_fields = false)

            @test loaded["solution"] ≈ solution
            @test loaded["fields"]["temperature"] ≈ solution .^ 2
            @test loaded["fields"]["density"] == fill(1.0, 5)
            @test !haskey(loaded_solution_only, "fields")
            @test loaded["metadata"]["case"] == "roundtrip"
            @test loaded["metadata"]["schema_version"] == 3
            @test loaded["metadata"]["solution_format_version"] == 1
            @test loaded["metadata"]["mesh_type"] == "Mesh1D"
            @test loaded["metadata"]["solution_writer"] == "FiniteVolumeMethod.FVMHdf5Ext"
        finally
            rm(tmpfile; force = true)
        end
    end

    @testset "InSitu Monitors" begin
        mesh = generate_mesh_1d(10, 1.0)
        probe = Probe(mesh, [0.55], "temperature")
        @test probe isa Probe
        @test probe.cell_index > 0

        u = collect(1.0:10.0)
        val = sample_probe(probe, mesh, u)
        @test !isnan(val)

        monitor = IntegralMonitor("temperature"; region = :volume)
        integral = compute_integral(monitor, mesh, ones(10))
        @test integral ≈ 1.0
    end

    @testset "Registry (round-trip)" begin
        tmpdir = mktempdir()
        mesh = generate_mesh_1d(5, 1.0)
        physics = Dict("diffusion" => Dict("gamma" => "1.0"))
        ic = collect(1.0:5.0)
        save_model_package(mesh, physics, ic, tmpdir)
        mesh_meta, phys_loaded, ic_loaded = load_model_package(tmpdir)
        @test mesh_meta["schema_version"] == 1
        @test haskey(mesh_meta, "type")
        @test phys_loaded["diffusion"]["gamma"] == "1.0"
        @test ic_loaded ≈ ic
    end

    @testset "Registry serialization is deterministic" begin
        tmpdir_a = mktempdir()
        tmpdir_b = mktempdir()
        mesh = generate_mesh_1d(5, 1.0)
        physics = Dict("diffusion" => Dict("gamma" => "1.0", "solver" => "cg"))
        ic = collect(1.0:5.0)
        save_model_package(mesh, physics, ic, tmpdir_a)
        save_model_package(mesh, physics, ic, tmpdir_b)
        @test read(joinpath(tmpdir_a, "mesh_meta.toml"), String) == read(joinpath(tmpdir_b, "mesh_meta.toml"), String)
        @test read(joinpath(tmpdir_a, "physics.toml"), String) == read(joinpath(tmpdir_b, "physics.toml"), String)
        @test read(joinpath(tmpdir_a, "ic.dat"), String) == read(joinpath(tmpdir_b, "ic.dat"), String)
    end

    @testset "Flux Inout" begin
        fluxes = Dict(
            (:phi, :left) => -1.5,
            (:phi, :right) => 2.0,
        )
        summary = flux_inout(fluxes)
        @test summary[:phi][:inflow] == -1.5
        @test summary[:phi][:outflow] == 2.0
    end

    @testset "Print Scientific" begin
        s = print_scientific(1.234e-5, 3)
        @test occursin("1.234e-05", s) || occursin("1.234e-5", s)
    end

    @testset "SimulationConfig" begin
        cfg = SimulationConfig(Dict(:key => "value"))
        @test cfg isa SimulationConfig
        @test cfg.options[:key] == "value"
    end
end

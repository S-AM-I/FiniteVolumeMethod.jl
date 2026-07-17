using FiniteVolumeMethod
using FiniteVolumeMethod: MeshQualityReport, Node3D, UnstructuredFVMMesh, check_mesh_quality, print_mesh_quality, read_openfoam_polymesh, volume_hex, volume_prism, volume_pyramid, volume_tet
using Test
using LinearAlgebra

# ── Mesh builder (shared helper) ─────────────────────────────────────
include("TestHelpers.jl")

# ── OpenFOAM helper: write a minimal 1-hex-cell polyMesh ──

function write_openfoam_unit_cube(case_dir::AbstractString)
    mesh_dir = joinpath(case_dir, "constant", "polyMesh")
    mkpath(mesh_dir)

    # points — unit cube, 8 vertices
    open(joinpath(mesh_dir, "points"), "w") do io
        println(
            io, """FoamFile
            {
                version     2.0;
                format      ascii;
                class       vectorField;
                object      points;
            }

            8
            (
            (0 0 0)
            (1 0 0)
            (1 1 0)
            (0 1 0)
            (0 0 1)
            (1 0 1)
            (1 1 1)
            (0 1 1)
            )"""
        )
    end

    # faces — 6 quad faces of the cube (0-indexed vertices)
    open(joinpath(mesh_dir, "faces"), "w") do io
        println(
            io, """FoamFile
            {
                version     2.0;
                format      ascii;
                class       faceList;
                object      faces;
            }

            6
            (
            4(0 3 2 1)
            4(4 5 6 7)
            4(0 1 5 4)
            4(2 3 7 6)
            4(0 4 7 3)
            4(1 2 6 5)
            )"""
        )
    end

    # owner — all 6 faces owned by cell 0
    open(joinpath(mesh_dir, "owner"), "w") do io
        println(
            io, """FoamFile
            {
                version     2.0;
                format      ascii;
                class       labelList;
                object      owner;
            }

            6
            (
            0
            0
            0
            0
            0
            0
            )"""
        )
    end

    # neighbour — no internal faces, empty list
    open(joinpath(mesh_dir, "neighbour"), "w") do io
        println(
            io, """FoamFile
            {
                version     2.0;
                format      ascii;
                class       labelList;
                object      neighbour;
            }

            0
            (
            )"""
        )
    end

    # boundary — 6 patches, one face each
    open(joinpath(mesh_dir, "boundary"), "w") do io
        println(
            io, """FoamFile
            {
                version     2.0;
                format      ascii;
                class       polyBoundaryMesh;
                object      boundary;
            }

            6
            (
            bottom
            {
                type            patch;
                nFaces          1;
                startFace       0;
            }
            top
            {
                type            patch;
                nFaces          1;
                startFace       1;
            }
            front
            {
                type            patch;
                nFaces          1;
                startFace       2;
            }
            back
            {
                type            patch;
                nFaces          1;
                startFace       3;
            }
            left
            {
                type            patch;
                nFaces          1;
                startFace       4;
            }
            right
            {
                type            patch;
                nFaces          1;
                startFace       5;
            }
            )"""
        )
    end

    return case_dir
end

# ── Tests ──────────────────────────────────────────────────────────────

@testset "Polyhedral Mesh I/O" begin

    # ── 1. volume_tet ─────────────────────────────────────────────────
    @testset "volume_tet" begin
        nodes = [
            Node3D(0.0, 0.0, 0.0),
            Node3D(1.0, 0.0, 0.0),
            Node3D(0.0, 1.0, 0.0),
            Node3D(0.0, 0.0, 1.0),
        ]
        V = volume_tet(nodes)
        @test V ≈ 1.0 / 6.0 atol = 1.0e-10
    end

    # ── 2. volume_hex ─────────────────────────────────────────────────
    @testset "volume_hex" begin
        nodes = [
            Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0),
            Node3D(1.0, 1.0, 0.0), Node3D(0.0, 1.0, 0.0),
            Node3D(0.0, 0.0, 1.0), Node3D(1.0, 0.0, 1.0),
            Node3D(1.0, 1.0, 1.0), Node3D(0.0, 1.0, 1.0),
        ]
        V = volume_hex(nodes)
        @test V ≈ 1.0 atol = 1.0e-10
    end

    # ── 3. volume_prism ───────────────────────────────────────────────
    @testset "volume_prism" begin
        # Triangular prism: triangle (0,0,0),(1,0,0),(0,1,0) extruded to z=1
        # Volume = base_area * height = 0.5 * 1.0 = 0.5
        nodes = [
            Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0), Node3D(0.0, 1.0, 0.0),
            Node3D(0.0, 0.0, 1.0), Node3D(1.0, 0.0, 1.0), Node3D(0.0, 1.0, 1.0),
        ]
        V = volume_prism(nodes)
        @test V ≈ 0.5 atol = 1.0e-10
    end

    # ── 4. volume_pyramid ─────────────────────────────────────────────
    @testset "volume_pyramid" begin
        # Square pyramid: base (0,0,0),(1,0,0),(1,1,0),(0,1,0), apex (0.5,0.5,1)
        # Volume = (1/3) * base_area * height = 1/3 * 1 * 1 = 1/3
        nodes = [
            Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0),
            Node3D(1.0, 1.0, 0.0), Node3D(0.0, 1.0, 0.0),
            Node3D(0.5, 0.5, 1.0),
        ]
        V = volume_pyramid(nodes)
        @test V ≈ 1.0 / 3.0 atol = 1.0e-10
    end

    # ── 5. OpenFOAM reader ────────────────────────────────────────────
    @testset "read_openfoam_polymesh" begin
        mktempdir() do case_dir
            write_openfoam_unit_cube(case_dir)
            mesh = read_openfoam_polymesh(case_dir)

            @test mesh isa UnstructuredFVMMesh{3, Float64}

            # 1 cell, 6 faces
            ncells_mesh = length(mesh.cell_volumes)
            nfaces_mesh = size(mesh.face_cells, 2)
            @test ncells_mesh == 1
            @test nfaces_mesh == 6

            # Cell volume should be 1.0 for a unit cube
            @test mesh.cell_volumes[1] ≈ 1.0 atol = 1.0e-10

            # All faces should be boundary (neighbour = 0)
            for f in 1:nfaces_mesh
                @test mesh.face_cells[2, f] == 0
            end

            # All faces should be owned by cell 1
            for f in 1:nfaces_mesh
                @test mesh.face_cells[1, f] == 1
            end

            # Face tags should include patch names
            @test mesh.face_tags !== nothing
            tag_set = Set(mesh.face_tags)
            @test :bottom in tag_set
            @test :top in tag_set

            # Cell center should be at (0.5, 0.5, 0.5)
            @test mesh.cell_centers[1, 1] ≈ 0.5 atol = 1.0e-10
            @test mesh.cell_centers[2, 1] ≈ 0.5 atol = 1.0e-10
            @test mesh.cell_centers[3, 1] ≈ 0.5 atol = 1.0e-10
        end
    end

    # ── 6. check_mesh_quality on Cartesian mesh ───────────────────────
    @testset "check_mesh_quality on Cartesian mesh" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        report = check_mesh_quality(mesh)

        @test report isa MeshQualityReport{Float64}

        # Orthogonal Cartesian mesh: non-orthogonality should be ~0
        @test report.max_non_orthogonality ≈ 0.0 atol = 1.0e-10
        @test report.avg_non_orthogonality ≈ 0.0 atol = 1.0e-10

        # Skewness should be ~0 for uniform Cartesian
        @test report.max_skewness ≈ 0.0 atol = 1.0e-10
        @test report.avg_skewness ≈ 0.0 atol = 1.0e-10

        # Aspect ratio should be defined for all cells
        @test length(report.aspect_ratio) == 16
    end

    # ── 7. print_mesh_quality ─────────────────────────────────────────
    @testset "print_mesh_quality" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        report = check_mesh_quality(mesh)

        buf = IOBuffer()
        print_mesh_quality(report; io = buf)
        output = String(take!(buf))

        @test contains(output, "Mesh Quality Report")
        @test contains(output, "Non-orthogonality")
        @test contains(output, "Skewness")
        @test contains(output, "Aspect ratio")
        @test contains(output, "Status:")
    end
end

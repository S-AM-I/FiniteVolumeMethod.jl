# test/v_and_v_stl_reader.jl — minimal STL reader round-trip (v3.1).

using FiniteVolumeMethod
using FiniteVolumeMethod: read_stl_ascii, write_stl_ascii
using StaticArrays
using LinearAlgebra: norm
using Test

function _unit_cube_stl(path::AbstractString)
    # 8 unit-cube vertices, 12 triangles (2 per face), outward normals.
    corners = [
        SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0), SVector(1.0, 1.0, 0.0),
        SVector(0.0, 0.0, 1.0), SVector(1.0, 0.0, 1.0),
        SVector(0.0, 1.0, 1.0), SVector(1.0, 1.0, 1.0),
    ]
    # (v_i, v_j, v_k, normal)
    tri_table = [
        # bottom (z = 0), normal (0, 0, -1)
        (1, 3, 2, SVector(0.0, 0.0, -1.0)),
        (2, 3, 4, SVector(0.0, 0.0, -1.0)),
        # top (z = 1), normal (0, 0, 1)
        (5, 6, 7, SVector(0.0, 0.0, 1.0)),
        (6, 8, 7, SVector(0.0, 0.0, 1.0)),
        # y = 0 face, normal (0, -1, 0)
        (1, 2, 5, SVector(0.0, -1.0, 0.0)),
        (2, 6, 5, SVector(0.0, -1.0, 0.0)),
        # y = 1 face, normal (0, 1, 0)
        (3, 7, 4, SVector(0.0, 1.0, 0.0)),
        (4, 7, 8, SVector(0.0, 1.0, 0.0)),
        # x = 0 face, normal (-1, 0, 0)
        (1, 5, 3, SVector(-1.0, 0.0, 0.0)),
        (3, 5, 7, SVector(-1.0, 0.0, 0.0)),
        # x = 1 face, normal (1, 0, 0)
        (2, 4, 6, SVector(1.0, 0.0, 0.0)),
        (4, 8, 6, SVector(1.0, 0.0, 0.0)),
    ]
    faces = [(t[1], t[2], t[3]) for t in tri_table]
    normals = [t[4] for t in tri_table]
    return write_stl_ascii(path, corners, faces, normals)
end

@testset "V&V: STL reader — unit-cube round-trip" begin
    path = joinpath(mktempdir(), "unit_cube.stl")
    _unit_cube_stl(path)
    vertices, faces, normals = read_stl_ascii(path)

    @test length(vertices) == 8  # dedup across 36 raw vertex entries
    @test length(faces) == 12
    @test length(normals) == 12

    # All face indices reference a valid vertex.
    for face in faces
        for k in 1:3
            @test 0 < face[k] <= length(vertices)
        end
    end

    # All normals unit length (written exactly, so tolerances are loose).
    for n in normals
        @test abs(norm(n) - 1.0) < 1.0e-9
    end

    # Unique vertex set contains the 8 cube corners.
    @test Set(Tuple.(vertices)) == Set(
        [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0),
        ]
    )
end

@testset "V&V: STL reader — error handling" begin
    # Missing file.
    @test_throws ArgumentError read_stl_ascii(joinpath(mktempdir(), "missing.stl"))

    # Empty file.
    empty_path = joinpath(mktempdir(), "empty.stl")
    open(empty_path, "w") do io
    end
    @test_throws ArgumentError read_stl_ascii(empty_path)

    # Header without facets.
    headerless_path = joinpath(mktempdir(), "headerless.stl")
    open(headerless_path, "w") do io
        println(io, "this is not an stl")
    end
    @test_throws ArgumentError read_stl_ascii(headerless_path)

    # `solid` header but no `endsolid`.
    truncated_path = joinpath(mktempdir(), "truncated.stl")
    open(truncated_path, "w") do io
        println(io, "solid only_header")
    end
    @test_throws ArgumentError read_stl_ascii(truncated_path)

    # Malformed facet block (non-numeric coords).
    bad_numeric_path = joinpath(mktempdir(), "bad_numeric.stl")
    open(bad_numeric_path, "w") do io
        println(io, "solid bad")
        println(io, "facet normal foo 0 0")
        println(io, "outer loop")
        println(io, "vertex 0 0 0")
        println(io, "vertex 1 0 0")
        println(io, "vertex 0 1 0")
        println(io, "endloop")
        println(io, "endfacet")
        println(io, "endsolid bad")
    end
    @test_throws ArgumentError read_stl_ascii(bad_numeric_path)
end

# mesh_generation/stl_reader.jl — minimal ASCII STL reader (v3.1).
#
# Parses the standard OpenFOAM/stereolithography ASCII STL grammar:
#
#     solid <name>
#       facet normal nx ny nz
#         outer loop
#           vertex vx vy vz
#           vertex vx vy vz
#           vertex vx vy vz
#         endloop
#       endfacet
#       ...
#     endsolid <name>
#
# Vertices are deduplicated via a hash-table keyed on the raw floating
# point triple, so downstream mesh code sees a (vertices, faces, normals)
# triple where `faces::Vector{NTuple{3,Int}}` references shared vertex
# indices.  Binary STL is left for v3.2.

using StaticArrays: SVector

"""
    read_stl_ascii(path::AbstractString) ->
        (vertices::Vector{SVector{3, Float64}},
         faces::Vector{NTuple{3, Int}},
         normals::Vector{SVector{3, Float64}})

Parse an ASCII STL file at `path`.  Vertices are deduplicated by exact
floating-point match so that each triangle face is a 1-based index
triple into `vertices`.

Throws `ArgumentError` on empty or malformed input (missing
`solid`/`endsolid` header, truncated facet blocks, non-numeric tokens).
"""
function read_stl_ascii(path::AbstractString)
    isfile(path) || throw(ArgumentError("STL file not found: $(path)"))
    raw = read(path, String)
    tokens = split(raw)
    isempty(tokens) && throw(ArgumentError("empty STL file: $(path)"))

    vertices = SVector{3, Float64}[]
    faces = NTuple{3, Int}[]
    normals = SVector{3, Float64}[]
    vertex_index = Dict{NTuple{3, Float64}, Int}()

    function parse_float(tok, ctx)
        v = tryparse(Float64, tok)
        v === nothing &&
            throw(ArgumentError("malformed STL: expected float in $(ctx), got `$(tok)`"))
        return v
    end

    function intern_vertex!(vx, vy, vz)
        key = (vx, vy, vz)
        idx = get(vertex_index, key, 0)
        if idx == 0
            push!(vertices, SVector{3, Float64}(vx, vy, vz))
            idx = length(vertices)
            vertex_index[key] = idx
        end
        return idx
    end

    i = 1
    n = length(tokens)

    tokens[i] == "solid" ||
        throw(ArgumentError("malformed STL: expected `solid` header, got `$(tokens[i])`"))
    # Skip the rest of the header line — advance to the first `facet` or
    # straight to `endsolid` for an empty solid.
    i += 1
    while i <= n && tokens[i] != "facet" && tokens[i] != "endsolid"
        i += 1
    end

    while i <= n && tokens[i] == "facet"
        i + 4 <= n || throw(ArgumentError("malformed STL: truncated facet normal"))
        tokens[i + 1] == "normal" ||
            throw(ArgumentError("malformed STL: expected `normal` after `facet`, got `$(tokens[i + 1])`"))
        nx = parse_float(tokens[i + 2], "facet normal x")
        ny = parse_float(tokens[i + 3], "facet normal y")
        nz = parse_float(tokens[i + 4], "facet normal z")
        normal = SVector{3, Float64}(nx, ny, nz)
        i += 5

        (i + 1 <= n && tokens[i] == "outer" && tokens[i + 1] == "loop") ||
            throw(ArgumentError("malformed STL: expected `outer loop`"))
        i += 2

        i + 11 <= n || throw(ArgumentError("malformed STL: truncated vertex block"))
        (tokens[i] == "vertex" && tokens[i + 4] == "vertex" && tokens[i + 8] == "vertex") ||
            throw(ArgumentError("malformed STL: expected three `vertex` entries"))
        a = intern_vertex!(
            parse_float(tokens[i + 1], "vertex 1 x"),
            parse_float(tokens[i + 2], "vertex 1 y"),
            parse_float(tokens[i + 3], "vertex 1 z"),
        )
        b = intern_vertex!(
            parse_float(tokens[i + 5], "vertex 2 x"),
            parse_float(tokens[i + 6], "vertex 2 y"),
            parse_float(tokens[i + 7], "vertex 2 z"),
        )
        c = intern_vertex!(
            parse_float(tokens[i + 9], "vertex 3 x"),
            parse_float(tokens[i + 10], "vertex 3 y"),
            parse_float(tokens[i + 11], "vertex 3 z"),
        )
        i += 12

        (i <= n && tokens[i] == "endloop") ||
            throw(ArgumentError("malformed STL: expected `endloop`"))
        i += 1
        (i <= n && tokens[i] == "endfacet") ||
            throw(ArgumentError("malformed STL: expected `endfacet`"))
        i += 1

        push!(faces, (a, b, c))
        push!(normals, normal)
    end

    (i <= n && tokens[i] == "endsolid") ||
        throw(ArgumentError("malformed STL: expected `endsolid` terminator"))

    isempty(faces) && throw(ArgumentError("malformed STL: no facets parsed"))

    return vertices, faces, normals
end

"""
    write_stl_ascii(path, vertices, faces, normals)

Write the triangle soup `(vertices, faces, normals)` to `path` in ASCII
STL format.  Used by the V&V harness for round-trip tests.
"""
function write_stl_ascii(
        path::AbstractString,
        vertices::AbstractVector{<:SVector{3}},
        faces::AbstractVector{<:NTuple{3, <:Integer}},
        normals::AbstractVector{<:SVector{3}},
    )
    length(faces) == length(normals) ||
        throw(ArgumentError("faces and normals length mismatch: $(length(faces)) vs $(length(normals))"))
    open(path, "w") do io
        println(io, "solid fvm_test")
        for (t, face) in enumerate(faces)
            n = normals[t]
            println(io, "  facet normal $(n[1]) $(n[2]) $(n[3])")
            println(io, "    outer loop")
            for k in 1:3
                v = vertices[face[k]]
                println(io, "      vertex $(v[1]) $(v[2]) $(v[3])")
            end
            println(io, "    endloop")
            println(io, "  endfacet")
        end
        println(io, "endsolid fvm_test")
    end
    return path
end

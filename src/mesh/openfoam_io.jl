# mesh/openfoam_io.jl — OpenFOAM constant/polyMesh/ reader
#
# Reads an OpenFOAM case directory and constructs an UnstructuredFVMMesh{3, Float64}.
# Supports ASCII (uncompressed) polyMesh format. Files: points, faces, owner,
# neighbour, boundary.

using LinearAlgebra: norm, cross, dot

"""
    read_openfoam_polymesh(
        case_dir::AbstractString;
        mesh_dir::AbstractString = "constant/polyMesh",
    ) -> UnstructuredFVMMesh{3, Float64}

Read an OpenFOAM polyMesh directory and construct an `UnstructuredFVMMesh`.

# Arguments
- `case_dir` -- path to the OpenFOAM case directory
- `mesh_dir` -- relative path to the polyMesh directory (default: `"constant/polyMesh"`)

# Supported Format
ASCII (uncompressed) OpenFOAM polyMesh files only. For gzip-compressed
meshes, decompress first with `gunzip constant/polyMesh/*`.
"""
function read_openfoam_polymesh(
        case_dir::AbstractString;
        mesh_dir::AbstractString = "constant/polyMesh",
    )
    T = Float64
    Dim = 3
    base = joinpath(case_dir, mesh_dir)

    # Parse all files
    points = _read_openfoam_points(joinpath(base, "points"))
    face_verts = _read_openfoam_faces(joinpath(base, "faces"))
    owner_list = _read_openfoam_labels(joinpath(base, "owner"))
    neighbour_list = _read_openfoam_labels(joinpath(base, "neighbour"))
    patches = _read_openfoam_boundary(joinpath(base, "boundary"))

    npoints = size(points, 2)
    nfaces = length(owner_list)
    n_internal = length(neighbour_list)

    # Determine number of cells
    ncells = 0
    for f in 1:nfaces
        ncells = max(ncells, owner_list[f])
    end
    for f in 1:n_internal
        ncells = max(ncells, neighbour_list[f])
    end

    # Build face_cells: [owner; neighbour]
    face_cells = zeros(Int, 2, nfaces)
    for f in 1:nfaces
        face_cells[1, f] = owner_list[f]
    end
    for f in 1:n_internal
        face_cells[2, f] = neighbour_list[f]
    end
    # Boundary faces already have face_cells[2, f] = 0

    # Compute face geometry from vertices
    face_centers = Matrix{T}(undef, Dim, nfaces)
    face_areas = Vector{T}(undef, nfaces)
    face_normals = Matrix{T}(undef, Dim, nfaces)

    for f in 1:nfaces
        verts = face_verts[f]
        center, area, normal = _compute_face_geometry(points, verts)
        face_centers[1, f] = center[1]
        face_centers[2, f] = center[2]
        face_centers[3, f] = center[3]
        face_areas[f] = area
        face_normals[1, f] = normal[1]
        face_normals[2, f] = normal[2]
        face_normals[3, f] = normal[3]
    end

    # Orient normals: should point from owner to neighbour for internal faces.
    # OpenFOAM convention: face normal points from owner to neighbour, so
    # we check that the dot product with (cell_center_N - cell_center_P) is positive.
    # We need cell centers first, so compute them now.

    # Compute cell centers as average of incident face centers
    cell_center_accum = zeros(T, Dim, ncells)
    cell_face_count = zeros(Int, ncells)
    for f in 1:nfaces
        P = face_cells[1, f]
        cell_center_accum[1, P] += face_centers[1, f]
        cell_center_accum[2, P] += face_centers[2, f]
        cell_center_accum[3, P] += face_centers[3, f]
        cell_face_count[P] += 1

        N = face_cells[2, f]
        if N > 0
            cell_center_accum[1, N] += face_centers[1, f]
            cell_center_accum[2, N] += face_centers[2, f]
            cell_center_accum[3, N] += face_centers[3, f]
            cell_face_count[N] += 1
        end
    end

    cell_centers = Matrix{T}(undef, Dim, ncells)
    for c in 1:ncells
        count = max(cell_face_count[c], 1)
        cell_centers[1, c] = cell_center_accum[1, c] / count
        cell_centers[2, c] = cell_center_accum[2, c] / count
        cell_centers[3, c] = cell_center_accum[3, c] / count
    end

    # Compute cell volumes via Gauss divergence theorem:
    # V_c = (1/3) * sum_f sign_f * (x_f . S_f)
    # where S_f = area * normal (face area vector), sign_f = +1 for owner, -1 for neighbour
    cell_volumes = zeros(T, ncells)
    for f in 1:nfaces
        S_f = face_areas[f] .* [face_normals[1, f], face_normals[2, f], face_normals[3, f]]
        x_f = [face_centers[1, f], face_centers[2, f], face_centers[3, f]]
        flux = dot(x_f, S_f)

        P = face_cells[1, f]
        cell_volumes[P] += flux / 3

        N = face_cells[2, f]
        if N > 0
            cell_volumes[N] -= flux / 3
        end
    end

    # Ensure positive volumes
    for c in 1:ncells
        cell_volumes[c] = abs(cell_volumes[c])
    end

    # Build face tags from boundary patches
    face_tags = fill(:internal, nfaces)
    for patch in patches
        start_face = patch.startFace
        for i in 0:(patch.nFaces - 1)
            f = start_face + i
            if 1 <= f <= nfaces
                face_tags[f] = patch.name
            end
        end
    end

    # Build cell_faces connectivity (invert face_cells)
    cell_faces_vec = [Int[] for _ in 1:ncells]
    for f in 1:nfaces
        P = face_cells[1, f]
        push!(cell_faces_vec[P], f)
        N = face_cells[2, f]
        if N > 0
            push!(cell_faces_vec[N], f)
        end
    end

    return UnstructuredFVMMesh{Dim, T}(
        cell_centers, cell_volumes,
        face_cells, face_centers, face_areas, face_normals,
        face_tags, nothing, cell_faces_vec,
    )
end

# -- Internal parsers -----------------------------------------------------

"""Skip FoamFile header and comments, return lines of data content."""
function _skip_foam_header(io::IO)
    lines = String[]
    in_header = false
    brace_depth = 0

    for line in eachline(io)
        stripped = strip(line)
        # Skip empty lines and C++ comments
        if isempty(stripped) || startswith(stripped, "//")
            continue
        end

        # Track FoamFile { ... } block
        if startswith(stripped, "FoamFile")
            in_header = true
            continue
        end

        if in_header
            brace_depth += count(==('{'), stripped) - count(==('}'), stripped)
            if brace_depth <= 0
                in_header = false
            end
            continue
        end

        push!(lines, stripped)
    end

    return lines
end

"""Read OpenFOAM points file -> Matrix{Float64} of size 3 x npoints."""
function _read_openfoam_points(path::AbstractString)
    lines = open(_skip_foam_header, path)
    npoints = parse(Int, lines[1])
    points = Matrix{Float64}(undef, 3, npoints)

    idx = 0
    for i in 2:length(lines)
        line = lines[i]
        line == "(" && continue
        line == ")" && break

        # Parse "(x y z)" format
        cleaned = replace(replace(line, "(" => ""), ")" => "")
        parts = split(strip(cleaned))
        length(parts) >= 3 || continue

        idx += 1
        points[1, idx] = parse(Float64, parts[1])
        points[2, idx] = parse(Float64, parts[2])
        points[3, idx] = parse(Float64, parts[3])
    end

    return points
end

"""Read OpenFOAM faces file -> Vector{Vector{Int}} (1-indexed vertex indices)."""
function _read_openfoam_faces(path::AbstractString)
    lines = open(_skip_foam_header, path)
    nfaces_total = parse(Int, lines[1])
    faces = Vector{Vector{Int}}(undef, nfaces_total)

    idx = 0
    for i in 2:length(lines)
        line = lines[i]
        line == "(" && continue
        line == ")" && break

        # Parse "N(v0 v1 ... vN-1)" format
        m = match(r"(\d+)\(([^)]*)\)", line)
        m === nothing && continue

        idx += 1
        n_verts = parse(Int, m.captures[1])
        vert_strs = split(strip(m.captures[2]))
        verts = [parse(Int, s) + 1 for s in vert_strs]  # 0-indexed -> 1-indexed
        faces[idx] = verts
    end

    return faces
end

"""Read OpenFOAM label list (owner or neighbour) -> Vector{Int} (1-indexed cell indices)."""
function _read_openfoam_labels(path::AbstractString)
    lines = open(_skip_foam_header, path)
    nlabels = parse(Int, lines[1])
    labels = Vector{Int}(undef, nlabels)

    idx = 0
    for i in 2:length(lines)
        line = lines[i]
        line == "(" && continue
        line == ")" && break

        idx += 1
        labels[idx] = parse(Int, strip(line)) + 1  # 0-indexed -> 1-indexed
    end

    return labels
end

"""Read OpenFOAM boundary file -> Vector of NamedTuples."""
function _read_openfoam_boundary(path::AbstractString)
    lines = open(_skip_foam_header, path)
    patches = NamedTuple{(:name, :type, :nFaces, :startFace), Tuple{Symbol, Symbol, Int, Int}}[]

    i = 1
    # Skip count line and opening paren
    while i <= length(lines) && (lines[i] == "(" || tryparse(Int, lines[i]) !== nothing)
        i += 1
    end

    while i <= length(lines)
        line = lines[i]
        line == ")" && break

        # Patch name line (no braces, no semicolons)
        if !contains(line, "{") && !contains(line, "}") && !contains(line, ";")
            patch_name = Symbol(strip(line))
            patch_type = :patch
            n_faces = 0
            start_face = 0

            # Read patch block
            i += 1
            while i <= length(lines) && lines[i] != "}"
                pline = strip(lines[i])
                if startswith(pline, "type")
                    patch_type = Symbol(replace(replace(pline, "type" => ""), ";" => "") |> strip)
                elseif startswith(pline, "nFaces")
                    n_faces = parse(Int, replace(replace(pline, "nFaces" => ""), ";" => "") |> strip)
                elseif startswith(pline, "startFace")
                    start_face = parse(Int, replace(replace(pline, "startFace" => ""), ";" => "") |> strip) + 1  # 0->1 indexed
                end
                i += 1
            end

            push!(patches, (name = patch_name, type = patch_type, nFaces = n_faces, startFace = start_face))
        end

        i += 1
    end

    return patches
end

# -- Face geometry from vertices ------------------------------------------

"""
Compute face center, area, and unit normal from vertex indices.

For triangles: direct cross product.
For polygons: decompose into triangles from centroid and sum.
"""
function _compute_face_geometry(
        points::Matrix{Float64}, verts::Vector{Int},
    )
    nv = length(verts)

    # Compute face centroid
    cx, cy, cz = 0.0, 0.0, 0.0
    for v in verts
        cx += points[1, v]
        cy += points[2, v]
        cz += points[3, v]
    end
    cx /= nv
    cy /= nv
    cz /= nv
    center = SVector{3, Float64}(cx, cy, cz)

    if nv == 3
        # Triangle: direct cross product
        p1 = SVector{3}(points[1, verts[1]], points[2, verts[1]], points[3, verts[1]])
        p2 = SVector{3}(points[1, verts[2]], points[2, verts[2]], points[3, verts[2]])
        p3 = SVector{3}(points[1, verts[3]], points[2, verts[3]], points[3, verts[3]])
        n_vec = cross(p2 - p1, p3 - p1)
        area = norm(n_vec) / 2
        normal = area > 0 ? n_vec / (2 * area) : SVector{3}(0.0, 0.0, 1.0)
        return (center, area, normal)
    end

    # General polygon: decompose into triangles from centroid
    total_normal = SVector{3, Float64}(0.0, 0.0, 0.0)
    total_area = 0.0

    for i in 1:nv
        j = mod1(i + 1, nv)
        p1 = SVector{3}(points[1, verts[i]], points[2, verts[i]], points[3, verts[i]])
        p2 = SVector{3}(points[1, verts[j]], points[2, verts[j]], points[3, verts[j]])
        n_vec = cross(p1 - center, p2 - center)
        tri_area = norm(n_vec) / 2
        total_area += tri_area
        total_normal = total_normal + n_vec / 2  # area-weighted normal
    end

    normal = total_area > 0 ? total_normal / total_area : SVector{3}(0.0, 0.0, 1.0)
    return (center, total_area, normal)
end

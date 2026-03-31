# FVM Mesh Types and Builders - Migrated from Simu.jl SimuGeometry/fvm_mesh.jl
# Structured, curvilinear, and unstructured FVM mesh wrappers with validation and builders.

using LinearAlgebra: norm, cross, det

# --- FVM Mesh Types ---

abstract type AbstractFVMMesh{Dim, T} <: AbstractParabolicMesh end

struct StructuredFVMMesh{Dim, T, A <: AbstractArray{T, Dim}} <: AbstractFVMMesh{Dim, T}
    xc::NTuple{Dim, Vector{T}}
    Δ::NTuple{Dim, Vector{T}}
    cell_volumes::A
    face_areas::NTuple{Dim, A}
    periodic::NTuple{Dim, Bool}
end

struct CurvilinearFVMMesh{Dim, T, A <: AbstractArray{T, Dim}} <: AbstractFVMMesh{Dim, T}
    xc::NTuple{Dim, Vector{T}}           # parametric centers
    Δ::NTuple{Dim, Vector{T}}            # parametric spacings
    cell_volumes::A                     # physical volumes
    face_areas::NTuple{Dim, A}           # physical face areas (per face)
    metrics::NamedTuple                 # e.g., (:J => jacobian, :g => metric tensors)
    periodic::NTuple{Dim, Bool}
end

struct UnstructuredFVMMesh{Dim, T} <: AbstractFVMMesh{Dim, T}
    cell_centers::Matrix{T}
    cell_volumes::Vector{T}
    face_cells::Matrix{Int}
    face_centers::Matrix{T}
    face_areas::Vector{T}
    face_normals::Matrix{T}
    face_tags::Union{Nothing, Vector{Symbol}}
    face_velocity::Union{Nothing, Vector{T}}
    cell_faces::Union{Nothing, Vector{Vector{Int}}}
end

# --- Validation ---

function validate_mesh(mesh::StructuredFVMMesh)
    dims = size(mesh.cell_volumes)
    length(mesh.xc) == length(mesh.Δ) == length(mesh.face_areas) || error("Structured mesh dimension mismatch")
    for d in 1:length(dims)
        length(mesh.xc[d]) == dims[d] || error("xc size mismatch in dim $d")
        length(mesh.Δ[d]) == dims[d] || error("Δ size mismatch in dim $d")
    end
    return nothing
end

function validate_mesh(mesh::UnstructuredFVMMesh)
    ncells = length(mesh.cell_volumes)
    size(mesh.cell_centers, 2) == ncells || error("cell_centers column count must match cell_volumes")
    size(mesh.face_cells, 1) == 2 || error("face_cells must be 2 x nfaces")
    size(mesh.face_centers, 2) == size(mesh.face_cells, 2) || error("face_centers mismatch with faces")
    length(mesh.face_areas) == size(mesh.face_cells, 2) || error("face_areas length mismatch")
    size(mesh.face_normals, 2) == size(mesh.face_cells, 2) || error("face_normals mismatch with faces")
    if mesh.face_tags !== nothing
        length(mesh.face_tags) == size(mesh.face_cells, 2) || error("face_tags length mismatch")
    end
    if mesh.face_velocity !== nothing
        length(mesh.face_velocity) == size(mesh.face_cells, 2) || error("face_velocity length mismatch")
    end
    if mesh.cell_faces !== nothing
        length(mesh.cell_faces) == ncells || error("cell_faces length must match number of cells")
        for (c, faces) in enumerate(mesh.cell_faces)
            for f in faces
                (1 <= f <= size(mesh.face_cells, 2)) || error("cell_faces[$c] references invalid face index $f")
                mesh.face_cells[1, f] == c || mesh.face_cells[2, f] == c || error("cell_faces[$c] must reference incident faces")
            end
        end
    end
    return nothing
end

function validate_mesh(mesh::CurvilinearFVMMesh)
    dims = size(mesh.cell_volumes)
    length(mesh.xc) == length(mesh.Δ) == length(mesh.face_areas) || error("Curvilinear mesh dimension mismatch")
    for d in 1:length(dims)
        length(mesh.xc[d]) == dims[d] || error("xc size mismatch in dim $d")
        length(mesh.Δ[d]) == dims[d] || error("Δ size mismatch in dim $d")
    end
    return nothing
end

# --- Builders ---

"""
    build_structured_mesh3d(x_edges, y_edges, z_edges)

Build a 3D structured Cartesian mesh from edge coordinates in x, y, z.
Returns `StructuredFVMMesh{3}` with cell centers, spacings, volumes, and face areas.
"""
function build_structured_mesh3d(x_edges::AbstractVector, y_edges::AbstractVector, z_edges::AbstractVector)
    Nx, Ny, Nz = (length(x_edges) - 1, length(y_edges) - 1, length(z_edges) - 1)
    xc = (
        collect((x_edges[1:(end - 1)] .+ x_edges[2:end]) ./ 2),
        collect((y_edges[1:(end - 1)] .+ y_edges[2:end]) ./ 2),
        collect((z_edges[1:(end - 1)] .+ z_edges[2:end]) ./ 2),
    )
    Δ = (diff(x_edges), diff(y_edges), diff(z_edges))
    cell_volumes = [Δ[1][i] * Δ[2][j] * Δ[3][k] for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    face_areas_x = [Δ[2][j] * Δ[3][k] for i in 1:(Nx + 1), j in 1:Ny, k in 1:Nz]
    face_areas_y = [Δ[1][i] * Δ[3][k] for i in 1:Nx, j in 1:(Ny + 1), k in 1:Nz]
    face_areas_z = [Δ[1][i] * Δ[2][j] for i in 1:Nx, j in 1:Ny, k in 1:(Nz + 1)]
    face_areas = (
        reshape(face_areas_x, Nx + 1, Ny, Nz),
        reshape(face_areas_y, Nx, Ny + 1, Nz),
        reshape(face_areas_z, Nx, Ny, Nz + 1),
    )
    return StructuredFVMMesh{3, Float64, Array{Float64, 3}}(xc, Δ, cell_volumes, face_areas, (false, false, false))
end

"""
    build_axisymmetric_rz_mesh(r_edges, z_edges)

Build an axisymmetric r-z structured mesh (theta symmetry) using r and z edge coordinates.
Metrics (2*pi*r) are baked into volumes and face areas for use with standard flux assembly.
"""
function build_axisymmetric_rz_mesh(r_edges::AbstractVector, z_edges::AbstractVector)
    Nr, Nz = (length(r_edges) - 1, length(z_edges) - 1)
    r_mid = (r_edges[1:(end - 1)] .+ r_edges[2:end]) ./ 2
    z_mid = (z_edges[1:(end - 1)] .+ z_edges[2:end]) ./ 2
    dr = diff(r_edges)
    dz = diff(z_edges)
    twoπ = 2π
    # volumes: 2π r dr dz
    cell_volumes = [twoπ * r_mid[i] * dr[i] * dz[j] for i in 1:Nr, j in 1:Nz]
    # radial faces: area = 2π r_face * dz
    r_faces = r_edges
    face_area_r = [twoπ * r_faces[i] * dz[j] for i in 1:(Nr + 1), j in 1:Nz]
    # axial faces (z): area = 2π r_mid * dr
    face_area_z = [twoπ * r_mid[i] * dr[i] for i in 1:Nr, j in 1:(Nz + 1)]
    face_areas = (reshape(face_area_r, Nr + 1, Nz), reshape(face_area_z, Nr, Nz + 1))
    xc = (collect(r_mid), collect(z_mid))
    Δ = (dr, dz)
    return StructuredFVMMesh{2, Float64, Array{Float64, 2}}(xc, Δ, cell_volumes, face_areas, (false, false))
end

"""
    structured_boundary_tags(mesh::StructuredFVMMesh)

Auto boundary tags for structured meshes by coordinate bounds.
Returns Dict mapping region symbols (:left,:right,:bottom,:top,:front,:back) to tuples of (axis, side_coord).
"""
function structured_boundary_tags(mesh::StructuredFVMMesh)
    dim = length(mesh.xc)
    tags = Dict{Symbol, Tuple{Int, Float64}}()
    if dim >= 1
        tags[:left] = (1, mesh.xc[1][1] - mesh.Δ[1][1] / 2)
        tags[:right] = (1, mesh.xc[1][end] + mesh.Δ[1][end] / 2)
    end
    if dim >= 2
        tags[:bottom] = (2, mesh.xc[2][1] - mesh.Δ[2][1] / 2)
        tags[:top] = (2, mesh.xc[2][end] + mesh.Δ[2][end] / 2)
    end
    if dim == 3
        tags[:front] = (3, mesh.xc[3][1] - mesh.Δ[3][1] / 2)
        tags[:back] = (3, mesh.xc[3][end] + mesh.Δ[3][end] / 2)
    end
    return tags
end

"""
    build_curvilinear_mesh(param_edges, coord_map; jacobian=nothing, periodic=ntuple(_->false, Dim))

Construct a `CurvilinearFVMMesh` from parametric edges and a coordinate mapping.
"""
function build_curvilinear_mesh(
        param_edges::NTuple{Dim, AbstractVector}, coord_map;
        jacobian = nothing, periodic = ntuple(_ -> false, Dim)
    ) where {Dim}
    Δ = ntuple(i -> diff(param_edges[i]), Dim)
    xc = ntuple(i -> collect((param_edges[i][1:(end - 1)] .+ param_edges[i][2:end]) ./ 2), Dim)
    dims = Base.map(length, Δ)
    cell_volumes = zeros(Float64, dims...)
    function face_shape(d)
        return ntuple(k -> k == d ? dims[k] + 1 : dims[k], Dim)
    end
    face_areas = ntuple(d -> zeros(Float64, face_shape(d)), Dim)

    function finite_diff_jac(ξ_tuple, Δloc)
        f0 = coord_map(ξ_tuple...)
        phydim = length(f0)
        J = zeros(Float64, phydim, Dim)
        for d in 1:Dim
            δ = 0.5 * Δloc[d]
            bumped = ntuple(i -> i == d ? ξ_tuple[i] + δ : ξ_tuple[i], Dim)
            fδ = coord_map(bumped...)
            for r in 1:phydim
                J[r, d] = (fδ[r] - f0[r]) / δ
            end
        end
        return J, f0
    end

    phys_centers = ntuple(i -> similar(xc[i]), Dim)

    for I in CartesianIndices(cell_volumes)
        ξ = ntuple(d -> xc[d][I[d]], Dim)
        Δloc = ntuple(d -> Δ[d][I[d]], Dim)
        if jacobian === nothing
            J, coords = finite_diff_jac(ξ, Δloc)
        else
            coords = coord_map(ξ...)
            J = jacobian(ξ...)
        end
        for d in 1:Dim
            phys_centers[d][I[d]] = coords[d]
        end
        phydim = size(J, 1)
        volume = phydim == Dim ? abs(det(J)) * prod(Δloc) : 0.0
        cell_volumes[I] = volume
        for d in 1:Dim
            idxs = Tuple(I)
            face_idx_lo = CartesianIndex(Base.setindex(idxs, idxs[d], d))
            face_idx_hi = CartesianIndex(Base.setindex(idxs, idxs[d] + 1, d))
            if Dim == 2
                other = d == 1 ? 2 : 1
                tvec = J[:, other] .* Δloc[other]
                area = norm(tvec)
            else
                others = setdiff(1:Dim, (d,))
                v1 = J[:, others[1]] .* Δloc[others[1]]
                v2 = J[:, others[2]] .* Δloc[others[2]]
                area = norm(cross(v1, v2))
            end
            face_areas[d][face_idx_lo] = area
            if face_idx_hi[d] <= size(face_areas[d], d)
                face_areas[d][face_idx_hi] = area
            end
        end
    end

    metrics = (map = coord_map, jacobian = jacobian, physical_centers = phys_centers)
    return CurvilinearFVMMesh{Dim, Float64, typeof(cell_volumes)}(xc, Δ, cell_volumes, face_areas, metrics, periodic)
end

# --- Loaders ---

function polygon_area(points2d::Vector{<:AbstractVector})
    n = length(points2d)
    area = 0.0
    for i in 1:n
        j = i == n ? 1 : i + 1
        area += points2d[i][1] * points2d[j][2] - points2d[j][1] * points2d[i][2]
    end
    return 0.5 * area
end

function parse_ply(path)
    lines = readlines(path)
    startswith(lines[1], "ply") || error("Not a PLY file")
    nverts = 0
    nfaces = 0
    header_end = 0
    for (i, ln) in enumerate(lines)
        if startswith(ln, "element vertex")
            nverts = parse(Int, split(ln)[3])
        elseif startswith(ln, "element face")
            nfaces = parse(Int, split(ln)[3])
        elseif ln == "end_header"
            header_end = i
            break
        end
    end
    verts = [parse.(Float64, split(lines[header_end + i])) for i in 1:nverts]
    faces = Vector{Vector{Int}}()
    for i in 1:nfaces
        parts = split(lines[header_end + nverts + i])
        m = parse(Int, parts[1])
        push!(faces, parse.(Int, parts[2:(1 + m)]) .+ 1) # ply is 0-based
    end
    return verts, faces
end

function parse_vtk(path)
    lines = readlines(path)
    lower = lowercase.(lines)
    points_idx = findfirst(contains("points"), lower)
    points_idx === nothing && error("POINTS section not found in VTK file")
    nverts = parse(Int, split(lines[points_idx])[2])
    vert_lines = String[]
    i = points_idx + 1
    while length(vert_lines) < nverts && i <= length(lines)
        append!(vert_lines, split(lines[i]))
        i += 1
    end
    verts = [parse.(Float64, vert_lines[(3 * (k - 1) + 1):(3 * k)]) for k in 1:nverts]
    poly_idx = findfirst(contains("polygons"), lower)
    poly_idx === nothing && error("POLYGONS section not found in VTK file")
    parts = split(lines[poly_idx])
    nfaces = parse(Int, parts[2])
    faces = Vector{Vector{Int}}()
    cursor = poly_idx + 1
    for _ in 1:nfaces
        pts = parse.(Int, split(lines[cursor]))
        m = pts[1]
        push!(faces, pts[2:end] .+ 1)
        cursor += 1
    end
    return verts, faces
end

function tag_unstructured_faces_by_bounds(mesh::UnstructuredFVMMesh; atol = 1.0e-8)
    nfaces = size(mesh.face_cells, 2)
    face_tags = mesh.face_tags === nothing ? fill(Symbol(""), nfaces) : copy(mesh.face_tags)
    coords = mesh.face_centers
    phydim = size(coords, 1)
    mins = [minimum(view(coords, d, :)) for d in 1:phydim]
    maxs = [maximum(view(coords, d, :)) for d in 1:phydim]
    for f in 1:nfaces
        left, right = mesh.face_cells[:, f]
        right != 0 && continue  # only boundary faces
        c = view(coords, :, f)
        if phydim >= 1
            if isapprox(c[1], mins[1]; atol = atol)
                face_tags[f] = :left
            elseif isapprox(c[1], maxs[1]; atol = atol)
                face_tags[f] = :right
            end
        end
        if phydim >= 2
            if isapprox(c[2], mins[2]; atol = atol)
                face_tags[f] = :bottom
            elseif isapprox(c[2], maxs[2]; atol = atol)
                face_tags[f] = :top
            end
        end
        if phydim >= 3
            if isapprox(c[3], mins[3]; atol = atol)
                face_tags[f] = :front
            elseif isapprox(c[3], maxs[3]; atol = atol)
                face_tags[f] = :back
            end
        end
        face_tags[f] == Symbol("") && (face_tags[f] = :boundary)
    end
    return face_tags
end

function build_unstructured_from_polygons(vertices::Vector{<:AbstractVector}, faces::Vector{Vector{Int}}; velocity = nothing, tag_boundary::Bool = true)
    isempty(vertices) && error("No vertices provided")
    dim = length(vertices[1]) >= 2 ? 2 : error("Only 2D polygonal meshes are supported")
    ncells = length(faces)
    cell_centers = zeros(Float64, dim, ncells)
    cell_volumes = zeros(Float64, ncells)
    face_map = Dict{Tuple{Int, Int}, Int}()
    face_cells = Int[]
    face_cells_other = Int[]
    face_centers = Vector{Vector{Float64}}()
    face_normals = Vector{Vector{Float64}}()
    face_lengths = Float64[]
    cell_faces_list = [Int[] for _ in 1:ncells]

    for (ci, fverts) in enumerate(faces)
        pts = [vertices[idx][1:dim] for idx in fverts]
        area = polygon_area(pts)
        cell_volumes[ci] = abs(area)
        cx = sum(first, pts) / length(pts)
        cy = sum(last, pts) / length(pts)
        cell_centers[:, ci] .= (cx, cy)
        nverts = length(fverts)
        for k in 1:nverts
            v1 = fverts[k]
            v2 = fverts[k == nverts ? 1 : k + 1]
            key = v1 < v2 ? (v1, v2) : (v2, v1)
            if haskey(face_map, key)
                fi = face_map[key]
                face_cells_other[fi] = ci
                push!(cell_faces_list[ci], fi)
            else
                fi = length(face_map) + 1
                face_map[key] = fi
                push!(cell_faces_list[ci], fi)
                push!(face_cells, ci)
                push!(face_cells_other, 0)
                p1 = vertices[v1][1:dim]
                p2 = vertices[v2][1:dim]
                push!(face_centers, [(p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2])
                edge = [p2[1] - p1[1], p2[2] - p1[2]]
                normal = [edge[2], -edge[1]]
                nrm = norm(normal)
                normal = nrm == 0 ? [0.0, 0.0] : normal ./ nrm
                push!(face_normals, normal)
                push!(face_lengths, norm(edge))
            end
        end
    end

    nfaces = length(face_map)
    face_cells_mat = zeros(Int, 2, nfaces)
    for (fi, l) in enumerate(face_cells)
        face_cells_mat[1, fi] = l
        face_cells_mat[2, fi] = face_cells_other[fi]
    end

    face_center_mat = zeros(Float64, dim, nfaces)
    face_area_vec = zeros(Float64, nfaces)
    face_normal_mat = zeros(Float64, dim, nfaces)
    for fi in 1:nfaces
        face_center_mat[:, fi] .= face_centers[fi]
        face_area_vec[fi] = face_lengths[fi]
        face_normal_mat[:, fi] .= face_normals[fi]
    end
    mesh = UnstructuredFVMMesh{dim, Float64}(cell_centers, cell_volumes, face_cells_mat, face_center_mat, face_area_vec, face_normal_mat, nothing, velocity === nothing ? nothing : velocity, cell_faces_list)
    face_tags = tag_boundary ? tag_unstructured_faces_by_bounds(mesh) : mesh.face_tags
    mesh = UnstructuredFVMMesh{dim, Float64}(mesh.cell_centers, mesh.cell_volumes, mesh.face_cells, mesh.face_centers, mesh.face_areas, mesh.face_normals, face_tags, mesh.face_velocity, mesh.cell_faces)
    return mesh
end

function load_unstructured_mesh(path::AbstractString; velocity = nothing, tag_boundary::Bool = true)
    ext = lowercase(splitext(path)[2])
    verts, faces = if ext == ".ply"
        parse_ply(path)
    else
        parse_vtk(path)
    end
    return build_unstructured_from_polygons(verts, faces; velocity, tag_boundary)
end

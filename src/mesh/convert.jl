# mesh/convert.jl — Convert UnstructuredMesh2D/3D to UnstructuredFVMMesh
#
# Bridges the object-oriented mesh types (from read_gmsh) to the
# matrix-oriented UnstructuredFVMMesh used by the collocated solver.

"""
    convert_to_fvm_mesh(mesh::UnstructuredMesh3D; tag_boundary = true)

Convert an `UnstructuredMesh3D` (from `read_gmsh`) to an
`UnstructuredFVMMesh{3, Float64}` suitable for the collocated solver.

Extracts cell centers, volumes, face connectivity, normals, areas, and
centers from the object-oriented mesh. If `tag_boundary = true`, assigns
boundary face tags via `tag_unstructured_faces_by_bounds`.
"""
function convert_to_fvm_mesh(
        mesh::UnstructuredMesh3D;
        tag_boundary::Bool = true,
    )
    ncells = length(mesh.cells)
    nfaces = length(mesh.faces)
    Dim = 3
    T = Float64

    # Cell data
    cell_centers = Matrix{T}(undef, Dim, ncells)
    cell_volumes = Vector{T}(undef, ncells)
    for c in 1:ncells
        cell = mesh.cells[c]
        cell_centers[1, c] = cell.center[1]
        cell_centers[2, c] = cell.center[2]
        cell_centers[3, c] = cell.center[3]
        cell_volumes[c] = cell.volume
    end

    # Face data
    face_cells = Matrix{Int}(undef, 2, nfaces)
    face_centers = Matrix{T}(undef, Dim, nfaces)
    face_areas = Vector{T}(undef, nfaces)
    face_normals = Matrix{T}(undef, Dim, nfaces)
    for f in 1:nfaces
        face = mesh.faces[f]
        face_cells[1, f] = face.owner
        face_cells[2, f] = face.neighbor
        face_centers[1, f] = face.center[1]
        face_centers[2, f] = face.center[2]
        face_centers[3, f] = face.center[3]
        face_areas[f] = face.area
        n_mag = norm(face.normal)
        if n_mag > 0
            face_normals[1, f] = face.normal[1] / n_mag
            face_normals[2, f] = face.normal[2] / n_mag
            face_normals[3, f] = face.normal[3] / n_mag
        else
            face_normals[1, f] = 0.0
            face_normals[2, f] = 0.0
            face_normals[3, f] = 1.0
        end
    end

    # Cell-to-face connectivity
    cell_faces_vec = [copy(mesh.cells[c].faces) for c in 1:ncells]

    # Build mesh without tags first
    fvm_mesh = UnstructuredFVMMesh{Dim, T}(
        cell_centers, cell_volumes,
        face_cells, face_centers, face_areas, face_normals,
        nothing, nothing, cell_faces_vec,
    )

    # Tag boundary faces
    if tag_boundary
        tags = tag_unstructured_faces_by_bounds(fvm_mesh)
        fvm_mesh = UnstructuredFVMMesh{Dim, T}(
            cell_centers, cell_volumes,
            face_cells, face_centers, face_areas, face_normals,
            tags, nothing, cell_faces_vec,
        )
    end

    return fvm_mesh
end

"""
    convert_to_fvm_mesh(mesh::UnstructuredMesh2D; tag_boundary = true)

Convert an `UnstructuredMesh2D` to an `UnstructuredFVMMesh{2, Float64}`.
"""
function convert_to_fvm_mesh(
        mesh::UnstructuredMesh2D;
        tag_boundary::Bool = true,
    )
    ncells = length(mesh.cells)
    nfaces = length(mesh.faces)
    Dim = 2
    T = Float64

    cell_centers = Matrix{T}(undef, Dim, ncells)
    cell_volumes = Vector{T}(undef, ncells)
    for c in 1:ncells
        cell = mesh.cells[c]
        cell_centers[1, c] = cell.center[1]
        cell_centers[2, c] = cell.center[2]
        cell_volumes[c] = cell.volume
    end

    face_cells = Matrix{Int}(undef, 2, nfaces)
    face_centers = Matrix{T}(undef, Dim, nfaces)
    face_areas = Vector{T}(undef, nfaces)
    face_normals = Matrix{T}(undef, Dim, nfaces)
    for f in 1:nfaces
        face = mesh.faces[f]
        face_cells[1, f] = face.owner
        face_cells[2, f] = face.neighbor
        face_centers[1, f] = face.center[1]
        face_centers[2, f] = face.center[2]
        face_areas[f] = face.area
        n_mag = norm(face.normal)
        if n_mag > 0
            face_normals[1, f] = face.normal[1] / n_mag
            face_normals[2, f] = face.normal[2] / n_mag
        else
            face_normals[1, f] = 0.0
            face_normals[2, f] = 1.0
        end
    end

    cell_faces_vec = [copy(mesh.cells[c].faces) for c in 1:ncells]

    fvm_mesh = UnstructuredFVMMesh{Dim, T}(
        cell_centers, cell_volumes,
        face_cells, face_centers, face_areas, face_normals,
        nothing, nothing, cell_faces_vec,
    )

    if tag_boundary
        tags = tag_unstructured_faces_by_bounds(fvm_mesh)
        fvm_mesh = UnstructuredFVMMesh{Dim, T}(
            cell_centers, cell_volumes,
            face_cells, face_centers, face_areas, face_normals,
            tags, nothing, cell_faces_vec,
        )
    end

    return fvm_mesh
end

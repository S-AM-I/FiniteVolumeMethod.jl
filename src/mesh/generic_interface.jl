# src/mesh/generic_interface.jl — Stage 1d generic-mesh interface overloads
#
# Loaded AFTER every concrete mesh type has been declared. Provides
# `n_cells(mesh)` and `n_faces(mesh)` overloads across the three solver
# families so downstream code can dispatch on `::AbstractFiniteVolumeMesh`.
#
# Keep declarations here focused on the umbrella contract; per-family
# convenience accessors (like Rhie-Chow owners or triangulation getters)
# stay in their own files.

# ── Collocated / parabolic FVM wrappers (src/parabolic/mesh/fvm_mesh.jl) ──

n_cells(mesh::UnstructuredFVMMesh) = length(mesh.cell_volumes)
n_faces(mesh::UnstructuredFVMMesh) = size(mesh.face_cells, 2)

n_cells(mesh::StructuredFVMMesh) = length(mesh.cell_volumes)
function n_faces(mesh::StructuredFVMMesh{Dim}) where {Dim}
    return _structured_face_count(mesh)
end

n_cells(mesh::CurvilinearFVMMesh) = length(mesh.cell_volumes)
function n_faces(mesh::CurvilinearFVMMesh{Dim}) where {Dim}
    return _structured_face_count(mesh)
end

@inline function _structured_face_count(mesh)
    dims = size(mesh.cell_volumes)
    Dim = length(dims)
    total = 0
    @inbounds for d in 1:Dim
        face_count = mesh.periodic[d] ? dims[d] : dims[d] + 1
        others = 1
        for e in 1:Dim
            e == d && continue
            others *= dims[e]
        end
        total += face_count * others
    end
    return total
end

# ── FVMGeometry (parabolic vertex-centered over a triangulation) ─────

n_cells(mesh::FVMGeometry) = length(mesh.cv_volumes)
function n_faces(mesh::FVMGeometry)
    return DelaunayTriangulation.num_solid_edges(mesh.triangulation_statistics)
end

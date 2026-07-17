# ============================================================
# Data Transfer Between Cell-Centered and Vertex-Centered
# ============================================================
#
# Interpolation utilities for transferring data between
# cell-centered (hyperbolic solver) and vertex-centered
# (cell-vertex FVM) representations on unstructured meshes.

"""
    cell_to_vertex(mesh::UnstructuredHyperbolicMesh, cell_values::AbstractVector)

Interpolate cell-centered values to vertices using area-weighted averaging.

Given values `u_T` at each triangle centroid T, compute vertex values:
    `u_v = Σ(A_T * u_T) / Σ(A_T)`
where the sum is over all triangles sharing vertex `v`.

# Arguments
- `mesh`: The unstructured triangular mesh.
- `cell_values`: Vector of length `mesh.ntri` with values at cell centroids.

# Returns
- `Dict{Int, V}`: Mapping from vertex ID to interpolated value.
"""
function cell_to_vertex(mesh::UnstructuredHyperbolicMesh, cell_values::AbstractVector)
    V = eltype(cell_values)
    vertex_sum = Dict{Int, V}()
    vertex_weight = Dict{Int, Float64}()

    for (tri_id, (v1, v2, v3)) in enumerate(mesh.tri_verts)
        area = mesh.tri_areas[tri_id]
        val = cell_values[tri_id]
        for v in (v1, v2, v3)
            if haskey(vertex_sum, v)
                vertex_sum[v] = vertex_sum[v] + area * val
                vertex_weight[v] = vertex_weight[v] + area
            else
                vertex_sum[v] = area * val
                vertex_weight[v] = area
            end
        end
    end

    result = Dict{Int, V}()
    for (v, w) in vertex_weight
        result[v] = vertex_sum[v] / w
    end
    return result
end

"""
    vertex_to_cell(mesh::UnstructuredHyperbolicMesh, vertex_values::Dict{Int})

Interpolate vertex-centered values to cell centroids by simple averaging.

    `u_T = (u_{v1} + u_{v2} + u_{v3}) / 3`

# Arguments
- `mesh`: The unstructured triangular mesh.
- `vertex_values`: Dict mapping vertex ID to value.

# Returns
- `Vector`: Values at each triangle centroid.
"""
function vertex_to_cell(mesh::UnstructuredHyperbolicMesh, vertex_values::Dict)
    V = valtype(vertex_values)
    cell_values = Vector{V}(undef, mesh.ntri)
    for (tri_id, (v1, v2, v3)) in enumerate(mesh.tri_verts)
        cell_values[tri_id] = (vertex_values[v1] + vertex_values[v2] + vertex_values[v3]) / 3
    end
    return cell_values
end

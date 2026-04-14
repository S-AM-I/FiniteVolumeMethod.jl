# postprocessing/sampling.jl — Field sampling along lines and at points
#
# Supports nearest-cell (0th order) and inverse-distance-weighted (IDW)
# interpolation for extracting field values along lines or at points.

using LinearAlgebra: norm

# -- IDW helpers ---------------------------------------------------------------

"""
    _find_nearest_cells(mesh, point, n) -> Vector{Int}

Find the `n` nearest cell indices to `point` (brute force).
"""
function _find_nearest_cells(
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T},
        n::Int,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    n = min(n, nc)
    # Compute all distances
    dists = Vector{T}(undef, nc)
    for c in 1:nc
        dists[c] = norm(point - cell_center(mesh, c))
    end
    return partialsortperm(dists, 1:n)
end

"""
    _idw_interpolate(field_values, mesh, cells, point; power) -> T

Inverse-distance-weighted interpolation over the given cell indices.
Falls back to the nearest cell value if the point coincides with a cell
center.
"""
function _idw_interpolate(
        field_values::Vector{V},
        mesh::UnstructuredFVMMesh{Dim, T},
        cells::AbstractVector{Int},
        point::SVector{Dim, T};
        power::T = T(2),
    ) where {Dim, T, V}
    weights = Vector{T}(undef, length(cells))
    for (i, c) in enumerate(cells)
        d = norm(point - cell_center(mesh, c))
        if d < eps(T) * T(100)
            return field_values[c]  # exact hit
        end
        weights[i] = one(T) / d^power
    end
    w_sum = sum(weights)
    result = weights[1] / w_sum * field_values[cells[1]]
    for i in 2:length(cells)
        result = result + weights[i] / w_sum * field_values[cells[i]]
    end
    return result
end

# -- Point sampling ------------------------------------------------------------

"""
    sample_field_at_point(field, mesh, point; interpolation = :nearest, n_neighbors = 4)

Sample a scalar field at `point`.

# Keyword Arguments
- `interpolation` — `:nearest` (default, 0th order) or `:idw` (inverse-distance-weighted)
- `n_neighbors` — number of nearest cells for IDW (default: 4, ignored for `:nearest`)
"""
function sample_field_at_point(
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T};
        interpolation::Symbol = :nearest,
        n_neighbors::Int = 4,
    ) where {Dim, T}
    if interpolation === :idw
        cells = _find_nearest_cells(mesh, point, n_neighbors)
        return _idw_interpolate(field.internal, mesh, cells, point)
    else
        c = find_nearest_cell(mesh, point)
        return field.internal[c]
    end
end

"""
    sample_field_at_point(field, mesh, point; interpolation = :nearest, n_neighbors = 4)

Sample a vector field at `point`.
"""
function sample_field_at_point(
        field::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T};
        interpolation::Symbol = :nearest,
        n_neighbors::Int = 4,
    ) where {Dim, T}
    if interpolation === :idw
        cells = _find_nearest_cells(mesh, point, n_neighbors)
        return _idw_interpolate(field.internal, mesh, cells, point)
    else
        c = find_nearest_cell(mesh, point)
        return field.internal[c]
    end
end

# -- Line sampling -------------------------------------------------------------

"""
    sample_line(field, mesh, p1, p2, n_points; interpolation = :nearest, n_neighbors = 4)

Sample a scalar field at `n_points` evenly spaced along the line from
`p1` to `p2`.

Returns `(positions, distances, values)` where:
- `positions::Vector{SVector{Dim, T}}` — sample point coordinates
- `distances::Vector{T}` — distance along the line from `p1`
- `values::Vector{T}` — field values at each sample point

Set `interpolation = :idw` for inverse-distance-weighted interpolation.
"""
function sample_line(
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        p1::SVector{Dim, T},
        p2::SVector{Dim, T},
        n_points::Int;
        interpolation::Symbol = :nearest,
        n_neighbors::Int = 4,
    ) where {Dim, T}
    positions = Vector{SVector{Dim, T}}(undef, n_points)
    distances = Vector{T}(undef, n_points)
    values = Vector{T}(undef, n_points)

    L = norm(p2 - p1)

    for i in 1:n_points
        t = (i - 1) / max(n_points - 1, 1)
        pt = p1 + t * (p2 - p1)
        positions[i] = pt
        distances[i] = t * L
        values[i] = sample_field_at_point(
            field, mesh, pt; interpolation = interpolation, n_neighbors = n_neighbors,
        )
    end

    return (positions = positions, distances = distances, values = values)
end

"""
    sample_line(field, mesh, p1, p2, n_points; interpolation = :nearest, n_neighbors = 4)

Sample a vector field along a line. Returns `values::Vector{SVector{Dim, T}}`.
"""
function sample_line(
        field::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        p1::SVector{Dim, T},
        p2::SVector{Dim, T},
        n_points::Int;
        interpolation::Symbol = :nearest,
        n_neighbors::Int = 4,
    ) where {Dim, T}
    positions = Vector{SVector{Dim, T}}(undef, n_points)
    distances = Vector{T}(undef, n_points)
    values = Vector{SVector{Dim, T}}(undef, n_points)

    L = norm(p2 - p1)

    for i in 1:n_points
        t = (i - 1) / max(n_points - 1, 1)
        pt = p1 + t * (p2 - p1)
        positions[i] = pt
        distances[i] = t * L
        values[i] = sample_field_at_point(
            field, mesh, pt; interpolation = interpolation, n_neighbors = n_neighbors,
        )
    end

    return (positions = positions, distances = distances, values = values)
end

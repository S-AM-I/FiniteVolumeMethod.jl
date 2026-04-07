# postprocessing/sampling.jl — Field sampling along lines and at points
#
# Nearest-cell-center interpolation (0th order) for extracting field
# values along lines or at arbitrary points.

using LinearAlgebra: norm

# -- Point sampling ------------------------------------------------------------

"""
    sample_field_at_point(field, mesh, point) -> T

Sample a scalar field at `point` using nearest-cell-center interpolation.
"""
function sample_field_at_point(
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T},
    ) where {Dim, T}
    c = find_nearest_cell(mesh, point)
    return field.internal[c]
end

"""
    sample_field_at_point(field, mesh, point) -> SVector{Dim, T}

Sample a vector field at `point` using nearest-cell-center interpolation.
"""
function sample_field_at_point(
        field::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T},
    ) where {Dim, T}
    c = find_nearest_cell(mesh, point)
    return field.internal[c]
end

# -- Line sampling -------------------------------------------------------------

"""
    sample_line(field, mesh, p1, p2, n_points)

Sample a scalar field at `n_points` evenly spaced along the line from
`p1` to `p2`.

Returns `(positions, distances, values)` where:
- `positions::Vector{SVector{Dim, T}}` — sample point coordinates
- `distances::Vector{T}` — distance along the line from `p1`
- `values::Vector{T}` — field values at each sample point
"""
function sample_line(
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        p1::SVector{Dim, T},
        p2::SVector{Dim, T},
        n_points::Int,
    ) where {Dim, T}
    positions = Vector{SVector{Dim, T}}(undef, n_points)
    distances = Vector{T}(undef, n_points)
    values = Vector{T}(undef, n_points)

    L = norm(p2 - p1)
    dir = L > zero(T) ? (p2 - p1) / L : zero(SVector{Dim, T})

    for i in 1:n_points
        t = (i - 1) / max(n_points - 1, 1)
        pt = p1 + t * (p2 - p1)
        positions[i] = pt
        distances[i] = t * L
        values[i] = sample_field_at_point(field, mesh, pt)
    end

    return (positions = positions, distances = distances, values = values)
end

"""
    sample_line(field, mesh, p1, p2, n_points)

Sample a vector field along a line. Returns `values::Vector{SVector{Dim, T}}`.
"""
function sample_line(
        field::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        p1::SVector{Dim, T},
        p2::SVector{Dim, T},
        n_points::Int,
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
        values[i] = sample_field_at_point(field, mesh, pt)
    end

    return (positions = positions, distances = distances, values = values)
end

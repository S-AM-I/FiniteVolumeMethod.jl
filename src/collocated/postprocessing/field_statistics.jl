# postprocessing/field_statistics.jl — Scalar field statistics
#
# Volume-weighted averages, extrema, and turbulence-derived quantities
# on UnstructuredFVMMesh.

# -- Volume-weighted average -----------------------------------------------

"""
    field_average(field::CollocatedScalarField{T}, mesh::UnstructuredFVMMesh{Dim, T}) -> T

Compute the volume-weighted average of a cell-centered scalar field:

    <f> = (sum_c f_c * V_c) / (sum_c V_c)
"""
function field_average(
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(field.internal)
    weighted_sum = zero(T)
    total_volume = zero(T)
    for c in 1:nc
        V = mesh.cell_volumes[c]
        weighted_sum += field.internal[c] * V
        total_volume += V
    end
    return weighted_sum / total_volume
end

# -- Min / max -------------------------------------------------------------

"""
    field_min_max(field::CollocatedScalarField{T}) -> (min::T, max::T)

Return the minimum and maximum values of a scalar field over all cells.
"""
function field_min_max(field::CollocatedScalarField{T}) where {T}
    vals = field.internal
    return (minimum(vals), maximum(vals))
end

# -- Turbulence intensity --------------------------------------------------

"""
    turbulence_intensity(k_field::CollocatedScalarField{T}, U_mean::T) -> Vector{T}

Compute local turbulence intensity at each cell:

    TI_c = sqrt(2 * k_c / 3) / U_mean

where `k_c` is the turbulent kinetic energy and `U_mean` is the
reference mean velocity magnitude. Returns zero where `U_mean` is
zero or negative.
"""
function turbulence_intensity(
        k_field::CollocatedScalarField{T},
        U_mean::T,
    ) where {T}
    nc = length(k_field.internal)
    TI = Vector{T}(undef, nc)
    if U_mean <= zero(T)
        fill!(TI, zero(T))
        return TI
    end
    inv_U = one(T) / U_mean
    for c in 1:nc
        k_c = max(k_field.internal[c], zero(T))
        TI[c] = sqrt(T(2) * k_c / T(3)) * inv_U
    end
    return TI
end

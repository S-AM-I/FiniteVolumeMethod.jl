# thermal/buoyancy.jl — Boussinesq buoyancy source term
#
# Computes the body force F_b = -rho * beta * (T - T_ref) * g for natural
# convection using the Boussinesq approximation.

"""
    compute_buoyancy_source(
        T_field::CollocatedScalarField{T},
        props::FluidThermalProperties{Dim, T},
        density::T,
    ) -> Vector{SVector{Dim, T}}

Compute the Boussinesq buoyancy body force per cell:

    F_b[c] = -rho * beta * (T[c] - T_ref) * g

Returns a vector of `SVector{Dim, T}` with one entry per cell.
Returns `nothing` when `beta == 0` (no buoyancy).
"""
function compute_buoyancy_source(
        T_field::CollocatedScalarField{T},
        props::FluidThermalProperties{Dim, T},
        density::T,
    ) where {Dim, T}
    if !has_buoyancy(props)
        return nothing
    end

    nc = length(T_field.internal)
    force = Vector{SVector{Dim, T}}(undef, nc)
    for c in 1:nc
        dT = T_field.internal[c] - props.T_ref
        force[c] = -density * props.beta * dT * props.g
    end
    return force
end

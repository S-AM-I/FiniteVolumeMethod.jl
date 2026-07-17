# thermal/buoyancy.jl — Boussinesq buoyancy source term
#
# Computes the KINEMATIC body force F_b = -beta * (T - T_ref) * g
# (force per unit mass) for natural convection using the Boussinesq
# approximation.  The collocated momentum equation is assembled in
# kinematic form (ν, volumetric flux, p/ρ), so sources must be per unit
# mass — the previous ρ-scaled (dynamic) form silently broke momentum
# whenever density ≠ 1.

"""
    compute_buoyancy_source(
        T_field::CollocatedScalarField{T},
        props::FluidThermalProperties{Dim, T},
        density::T,
    ) -> Vector{SVector{Dim, T}}

Compute the Boussinesq buoyancy force per cell in KINEMATIC units
(force per unit mass), matching the kinematic momentum equation:

    F_b[c] = -beta * (T[c] - T_ref) * g

The `density` argument is retained for API compatibility but no longer
scales the force (under Boussinesq, ρ cancels when the momentum equation
is divided through by ρ).

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
        force[c] = -props.beta * dT * props.g
    end
    return force
end

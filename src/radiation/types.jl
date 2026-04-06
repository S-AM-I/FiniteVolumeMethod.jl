# radiation/types.jl — Core types for radiation modeling
#
# Defines the radiation model hierarchy, the P1 model, radiation state,
# and boundary condition convenience constructors.

"""Stefan-Boltzmann constant [W/(m^2 K^4)]."""
const STEFAN_BOLTZMANN = 5.670374419e-8

"""
    AbstractRadiationModel

Supertype for radiation models.
"""
abstract type AbstractRadiationModel end

"""
    P1Model{T} <: AbstractRadiationModel

P1 radiation approximation. Solves a single diffusion equation for the
incident radiation field G:

    -div(Gamma * grad(G)) + a * G = 4 * a * sigma * T^4

where `Gamma = 1/(3a)` and `a` is the absorption coefficient.

# Fields
- `a::T` --- absorption coefficient [1/m]
"""
struct P1Model{T} <: AbstractRadiationModel
    a::T
end

"""
    P1Model(; a = 0.1)

Construct a P1 radiation model with constant absorption coefficient.
"""
P1Model(; a::Real = 0.1) = P1Model{typeof(Float64(a))}(Float64(a))

"""
    RadiationState{T}

Mutable state for radiation models. Holds the incident radiation field.

# Fields
- `G::CollocatedScalarField{T}` --- incident radiation [W/m^2]
"""
mutable struct RadiationState{T}
    G::CollocatedScalarField{T}
end

"""
    RadiationState(mesh; G_init = 0.0)

Construct a zero-initialized radiation state.
"""
function RadiationState(
        mesh::UnstructuredFVMMesh{Dim, T};
        G_init::Real = 0.0,
    ) where {Dim, T}
    G = CollocatedScalarField(:G, mesh; value = T(G_init))
    return RadiationState{T}(G)
end

# -- BC convenience constructors ------------------------------------------------

"""
    marshak_wall_bc(rad_model::P1Model, T_wall)

Marshak boundary condition for an opaque wall at temperature `T_wall`:
`G + (2/(3a)) * dG/dn = 4 * sigma * T_wall^4`

Implemented as `ParabolicRobin(1, 2/(3a), 4 * sigma * T^4)`.
"""
function marshak_wall_bc(rad_model::P1Model{T}, T_wall::Real) where {T}
    b_coeff = T(2) / (T(3) * rad_model.a)
    c_val = T(4) * T(STEFAN_BOLTZMANN) * T(T_wall)^4
    return ParabolicRobin(one(T), b_coeff, c_val)
end

"""
    radiation_inlet_bc(T_inlet)

Fixed incident radiation BC from a known temperature:
`G = 4 * sigma * T^4`
"""
function radiation_inlet_bc(T_inlet::Real)
    G_val = 4.0 * STEFAN_BOLTZMANN * Float64(T_inlet)^4
    return ParabolicDirichlet(G_val)
end

# multiphase/types.jl — Core types for Volume of Fluid multiphase
#
# Defines fluid property pairs for two immiscible phases and the
# mutable VOF state (volume fraction + mixture properties).

"""
    TwoPhaseProperties{T}

Physical properties for a two-phase immiscible flow system.

# Fields
- `rho1::T` — density of fluid 1 (α = 1), e.g., water = 1000 kg/m³
- `rho2::T` — density of fluid 2 (α = 0), e.g., air = 1.225 kg/m³
- `mu1::T` — dynamic viscosity of fluid 1, e.g., water = 1e-3 Pa·s
- `mu2::T` — dynamic viscosity of fluid 2, e.g., air = 1.8e-5 Pa·s
- `sigma::T` — surface tension coefficient [N/m] (0 = disabled)
"""
struct TwoPhaseProperties{T}
    rho1::T
    rho2::T
    mu1::T
    mu2::T
    sigma::T
end

"""
    TwoPhaseProperties(; rho1, rho2, mu1, mu2, sigma)

Construct two-phase properties with keyword defaults for water/air at 20°C.
"""
function TwoPhaseProperties(;
        rho1::Real = 1000.0,
        rho2::Real = 1.225,
        mu1::Real = 1.0e-3,
        mu2::Real = 1.8e-5,
        sigma::Real = 0.072,
    )
    T = promote_type(typeof(rho1), typeof(rho2), typeof(mu1), typeof(mu2), typeof(sigma))
    return TwoPhaseProperties{T}(T(rho1), T(rho2), T(mu1), T(mu2), T(sigma))
end

"""Check if surface tension is active."""
has_surface_tension(props::TwoPhaseProperties) = props.sigma > 0

"""
    VOFState{T}

Mutable state for VOF multiphase simulation.

# Fields
- `alpha::CollocatedScalarField{T}` — volume fraction [0, 1]
- `rho::Vector{T}` — mixture density per cell
- `mu::Vector{T}` — mixture dynamic viscosity per cell
"""
mutable struct VOFState{T}
    alpha::CollocatedScalarField{T}
    rho::Vector{T}
    mu::Vector{T}
end

"""
    VOFState(mesh; alpha_init = 0.0)

Construct a VOF state with uniform initial volume fraction.
"""
function VOFState(
        mesh::UnstructuredFVMMesh{Dim, T};
        alpha_init::Real = 0.0,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    alpha = CollocatedScalarField(:alpha, mesh; value = T(alpha_init))
    rho = fill(T(1.0), nc)    # placeholder, updated by update_mixture_properties!
    mu = fill(T(1.0e-3), nc)  # placeholder
    return VOFState{T}(alpha, rho, mu)
end

"""
    VOFState(mesh, alpha_func::Function, props::TwoPhaseProperties)

Construct a VOF state with spatially varying initial alpha defined by
`alpha_func(x::SVector) -> T`. Mixture properties are initialized to
placeholders; call `update_mixture_properties!` after construction.
"""
function VOFState(
        mesh::UnstructuredFVMMesh{Dim, T},
        alpha_func::Function,
        props::TwoPhaseProperties{T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        x_c = cell_center(mesh, c)
        alpha.internal[c] = clamp(alpha_func(x_c), zero(T), one(T))
    end
    # Set boundary values
    for (i, f) in enumerate(alpha.boundary_face_indices)
        x_f = face_center(mesh, f)
        alpha.boundary[i] = clamp(alpha_func(x_f), zero(T), one(T))
    end
    rho = fill(T(1.0), nc)    # placeholder
    mu = fill(T(1.0e-3), nc)  # placeholder
    return VOFState{T}(alpha, rho, mu)
end

# turbulence/wall_functions.jl — Wall function BC generation for turbulence models
#
# Generates boundary condition dictionaries for turbulence fields at wall
# and inlet patches. Reuses existing compute_friction_velocity, k_wall_value,
# epsilon_wall_value from src/physics/turbulence/k_epsilon.jl.

"""
    turbulence_inlet_bc(model::StandardKEpsilon, U_mag, intensity, length_scale)

Generate inlet BCs for k-ε from freestream conditions.

- `k_inlet = 1.5 * (U_mag * intensity)²`
- `ε_inlet = C_μ^0.75 * k^1.5 / length_scale`
"""
function turbulence_inlet_bc(
        model::StandardKEpsilon, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    k_inlet = T(1.5) * (U_mag * intensity)^2
    eps_inlet = model.C_mu^T(0.75) * k_inlet^T(1.5) / length_scale
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicDirichlet(k_inlet),
        :epsilon => ParabolicDirichlet(eps_inlet),
    )
end

"""
    turbulence_inlet_bc(model::KOmega, U_mag, intensity, length_scale)

Generate inlet BCs for k-ω from freestream conditions.

- `k_inlet = 1.5 * (U_mag * intensity)²`
- `ω_inlet = k^0.5 / (C_μ^0.25 * length_scale)`
"""
function turbulence_inlet_bc(
        model::KOmega, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    k_inlet = T(1.5) * (U_mag * intensity)^2
    omega_inlet = sqrt(k_inlet) / (T(0.09)^T(0.25) * length_scale)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicDirichlet(k_inlet),
        :omega => ParabolicDirichlet(omega_inlet),
    )
end

"""
    turbulence_inlet_bc(model::KOmegaSSTModel, U_mag, intensity, length_scale)

Generate inlet BCs for k-ω SST (same as k-ω).
"""
function turbulence_inlet_bc(
        model::KOmegaSSTModel, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    k_inlet = T(1.5) * (U_mag * intensity)^2
    omega_inlet = sqrt(k_inlet) / (model.coeffs.beta_star^T(0.25) * length_scale)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicDirichlet(k_inlet),
        :omega => ParabolicDirichlet(omega_inlet),
    )
end

"""
    turbulence_inlet_bc(model::SpalartAllmaras, U_mag, intensity, length_scale)

Generate inlet BCs for SA. ν̃_inlet ≈ 3-5 * ν for freestream.
"""
function turbulence_inlet_bc(
        ::SpalartAllmaras, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    # SA freestream: nu_tilde ≈ 3 * nu_laminar is typical
    # Use intensity to scale: higher TI → higher nu_tilde
    nu_tilde_inlet = T(3) * intensity * U_mag * length_scale
    return Dict{Symbol, AbstractBoundaryCondition}(
        :nu_tilde => ParabolicDirichlet(nu_tilde_inlet),
    )
end

"""
    turbulence_wall_bc(model::StandardKEpsilon)

Generate wall BCs for k-ε (zero-gradient for k, fixed ε via wall function).
For now returns Neumann(0) for both — the wall function values are
set dynamically during the solve.
"""
function turbulence_wall_bc(::StandardKEpsilon)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicNeumann(0.0),
        :epsilon => ParabolicNeumann(0.0),
    )
end

function turbulence_wall_bc(::Union{KOmega, KOmegaSSTModel})
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicNeumann(0.0),
        :omega => ParabolicNeumann(0.0),
    )
end

function turbulence_wall_bc(::SpalartAllmaras)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :nu_tilde => ParabolicDirichlet(0.0),
    )
end

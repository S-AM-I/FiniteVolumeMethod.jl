# thermal/energy_equation.jl — Energy equation assembly for incompressible flow
#
# Assembles the temperature transport equation:
#   ∂T/∂t + div(phi·T) = div(alpha_eff · grad(T))
# where alpha_eff = k_eff / (rho·Cp) is the effective thermal diffusivity.
# The equation is divided by rho·Cp so convection uses phi directly.

"""
    assemble_energy!(
        eq::CollocatedEquation{T},
        T_field::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        alpha_eff::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
    )

Assemble the energy (temperature) transport equation into `eq`.

The equation is scaled by `1/(rho*Cp)` so that:
- Convection uses the volumetric face flux `phi` directly
- Diffusion uses thermal diffusivity `alpha_eff = k_eff/(rho*Cp)`
- Temporal term has unit density coefficient

# Arguments
- `eq` — equation (modified in-place)
- `T_field` — current temperature field (for temporal term)
- `phi` — face volumetric flux from the flow solver
- `alpha_eff` — effective thermal diffusivity: scalar or per-cell vector
- `mesh` — unstructured FVM mesh
- `bcs_T` — temperature boundary conditions
- `dt` — time step (nothing for steady state)
"""
function assemble_energy!(
        eq::CollocatedEquation{T},
        T_field::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        alpha_eff::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
    ) where {Dim, T}
    # Convection: div(phi · T)
    assemble_convection!(eq, phi, mesh, bcs_T)

    # Diffusion: div(alpha_eff · grad(T))
    assemble_laplacian!(eq, alpha_eff, mesh, bcs_T)

    # Temporal term (if transient)
    if dt !== nothing
        assemble_ddt_euler!(eq, one(T), T_field.internal, mesh, dt)
    end

    return nothing
end

"""
    update_k_eff!(
        thermal_state::ThermalState{T},
        thermal_props::FluidThermalProperties{Dim, T},
        nu_t::Union{Nothing, Vector{T}},
        density::T,
    )

Update effective thermal conductivity from turbulent viscosity:
  `k_eff[c] = k_laminar + rho * Cp * nu_t[c] / Pr_t`

When `nu_t` is `nothing`, uses laminar conductivity only.
"""
function update_k_eff!(
        thermal_state::ThermalState{T},
        thermal_props::FluidThermalProperties{Dim, T},
        nu_t::Union{Nothing, Vector{T}},
        density::T,
    ) where {Dim, T}
    k_lam = thermal_props.k
    for c in eachindex(thermal_state.k_eff)
        if nu_t === nothing
            thermal_state.k_eff[c] = k_lam
        else
            k_t = density * thermal_props.Cp * nu_t[c] / thermal_props.Pr_t
            thermal_state.k_eff[c] = k_lam + k_t
        end
    end
    return nothing
end

"""
    compute_alpha_eff(k_eff::Vector{T}, rho::T, Cp::T) -> Vector{T}

Compute thermal diffusivity `alpha_eff = k_eff / (rho * Cp)`.
"""
function compute_alpha_eff(k_eff::Vector{T}, rho::T, Cp::T) where {T}
    rho_Cp = rho * Cp
    alpha = Vector{T}(undef, length(k_eff))
    for c in eachindex(k_eff)
        alpha[c] = k_eff[c] / rho_Cp
    end
    return alpha
end

# Enthalpy formulation lives in a sibling file to keep this module small.
include("enthalpy_equation.jl")

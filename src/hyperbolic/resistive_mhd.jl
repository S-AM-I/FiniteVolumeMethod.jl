# ============================================================
# Resistive MHD Equations
# ============================================================
#
# Extends ideal MHD with magnetic diffusivity η (resistivity).
#
# The inviscid (hyperbolic) part is identical to ideal MHD.
# The resistive part adds:
#   - B diffusion: η∇²B
#   - Ohmic heating: η|J|²
#
# Ohm's law with resistivity:
#   E = -v×B + ηJ,  where  J = ∇×B  (current density)
#
# Induction equation:
#   ∂B/∂t + ∇×E = 0
#   ∂B/∂t = ∇×(v×B) + η∇²B
#
# Conserved:  U = [ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz]
# Primitive:  W = [ρ, vx, vy, vz, P, Bx, By, Bz]

"""
    ResistiveMHDEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}

The resistive magnetohydrodynamics equations in `Dim` spatial dimensions.

Wraps `IdealMHDEquations` and adds magnetic diffusivity η. All inviscid
interface methods (`physical_flux`, `wave_speeds`, etc.) are delegated to
the underlying ideal MHD equations.

The resistive terms add parabolic (diffusion) contributions to the
magnetic field evolution and Ohmic heating to the energy equation:
  `∂B/∂t = ∇×(v×B) + η∇²B`
  `∂E/∂t = ... + η|J|²`

# Fields
- `eos::EOS`: Equation of state.
- `eta::Float64`: Magnetic diffusivity (resistivity / μ₀).
"""
struct ResistiveMHDEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}
    eos::EOS
    eta::Float64
end

function ResistiveMHDEquations{Dim}(eos::EOS; eta = 0.0) where {Dim, EOS <: AbstractEOS}
    return ResistiveMHDEquations{Dim, EOS}(eos, eta)
end

# ============================================================
# Internal helper: construct the equivalent IdealMHDEquations
# ============================================================

@inline _ideal_mhd(law::ResistiveMHDEquations{Dim}) where {Dim} = IdealMHDEquations{Dim}(law.eos)

# ============================================================
# Delegation to IdealMHDEquations
# ============================================================

nvariables(::ResistiveMHDEquations) = 8

@inline conserved_to_primitive(law::ResistiveMHDEquations, u::SVector{8}) = conserved_to_primitive(_ideal_mhd(law), u)
@inline primitive_to_conserved(law::ResistiveMHDEquations, w::SVector{8}) = primitive_to_conserved(_ideal_mhd(law), w)
@inline physical_flux(law::ResistiveMHDEquations, w::SVector{8}, dir::Int) = physical_flux(_ideal_mhd(law), w, dir)
@inline max_wave_speed(law::ResistiveMHDEquations, w::SVector{8}, dir::Int) = max_wave_speed(_ideal_mhd(law), w, dir)
@inline wave_speeds(law::ResistiveMHDEquations, w::SVector{8}, dir::Int) = wave_speeds(_ideal_mhd(law), w, dir)
@inline fast_magnetosonic_speed(law::ResistiveMHDEquations, w::SVector{8}, dir::Int) = fast_magnetosonic_speed(_ideal_mhd(law), w, dir)

# ============================================================
# HLLD Riemann solver forwarding
# ============================================================

function solve_riemann(s::HLLDSolver, law::ResistiveMHDEquations, wL::SVector{8}, wR::SVector{8}, dir::Int)
    return solve_riemann(s, _ideal_mhd(law), wL, wR, dir)
end

# ============================================================
# Resistive Flux Functions (Parabolic Terms)
# ============================================================

"""
    resistive_flux_x(law::ResistiveMHDEquations, uL::SVector{8}, uR::SVector{8}, dx) -> SVector{8}

Compute the resistive (parabolic) flux correction at an x-direction face
between left state `uL` and right state `uR` separated by distance `dx`.

In 1D (x-variation only), the current density components are:
  `J_y = -∂Bz/∂x`,  `J_z = ∂By/∂x`

The resistive diffusion fluxes for the magnetic field in the x-direction:
  `F_By = -η ∂By/∂x`  (diffusion of By)
  `F_Bz = -η ∂Bz/∂x`  (diffusion of Bz)

The Ohmic heating contribution to the energy flux:
  `F_E = -η (J_y Bz_avg - J_z By_avg)`

where the averages are face-centred values.

Returns an `SVector{8}` flux correction to be added to the inviscid face flux.
"""
@inline function resistive_flux_x(law::ResistiveMHDEquations, uL::SVector{8}, uR::SVector{8}, dx)
    η = law.eta
    # B-field differences across the face
    dBy = uR[7] - uL[7]
    dBz = uR[8] - uL[8]

    # Approximate current density at the face
    Jy = -dBz / dx   # J_y = -∂Bz/∂x
    Jz = dBy / dx     # J_z = ∂By/∂x

    # Face-averaged magnetic field
    By_avg = 0.5 * (uL[7] + uR[7])
    Bz_avg = 0.5 * (uL[8] + uR[8])

    # Diffusion flux for B
    By_flux = -η * dBy / dx
    Bz_flux = -η * dBz / dx

    # Ohmic heating contribution to energy flux:
    # The resistive electric field E_res = ηJ contributes to the
    # Poynting flux: -E_res × B. In the x-direction this gives:
    #   F_E = -η (J_y Bz - J_z By)
    E_flux = -η * (Jy * Bz_avg - Jz * By_avg)

    return SVector(zero(η), zero(η), zero(η), zero(η), E_flux, zero(η), By_flux, Bz_flux)
end

"""
    resistive_flux_y(law::ResistiveMHDEquations, uL::SVector{8}, uR::SVector{8}, dy) -> SVector{8}

Compute the resistive (parabolic) flux correction at a y-direction face
between bottom state `uL` and top state `uR` separated by distance `dy`.

In the y-direction, the current density components from y-variation are:
  `J_x = ∂Bz/∂y`,  `J_z = -∂Bx/∂y`

The resistive diffusion fluxes for the magnetic field in the y-direction:
  `G_Bx = -η ∂Bx/∂y`  (diffusion of Bx)
  `G_Bz = -η ∂Bz/∂y`  (diffusion of Bz)

The Ohmic heating contribution to the energy flux:
  `G_E = -η (J_z Bx_avg - J_x Bz_avg)`

Returns an `SVector{8}` flux correction to be added to the inviscid face flux.
"""
@inline function resistive_flux_y(law::ResistiveMHDEquations, uL::SVector{8}, uR::SVector{8}, dy)
    η = law.eta
    # B-field differences across the face
    dBx = uR[6] - uL[6]
    dBz = uR[8] - uL[8]

    # Approximate current density at the face
    Jx = dBz / dy      # J_x = ∂Bz/∂y
    Jz = -dBx / dy     # J_z = -∂Bx/∂y

    # Face-averaged magnetic field
    Bx_avg = 0.5 * (uL[6] + uR[6])
    Bz_avg = 0.5 * (uL[8] + uR[8])

    # Diffusion flux for B
    Bx_flux = -η * dBx / dy
    Bz_flux = -η * dBz / dy

    # Ohmic heating contribution to energy flux:
    # In the y-direction: G_E = -η (J_z Bx - J_x Bz)
    E_flux = -η * (Jz * Bx_avg - Jx * Bz_avg)

    return SVector(zero(η), zero(η), zero(η), zero(η), E_flux, Bx_flux, zero(η), Bz_flux)
end

# ============================================================
# Ohmic Heating
# ============================================================

"""
    ohmic_heating(law::ResistiveMHDEquations, J_sq) -> Real

Compute the volumetric Ohmic heating rate `η |J|²`.

# Arguments
- `law`: The resistive MHD equations (provides η).
- `J_sq`: The squared magnitude of the current density `|J|² = Jx² + Jy² + Jz²`.

# Returns
The Ohmic heating rate `η |J|²`.
"""
@inline function ohmic_heating(law::ResistiveMHDEquations, J_sq)
    return law.eta * J_sq
end

# ============================================================
# CFL with Resistive (Parabolic) Term
# ============================================================

"""
    resistive_dt(law::ResistiveMHDEquations{1}, dx) -> Real

Compute the parabolic CFL time-step limit for the resistive term in 1D:
  `Δt_resistive = 0.5 dx² / η`
"""
@inline function resistive_dt(law::ResistiveMHDEquations{1}, dx)
    return 0.5 * dx^2 / law.eta
end

"""
    resistive_dt(law::ResistiveMHDEquations{2}, dx, dy) -> Real

Compute the parabolic CFL time-step limit for the resistive term in 2D:
  `Δt_resistive = 0.5 / (η (1/dx² + 1/dy²))`
"""
@inline function resistive_dt(law::ResistiveMHDEquations{2}, dx, dy)
    return 0.5 / (law.eta * (1 / dx^2 + 1 / dy^2))
end

# ============================================================
# ReflectiveBC forwarding for 1D ResistiveMHD
# ============================================================

function apply_bc_left!(U::AbstractVector, bc::ReflectiveBC, law::ResistiveMHDEquations{1}, ncells::Int, t)
    return apply_bc_left!(U, bc, _ideal_mhd(law), ncells, t)
end

function apply_bc_right!(U::AbstractVector, bc::ReflectiveBC, law::ResistiveMHDEquations{1}, ncells::Int, t)
    return apply_bc_right!(U, bc, _ideal_mhd(law), ncells, t)
end

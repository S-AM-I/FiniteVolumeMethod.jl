"""
    NavierStokesEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}

The compressible Navier-Stokes equations in `Dim` spatial dimensions.

Wraps `EulerEquations` and adds viscous transport coefficients.
All inviscid interface methods (physical_flux, wave_speeds, etc.)
are delegated to the underlying Euler equations.

# Fields
- `euler::EulerEquations{Dim, EOS}`: The inviscid part.
- `mu::Float64`: Dynamic viscosity μ.
- `Pr::Float64`: Prandtl number.
"""
struct NavierStokesEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}
    euler::EulerEquations{Dim, EOS}
    mu::Float64
    Pr::Float64
end

function NavierStokesEquations{Dim}(eos::EOS; mu, Pr = 0.72) where {Dim, EOS <: AbstractEOS}
    return NavierStokesEquations{Dim, EOS}(EulerEquations{Dim}(eos), mu, Pr)
end

"""
    thermal_conductivity(ns::NavierStokesEquations) -> κ

Compute the thermal conductivity `κ = μ γ / (Pr (γ - 1))`.
"""
@inline function thermal_conductivity(ns::NavierStokesEquations)
    γ = ns.euler.eos.gamma
    return ns.mu * γ / (ns.Pr * (γ - 1))
end

# ============================================================
# Delegation to EulerEquations
# ============================================================

nvariables(ns::NavierStokesEquations{1}) = 3
nvariables(ns::NavierStokesEquations{2}) = 4

@inline physical_flux(ns::NavierStokesEquations, w::SVector, dir::Int) = physical_flux(ns.euler, w, dir)
@inline max_wave_speed(ns::NavierStokesEquations, w::SVector, dir::Int) = max_wave_speed(ns.euler, w, dir)
@inline wave_speeds(ns::NavierStokesEquations, w::SVector, dir::Int) = wave_speeds(ns.euler, w, dir)

@inline conserved_to_primitive(ns::NavierStokesEquations{1}, u::SVector{3}) = conserved_to_primitive(ns.euler, u)
@inline primitive_to_conserved(ns::NavierStokesEquations{1}, w::SVector{3}) = primitive_to_conserved(ns.euler, w)
@inline conserved_to_primitive(ns::NavierStokesEquations{2}, u::SVector{4}) = conserved_to_primitive(ns.euler, u)
@inline primitive_to_conserved(ns::NavierStokesEquations{2}, w::SVector{4}) = primitive_to_conserved(ns.euler, w)

# ============================================================
# HLLC forwarding — HLLC dispatches on EulerEquations{D}
# ============================================================

function solve_riemann(s::HLLCSolver, ns::NavierStokesEquations{1}, wL::SVector{3}, wR::SVector{3}, dir::Int)
    return solve_riemann(s, ns.euler, wL, wR, dir)
end

function solve_riemann(s::HLLCSolver, ns::NavierStokesEquations{2}, wL::SVector{4}, wR::SVector{4}, dir::Int)
    return solve_riemann(s, ns.euler, wL, wR, dir)
end

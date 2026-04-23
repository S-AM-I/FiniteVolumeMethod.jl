# aeroacoustics/fwh.jl — Ffowcs-Williams & Hawkings far-field noise (Stage 6f)
#
# Predicts acoustic pressure p'(x, t) at a far-field observer from
# time-resolved near-field CFD data on a control surface `S`. The FW-H
# equation decomposes the noise into three source terms:
#
#   monopole   (thickness) — fluid mass flux across S
#   dipole     (loading)   — unsteady forces on S
#   quadrupole             — volume-integrated turbulent stresses
#                            (ignored here — dominant only at high Mach)
#
# This skeleton computes the **retarded-time convective Curle integral**
# for stationary surfaces (the simplest FW-H variant, exact for
# stationary bodies with M_source = 0) and provides the machinery to
# accumulate monopole + dipole contributions. Moving-surface and
# porous-FW-H variants are Stage 6f follow-ups.
#
# References: Ffowcs-Williams & Hawkings (1969), Phil. Trans. R. Soc.;
# Farassat (1975), NASA TR R-451 — "Formulation 1A".

using StaticArrays: SVector
using LinearAlgebra: norm, dot

"""
    FWHSurface{Dim, T}

A closed control surface composed of face patches on the CFD mesh.
Each face carries time-resolved pressure and mass-flux samples.
"""
struct FWHSurface{Dim, T}
    face_indices::Vector{Int}                 # mesh face indices forming S
    face_centers::Vector{SVector{Dim, T}}     # cached for retarded-time distance
    face_normals::Vector{SVector{Dim, T}}     # outward unit normals (pointing out of body)
    face_areas::Vector{T}                     # |S_f|
end

"""
    FWHObserver{Dim, T}(position, c_inf = 343.0, rho_inf = 1.225)

Far-field observer. `position` is the observer location in the CFD
frame; `c_inf` and `rho_inf` are the ambient speed of sound and density.
"""
struct FWHObserver{Dim, T}
    position::SVector{Dim, T}
    c_inf::T
    rho_inf::T
end
FWHObserver(position::SVector{Dim, T}; c_inf::Real = 343.0, rho_inf::Real = 1.225) where {Dim, T} =
    FWHObserver{Dim, T}(position, T(c_inf), T(rho_inf))

"""
    curle_dipole_pressure(observer, surface, p_surface, p_inf) -> T

Curle's (1955) approximation: for a compact, stationary solid body the
far-field acoustic pressure is dominated by the dipole (loading) term:

    p'(x, t) ≈ 1/(4π c∞) · ∂/∂t ∫_S [ (x - y) · n̂(y) / r² ] · (p - p∞) dS(y)

This routine returns an instantaneous estimate assuming `p_surface`
already represents the time-fluctuating component
`(p(y, t - r/c∞) - p∞)` at retarded time. The user must provide the
time-differentiated surface pressure if higher-order terms are needed;
this MVP returns the "equivalent compact-dipole pressure" used to
validate the surface-integration machinery.
"""
function curle_dipole_pressure(
        observer::FWHObserver{Dim, T},
        surface::FWHSurface{Dim, T},
        p_surface::AbstractVector{T},
        p_inf::T,
    ) where {Dim, T}
    length(p_surface) == length(surface.face_indices) ||
        error("p_surface length ≠ number of FW-H faces")

    sum_term = zero(T)
    @inbounds for i in eachindex(surface.face_indices)
        y = surface.face_centers[i]
        n_hat = surface.face_normals[i]
        A = surface.face_areas[i]
        r_vec = observer.position - y
        r = norm(r_vec)
        r > T(1.0e-12) || continue
        r_hat = r_vec / r

        # (r̂ · n̂) / r  weighting for compact dipole
        weight = dot(r_hat, n_hat) / r
        sum_term += weight * (p_surface[i] - p_inf) * A
    end

    return sum_term / (T(4) * T(pi) * observer.c_inf)
end

"""
    fwh_monopole_pressure(observer, surface, mass_flux, dmass_flux_dt) -> T

FW-H monopole (thickness) term at the observer from a stationary
surface. `mass_flux[i]` is the instantaneous mass flux `ρ u · n̂` per
face, and `dmass_flux_dt[i]` is its time derivative. The far-field
pressure contribution is:

    p'_T ≈ 1/(4π) · ∫_S (∂ρu·n̂/∂t) / r dS(y)

Time derivatives are supplied by the caller since FW-H is a
time-domain integration; callers typically cache two successive time
samples and finite-difference.
"""
function fwh_monopole_pressure(
        observer::FWHObserver{Dim, T},
        surface::FWHSurface{Dim, T},
        dmass_flux_dt::AbstractVector{T},
    ) where {Dim, T}
    length(dmass_flux_dt) == length(surface.face_indices) ||
        error("dmass_flux_dt length ≠ number of FW-H faces")

    sum_term = zero(T)
    @inbounds for i in eachindex(surface.face_indices)
        y = surface.face_centers[i]
        A = surface.face_areas[i]
        r = norm(observer.position - y)
        r > T(1.0e-12) || continue
        sum_term += dmass_flux_dt[i] * A / r
    end
    return sum_term / (T(4) * T(pi))
end

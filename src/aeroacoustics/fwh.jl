# aeroacoustics/fwh.jl — Ffowcs-Williams & Hawkings far-field noise (Stage 6f)
#
# Predicts acoustic pressure p'(x, t) at a far-field observer from
# time-resolved near-field CFD data on a control surface `S`. The FW-H
# equation decomposes the noise into three source terms:
#
#   thickness (monopole)    p'_T  ← mass flow through S
#   loading   (dipole)      p'_L  ← surface pressure
#   quadrupole              p'_Q  ← volume-integrated Lighthill stresses
#                                    (dominant only at high Mach; stubbed here)
#
# This module ships with
#
#   - a simplified, instantaneous (single-snapshot) FW-H-style surface
#     integration for a stationary observer: static thickness + loading
#     surface sums with an optional Doppler factor. It does NOT evaluate
#     retarded times or the time derivatives of the integrands, so it is
#     not Farassat Formulation 1A — treat outputs as qualitative
#     compact-source estimates only;
#   - `CurleSurface` — hard-wall variant (U_n ≡ 0 → only the loading
#     dipole survives);
#   - `LighthillVolume` — stub volume integral of the Lighthill tensor
#     T_ij, emitting `@warn` because the second-derivative
#     pre-computation is prohibitively expensive for routine use.
#
# References: Ffowcs-Williams & Hawkings (1969), Phil. Trans. R. Soc.;
# Farassat (1975), NASA TR R-451 — "Formulation 1A"; Curle (1955).

using StaticArrays: SVector
using LinearAlgebra: norm, dot

"""
    FWHSurface{Dim, T}

A closed porous control surface composed of face patches. Each face
carries outward unit normal, area, and — when used with the full FW-H
pressure routine — instantaneous surface pressure and normal fluid
velocity `U_n = u · n̂` (for the porous formulation: `U_n` is the
fluid velocity through the surface, not the surface motion).
"""
struct FWHSurface{Dim, T}
    face_indices::Vector{Int}                 # mesh face indices forming S
    face_centers::Vector{SVector{Dim, T}}     # cached for retarded-time distance
    face_normals::Vector{SVector{Dim, T}}     # outward unit normals (out of body)
    face_areas::Vector{T}                     # |S_f|

    function FWHSurface{Dim, T}(
            face_indices::Vector{Int},
            face_centers::Vector{SVector{Dim, T}},
            face_normals::Vector{SVector{Dim, T}},
            face_areas::Vector{T},
        ) where {Dim, T}
        n = length(face_indices)
        length(face_centers) == n ||
            error("FWHSurface: face_centers length ($(length(face_centers))) ≠ face_indices length ($n)")
        length(face_normals) == n ||
            error("FWHSurface: face_normals length ($(length(face_normals))) ≠ face_indices length ($n)")
        length(face_areas) == n ||
            error("FWHSurface: face_areas length ($(length(face_areas))) ≠ face_indices length ($n)")
        return new{Dim, T}(face_indices, face_centers, face_normals, face_areas)
    end
end
FWHSurface(
    face_indices::Vector{Int},
    face_centers::Vector{SVector{Dim, T}},
    face_normals::Vector{SVector{Dim, T}},
    face_areas::Vector{T},
) where {Dim, T} = FWHSurface{Dim, T}(face_indices, face_centers, face_normals, face_areas)

"""
    CurleSurface{Dim, T}

Curle's (1955) analogy: the body is a hard wall, so the fluid normal
velocity `U_n ≡ 0` and only the dipole (loading) term contributes.
A `CurleSurface` wraps an `FWHSurface` for type-level dispatch so
callers may intentionally select the stationary-hard-wall variant.
"""
struct CurleSurface{Dim, T}
    surface::FWHSurface{Dim, T}
end

"""
    LighthillVolume{Dim, T}

Stub volume-integral surrogate for the Lighthill quadrupole source.
Carries per-cell centres, volumes, and a time-resolved Lighthill stress
tensor field `T_ij(x, t)`. The actual volume-integration kernel is an
expensive second-derivative quadrature and is not implemented here; the
public `lighthill_pressure` entry point issues a `@warn` directing
users toward the far cheaper porous-FW-H surface form.
"""
struct LighthillVolume{Dim, T}
    cell_centers::Vector{SVector{Dim, T}}
    cell_volumes::Vector{T}
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

# -------------------------------------------------------------------------
# Full FW-H porous-surface integration (thickness + loading)
# -------------------------------------------------------------------------

"""
    fwh_thickness_term(observer, surface, U_n) -> T

Thickness (monopole) contribution at the observer

    p'_T(x) = (ρ_0 / 4π) · ∫_S  U_n(y) / (r · (1 − M_r))  dS(y)

for a stationary observer. `U_n[i]` is the instantaneous normal fluid
velocity through face `i` (positive = outward). `M_r` — the observer-
facing Mach number — is zero for a stationary surface in quiescent
fluid; callers wanting the subsonic-advected correction pass a finite
`M_r_vec` to the full FW-H routine below.
"""
function fwh_thickness_term(
        observer::FWHObserver{Dim, T},
        surface::FWHSurface{Dim, T},
        U_n::AbstractVector{T},
    ) where {Dim, T}
    length(U_n) == length(surface.face_indices) ||
        error("fwh_thickness_term: U_n length ≠ number of FW-H faces")

    acc = zero(T)
    @inbounds for i in eachindex(surface.face_indices)
        y = surface.face_centers[i]
        A = surface.face_areas[i]
        r = norm(observer.position - y)
        r > T(1.0e-12) || continue
        acc += U_n[i] * A / r
    end
    return observer.rho_inf * acc / (T(4) * T(pi))
end

"""
    fwh_loading_term(observer, surface, p_surface, p_inf; M_r = zero) -> T

Loading (dipole) contribution at the observer

    p'_L(x) = (1/4π) · ∫_S  [ (p − p∞) · (r̂ · n̂) · (1 − M_r)² /
                              ( r² · (1 − M_r) ) ] dS(y)

Reduces to the compact-dipole far-field form used by Curle when
`M_r = 0` (stationary body): weight = (r̂ · n̂) / r².

`M_r` defaults to zero (stationary surface). Pass a scalar or
per-face vector to include Doppler-scaled subsonic advection.
"""
function fwh_loading_term(
        observer::FWHObserver{Dim, T},
        surface::FWHSurface{Dim, T},
        p_surface::AbstractVector{T},
        p_inf::T;
        M_r::Union{T, AbstractVector{T}} = zero(T),
    ) where {Dim, T}
    length(p_surface) == length(surface.face_indices) ||
        error("fwh_loading_term: p_surface length ≠ number of FW-H faces")

    acc = zero(T)
    @inbounds for i in eachindex(surface.face_indices)
        y = surface.face_centers[i]
        n_hat = surface.face_normals[i]
        A = surface.face_areas[i]
        r_vec = observer.position - y
        r = norm(r_vec)
        r > T(1.0e-12) || continue
        r_hat = r_vec / r

        Mr_i = M_r isa AbstractVector ? M_r[i] : M_r
        one_m_Mr = T(1) - Mr_i
        abs(one_m_Mr) > T(1.0e-12) || continue
        doppler = (one_m_Mr^2) / (r^2 * one_m_Mr)

        acc += (p_surface[i] - p_inf) * dot(r_hat, n_hat) * doppler * A
    end
    return acc / (T(4) * T(pi))
end

"""
    compute_fwh_pressure(surface, observer, p_surface, U_n, p_inf;
                        c_0 = observer.c_inf, M_r = 0) -> T

Sum of thickness + loading FW-H contributions at a stationary observer
in the frequency-free (time-domain, single-sample) static-observer
form. Returns the instantaneous acoustic pressure `p'(x, t)` at the
observer induced by the supplied snapshot of surface data.

Quadrupole is intentionally omitted (see `LighthillVolume`).
"""
function compute_fwh_pressure(
        surface::FWHSurface{Dim, T},
        observer::FWHObserver{Dim, T},
        p_surface::AbstractVector{T},
        U_n::AbstractVector{T},
        p_inf::T;
        M_r::Union{T, AbstractVector{T}} = zero(T),
    ) where {Dim, T}
    p_T = fwh_thickness_term(observer, surface, U_n)
    p_L = fwh_loading_term(observer, surface, p_surface, p_inf; M_r = M_r)
    return p_T + p_L
end

"""
    compute_fwh_pressure(curle::CurleSurface, observer, p_surface, p_inf;
                        M_r = 0) -> T

Curle-variant dispatch: hard wall ⇒ `U_n ≡ 0`, only the loading
dipole remains.
"""
function compute_fwh_pressure(
        curle::CurleSurface{Dim, T},
        observer::FWHObserver{Dim, T},
        p_surface::AbstractVector{T},
        p_inf::T;
        M_r::Union{T, AbstractVector{T}} = zero(T),
    ) where {Dim, T}
    return fwh_loading_term(observer, curle.surface, p_surface, p_inf; M_r = M_r)
end

# -------------------------------------------------------------------------
# Retained legacy entry points (kept for existing callers)
# -------------------------------------------------------------------------

"""
    curle_dipole_pressure(observer, surface, p_surface, p_inf) -> T

Curle's (1955) compact-dipole pressure. Equivalent to
`fwh_loading_term / c_inf` and retained for legacy callers; the factor
`1 / c_inf` preserves the original v1 behaviour which is dimensionally
a compact-source far-field amplitude.
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

        weight = dot(r_hat, n_hat) / r
        sum_term += weight * (p_surface[i] - p_inf) * A
    end

    return sum_term / (T(4) * T(pi) * observer.c_inf)
end

"""
    fwh_monopole_pressure(observer, surface, dmass_flux_dt) -> T

Legacy time-derivative monopole entry point retained for callers that
have already cached `d(ρ u·n̂)/dt` on each face. Use `fwh_thickness_term`
for the snapshot formulation.
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

# -------------------------------------------------------------------------
# Lighthill volume quadrupole — stub
# -------------------------------------------------------------------------

"""
    lighthill_pressure(volume, observer, T_ij_field; c_0 = observer.c_inf) -> T

Volume integral of the Lighthill stress tensor

    p'_Q(x, t) = (1 / (4π c_0²)) ·
                  ∫_V  ∂²T_ij/∂x_i∂x_j (y, t − r/c_0) / r  dV(y)

The pre-computation of the second-derivative field is expensive; this
entry point currently returns `zero(T)` and emits a warning pointing
users at the porous-FW-H surface formulation (`compute_fwh_pressure`)
which captures the quadrupole-in-volume contribution implicitly as long
as the surface encloses the turbulent region.
"""
function lighthill_pressure(
        volume::LighthillVolume{Dim, T},
        observer::FWHObserver{Dim, T},
        T_ij_field::AbstractVector;
        c_0::T = observer.c_inf,
    ) where {Dim, T}
    @warn "Lighthill volume integration expensive; prefer FW-H porous surface" maxlog = 1
    return zero(T)
end

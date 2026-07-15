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
#   - `fwh_farassat1a` — retarded-time Farassat Formulation 1A for a
#     STATIC (non-moving) impermeable or permeable surface, stationary
#     observer, quiescent medium. Takes per-face time series of surface
#     pressure and normal velocity, evaluates the source-time
#     derivatives numerically, and sums per-face contributions at the
#     per-face emission time t = τ + r/c. This is the quantitative
#     entry point — validated against analytic monopole/dipole
#     solutions in test/v_and_v_fwh.jl;
#   - a simplified, instantaneous (single-snapshot) FW-H-style surface
#     integration for a stationary observer: static thickness + loading
#     surface sums. It does NOT evaluate retarded times or the time
#     derivatives of the integrands, so it is not Farassat Formulation
#     1A — treat outputs as qualitative compact-source estimates only;
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
velocity through face `i` (positive = outward).

Static snapshot approximation: no retarded time and no source-time
derivative `∂U_n/∂τ` (the integrand is dimensionally `ρ_0 U_n / r`,
not the Formulation-1A `ρ_0 U̇_n / r`). Use [`fwh_farassat1a`](@ref)
for quantitative predictions.
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

Reduces to the compact-dipole near-field form used by Curle when
`M_r = 0` (stationary body): weight = (r̂ · n̂) / r².

Static snapshot approximation: no retarded time, and only the
near-field `1/r²` part of the loading term is kept — the far-field
`ṗ cosθ / (c r)` term requires a pressure time series and lives in
[`fwh_farassat1a`](@ref). For a static surface the Doppler factor is
identically 1; the `M_r` keyword exists only for moving-surface
extensions and should be left at zero here.
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

Static snapshot approximation — no retarded time, no source-time
derivatives. For quantitative acoustics use [`fwh_farassat1a`](@ref)
with per-face time series.

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
# Farassat Formulation 1A — retarded-time FW-H for static surfaces
# -------------------------------------------------------------------------

# Validates that `times` is a uniformly spaced, strictly increasing grid
# and returns the step Δt.
function _fwh_uniform_dt(times::AbstractVector{T}) where {T}
    nt = length(times)
    nt >= 3 || error("fwh_farassat1a: need at least 3 time samples, got $nt")
    dt = (times[end] - times[1]) / (nt - 1)
    dt > zero(T) || error("fwh_farassat1a: times must be strictly increasing")
    @inbounds for k in 1:(nt - 1)
        step = times[k + 1] - times[k]
        abs(step - dt) <= T(1.0e-6) * dt || error(
            "fwh_farassat1a: times must be uniformly spaced " *
                "(Δt deviates at index $k: $step vs $dt)",
        )
    end
    return dt
end

# Source-time derivative of per-face histories `f` (nfaces × ntimes):
# second-order central differences in the interior, second-order
# one-sided stencils at the two endpoints.
function _fwh_source_time_derivative(f::AbstractMatrix{T}, dt::T) where {T}
    nf, nt = size(f)
    ddt = Matrix{T}(undef, nf, nt)
    inv2dt = one(T) / (2 * dt)
    @inbounds for i in 1:nf
        ddt[i, 1] = (-T(3) * f[i, 1] + T(4) * f[i, 2] - f[i, 3]) * inv2dt
        ddt[i, nt] = (T(3) * f[i, nt] - T(4) * f[i, nt - 1] + f[i, nt - 2]) * inv2dt
    end
    @inbounds for k in 2:(nt - 1), i in 1:nf
        ddt[i, k] = (f[i, k + 1] - f[i, k - 1]) * inv2dt
    end
    return ddt
end

# Shared 1A kernel. `dUn_dt === nothing` selects the loading-only
# (Curle hard-wall) variant. All inputs are already validated.
function _fwh_farassat1a_core(
        surface::FWHSurface{Dim, T},
        observer::FWHObserver{Dim, T},
        times::AbstractVector{T},
        dt::T,
        delta_p::AbstractMatrix{T},
        dp_dt::AbstractMatrix{T},
        dUn_dt::Union{Nothing, AbstractMatrix{T}},
    ) where {Dim, T}
    nf = length(surface.face_indices)
    nt = length(times)
    c = observer.c_inf
    four_pi = T(4) * T(pi)

    radii = Vector{T}(undef, nf)
    cos_theta = Vector{T}(undef, nf)
    @inbounds for i in 1:nf
        r_vec = observer.position - surface.face_centers[i]
        r = norm(r_vec)
        r > T(1.0e-12) ||
            error("fwh_farassat1a: observer coincides with FW-H face $i")
        radii[i] = r
        cos_theta[i] = dot(r_vec, surface.face_normals[i]) / r
    end
    r_min, r_max = extrema(radii)

    # Observer (advanced-time) grid: every sample must be reachable from
    # source data on [times[1], times[end]] for ALL faces, i.e.
    # t ∈ [times[1] + r_max/c, times[end] + r_min/c].
    t_start = times[1] + r_max / c
    t_stop = times[end] + r_min / c
    n_obs = floor(Int, (t_stop - t_start) / dt + T(1.0e-9)) + 1
    n_obs >= 2 || error(
        "fwh_farassat1a: source recording too short — after retarded-time " *
            "trimming the observer window holds < 2 samples; extend the " *
            "time series by ≥ $((r_max - r_min) / c) s",
    )
    t_obs = range(t_start; step = dt, length = n_obs)

    p_thickness = zeros(T, n_obs)
    p_loading = zeros(T, n_obs)
    s_max = T(nt - 1)

    @inbounds for i in 1:nf
        A = surface.face_areas[i]
        r = radii[i]
        ct = cos_theta[i]
        w_thick = observer.rho_inf * A / (four_pi * r)
        w_load_far = ct * A / (four_pi * c * r)
        w_load_near = ct * A / (four_pi * r * r)
        # Fractional source-sample index of the first observer sample's
        # emission time τ = t_obs[1] − r/c for this face.
        s0 = (t_start - r / c - times[1]) / dt
        for k in 1:n_obs
            s = clamp(s0 + (k - 1), zero(T), s_max)
            j = min(unsafe_trunc(Int, s), nt - 2)  # 0-based lower bracket
            xi = s - j
            jj = j + 1
            dp = (one(T) - xi) * delta_p[i, jj] + xi * delta_p[i, jj + 1]
            dpd = (one(T) - xi) * dp_dt[i, jj] + xi * dp_dt[i, jj + 1]
            p_loading[k] += w_load_far * dpd + w_load_near * dp
            if dUn_dt !== nothing
                dud = (one(T) - xi) * dUn_dt[i, jj] + xi * dUn_dt[i, jj + 1]
                p_thickness[k] += w_thick * dud
            end
        end
    end
    return (
        t = t_obs,
        p = p_thickness .+ p_loading,
        p_thickness = p_thickness,
        p_loading = p_loading,
    )
end

"""
    fwh_farassat1a(surface, observer, times, p_surface, U_n; p_inf = 0)
        -> (; t, p, p_thickness, p_loading)

Retarded-time Ffowcs-Williams & Hawkings prediction in Farassat's
Formulation 1A, specialised to a STATIC (non-moving) impermeable or
permeable data surface, a stationary observer, and a quiescent medium
— so the Doppler factor `1 − M_r` is identically 1 and no `M_r`
parameter exists (moving-surface extensions would reintroduce it).

    4π p'_T(x, t) = ∫_S [ ρ_∞ ∂U_n/∂τ / r ]_ret dS          (thickness)
    4π p'_L(x, t) = ∫_S [ ∂Δp/∂τ · cosθ / (c_∞ r) ]_ret dS   (loading, far)
                  + ∫_S [ Δp · cosθ / r² ]_ret dS            (loading, near)

with `Δp = p − p_∞`, `cosθ = r̂ · n̂`, and every integrand evaluated at
the per-face emission time `τ = t − r/c_∞`. For the permeable
formulation `U_n` is the fluid normal velocity through the surface
(`ρ u_n / ρ_∞ ≈ u_n` to linear order); for an impermeable wall pass
`U_n ≡ 0` or use the [`CurleSurface`](@ref) method.

Arguments:
- `times` — uniformly spaced source-time grid (length `nt ≥ 3`).
  `Δt` must resolve the highest significant source frequency: the
  central-difference derivative and the linear source-time
  interpolation each carry an `O((ωΔt)²)` amplitude error, so use
  ≥ ~50 samples per period for < 0.5 % error.
- `p_surface`, `U_n` — per-face histories, size `(nfaces, nt)`;
  `p_surface[i, k]` is the pressure on face `i` at `times[k]`.
- `p_inf` — ambient pressure subtracted from `p_surface`.

Returns a named tuple: `t` is the observer (advanced) time grid
`[times[1] + r_max/c, times[end] + r_min/c]` at the same `Δt`
(trimmed so every face has source data at its emission time), `p` the
acoustic pressure series at the observer, and `p_thickness` /
`p_loading` its two components. Quadrupole is omitted; enclose the
nonlinear source region with a permeable surface to capture it.
"""
function fwh_farassat1a(
        surface::FWHSurface{Dim, T},
        observer::FWHObserver{Dim, T},
        times::AbstractVector{T},
        p_surface::AbstractMatrix{T},
        U_n::AbstractMatrix{T};
        p_inf::T = zero(T),
    ) where {Dim, T}
    nf = length(surface.face_indices)
    dt = _fwh_uniform_dt(times)
    size(p_surface) == (nf, length(times)) || error(
        "fwh_farassat1a: p_surface size $(size(p_surface)) ≠ (nfaces, ntimes) = " *
            "($nf, $(length(times)))",
    )
    size(U_n) == size(p_surface) ||
        error("fwh_farassat1a: U_n size $(size(U_n)) ≠ p_surface size $(size(p_surface))")

    delta_p = p_surface .- p_inf
    dp_dt = _fwh_source_time_derivative(delta_p, dt)
    dUn_dt = _fwh_source_time_derivative(U_n, dt)
    return _fwh_farassat1a_core(surface, observer, times, dt, delta_p, dp_dt, dUn_dt)
end

"""
    fwh_farassat1a(curle::CurleSurface, observer, times, p_surface; p_inf = 0)
        -> (; t, p, p_thickness, p_loading)

Curle (hard-wall) variant of [`fwh_farassat1a`](@ref): `U_n ≡ 0`, so
only the loading term — far-field `∂Δp/∂τ · cosθ / (c r)` plus
near-field `Δp · cosθ / r²`, both at retarded time — survives.
`p_thickness` is returned as all-zero for interface uniformity. This
is the quantitative, time-series replacement for
[`curle_dipole_pressure`](@ref).
"""
function fwh_farassat1a(
        curle::CurleSurface{Dim, T},
        observer::FWHObserver{Dim, T},
        times::AbstractVector{T},
        p_surface::AbstractMatrix{T};
        p_inf::T = zero(T),
    ) where {Dim, T}
    surface = curle.surface
    nf = length(surface.face_indices)
    dt = _fwh_uniform_dt(times)
    size(p_surface) == (nf, length(times)) || error(
        "fwh_farassat1a: p_surface size $(size(p_surface)) ≠ (nfaces, ntimes) = " *
            "($nf, $(length(times)))",
    )
    delta_p = p_surface .- p_inf
    dp_dt = _fwh_source_time_derivative(delta_p, dt)
    return _fwh_farassat1a_core(surface, observer, times, dt, delta_p, dp_dt, nothing)
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

Deprecated for quantitative use: this static snapshot keeps only a
`Δp / r` weight with no retarded time and no `∂Δp/∂τ`. Prefer
`fwh_farassat1a(CurleSurface(surface), observer, times, p_surface)`
which evaluates the full retarded-time loading term from the pressure
time series.
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

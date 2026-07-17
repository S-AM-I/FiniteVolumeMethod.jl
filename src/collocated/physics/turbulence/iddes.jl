# turbulence/iddes.jl — Improved Delayed Detached Eddy Simulation
#
# STATUS (v3.1 Wave, Agent D): FULL SHIELDING IMPLEMENTATION.
#
# Shur, Spalart, Strelets & Travin (IJHFF 2008) IDDES formulation on top
# of a Spalart–Allmaras RANS core. Compared to SA-DDES, IDDES adds:
#
# 1. A near-wall cell-based blending function
#        f_B = min(2·exp(-9·α²), 1)
#    where
#        α = 0.25 − d_w / h_max
#    so that f_B saturates at 1 for small d_w (wall-modelled LES zone)
#    and decays to 0 for d_w / h_max ≫ 1 (SA-DDES reduces to SA-RANS or
#    pure LES, matching the DDES behaviour there).
#
# 2. A modified delayed-detached shielding
#        f_d_tilde = max(1 − f_dt, f_B)
#    with
#        r_dt = ν_t / (κ²·d_w²·S)
#        f_dt = 1 − tanh((8·r_dt)³)
#    so that the RANS side of the blend is the DDES sensor when we are
#    outside the WMLES zone (f_B ≈ 0) and the WMLES cell sensor when we
#    are inside it (f_B ≈ 1).
#
# 3. An elevating function f_e which cancels the log-layer mismatch that
#    otherwise appears in the blending zone of a pure DDES:
#        r_dl = ν / (κ²·d_w²·S)
#        f_t  = tanh((C_t²·r_dt)³),  C_t = 1.63
#        f_l  = tanh((C_l²·r_dl)^10), C_l = 3.55
#        f_e2 = 1 − max(f_t, f_l)
#        f_e1 = 2·exp(-11.09·α²)   if α ≥ 0
#             = 2·exp( -9.0·α²)   if α <  0
#        f_e  = f_e2 · max(f_e1 − 1, 0)
#    Clamp to ≥ 0 — the Shur paper explicitly sets the elevating
#    contribution to the max(·, 0) of this expression.
#
# 4. The hybrid length scale
#        L_IDDES = f_d_tilde · (1 + f_e) · L_RANS + (1 − f_d_tilde) · L_LES
#    with L_RANS = d_w (SA convention) and L_LES = C_DES · Δ.
#
# All the composites are implemented as allocation-free scalar helpers so
# the V&V suite can test their algebraic invariants in isolation.
#
# h_max: Shur 2008 specifies the longest edge of the cell. Our
# `UnstructuredFVMMesh` does not carry per-cell edge lengths. We store a
# conservative surrogate `h_max[c] = V_c^(1/Dim)` (i.e. the cubic/square
# root of the cell volume) at construction time; for a Cartesian mesh
# this equals the grid spacing, which is the practical case Shur uses.
# Callers who need the strict longest-edge definition can override by
# passing an explicit `h_max::Vector{T}` to the constructor.

"""
    IDDES{T} <: AbstractHybridModel

Improved Delayed Detached Eddy Simulation (Shur et al. 2008).

Wraps a Spalart–Allmaras RANS core with the full IDDES blending
apparatus: wall-modelled LES branch via `f_B`, delayed-detached
shielding via `f_dt`, a log-layer-mismatch elevating function `f_e`,
and the hybrid length scale
`L_IDDES = f_d_tilde · (1 + f_e) · L_RANS + (1 − f_d_tilde) · L_LES`.

# Fields
- `base_model::SpalartAllmaras{T}` — SA RANS core
- `C_DES::T` — DES constant (default 0.65)
- `C_t::T` — SA-based `r_dt` coefficient (default 1.63)
- `C_l::T` — laminar `r_dl` coefficient (default 3.55)
- `C_w::T` — IDDES filter-width constant (default 0.15) — reserved
  for the Shur `Δ_IDDES` extension; not used with the default
  `h_max` surrogate
- `kappa::T` — von Kármán constant (default 0.41)
- `delta::Vector{T}` — DES grid filter width per cell (volume^(1/Dim))
- `h_max::Vector{T}` — per-cell longest-edge surrogate (default same
  as `delta`)
- `d_wall::Vector{T}` — wall distance per cell

# Construction

```julia
iddes = IDDES(sa, mesh, [:wall]; C_DES = 0.65)
```

The base model must be a `SpalartAllmaras` — the length-scale blend
is SA-specific.
"""
struct IDDES{T} <: AbstractHybridModel
    base_model::SpalartAllmaras{T}
    C_DES::T
    C_t::T
    C_l::T
    C_w::T
    kappa::T
    delta::Vector{T}
    h_max::Vector{T}
    d_wall::Vector{T}
end

"""
    IDDES(base, mesh, wall_patches; C_DES = 0.65, C_t = 1.63,
          C_l = 3.55, C_w = 0.15, kappa = 0.41, h_max = nothing)

Construct a full IDDES model with precomputed filter width, wall
distance, and max-edge-length surrogate. Pass `h_max::Vector{T}`
explicitly to override the default `V_c^(1/Dim)` surrogate.
"""
function IDDES(
        base::SpalartAllmaras{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol};
        C_DES::Real = 0.65,
        C_t::Real = 1.63,
        C_l::Real = 3.55,
        C_w::Real = 0.15,
        kappa::Real = 0.41,
        h_max::Union{Nothing, Vector{T}} = nothing,
    ) where {Dim, T}
    delta = compute_filter_width(mesh)
    d_wall = compute_wall_distance(mesh, wall_patches)
    hmx = h_max === nothing ? copy(delta) : h_max
    length(hmx) == length(delta) ||
        error("IDDES: h_max length ($(length(hmx))) must match number of cells ($(length(delta)))")
    return IDDES{T}(
        base, T(C_DES), T(C_t), T(C_l), T(C_w), T(kappa),
        delta, hmx, T.(d_wall),
    )
end

# ── Interface implementation ─────────────────────────────────────────

n_turbulence_fields(model::IDDES) = n_turbulence_fields(model.base_model)
turbulence_field_names(model::IDDES) = turbulence_field_names(model.base_model)

# ── Scalar algebraic primitives (all allocation-free) ────────────────

"""
    _iddes_r_dt(nu_t, d, S, kappa) -> r_dt

Turbulent sensor ratio `r_dt = ν_t / (κ²·d²·S)` used by `f_dt` and
`f_t`.
"""
function _iddes_r_dt(nu_t::T, d::T, S::T, kappa::T) where {T}
    d_safe = max(d, T(1.0e-10))
    S_safe = max(S, T(1.0e-10))
    return nu_t / (kappa^2 * d_safe^2 * S_safe)
end

"""
    _iddes_r_dl(nu, d, S, kappa) -> r_dl

Laminar sensor ratio `r_dl = ν / (κ²·d²·S)` used by `f_l`.
"""
function _iddes_r_dl(nu::T, d::T, S::T, kappa::T) where {T}
    d_safe = max(d, T(1.0e-10))
    S_safe = max(S, T(1.0e-10))
    return nu / (kappa^2 * d_safe^2 * S_safe)
end

"""
    _iddes_f_dt(r_dt) -> f_dt

DDES-style delayed-detached sensor `f_dt = 1 − tanh((8·r_dt)³)`.
Returns a value in `[0, 1]`: ≈ 1 far from the wall, ≈ 0 near the
wall.
"""
function _iddes_f_dt(r_dt::T) where {T}
    return one(T) - tanh((T(8) * r_dt)^3)
end

"""
    _iddes_alpha(d_w, h_max) -> α

Cell-based blending coordinate `α = 0.25 − d_w / h_max`. Positive
for `d_w < h_max/4` (wall-modelled LES zone) and negative away from
the wall. Not clamped — `f_B` handles the decay via the Gaussian.
"""
function _iddes_alpha(d_w::T, h_max::T) where {T}
    h_safe = max(h_max, T(1.0e-10))
    return T(0.25) - d_w / h_safe
end

"""
    _iddes_f_B(alpha) -> f_B

Wall-cell blending function `f_B = min(2·exp(-9·α²), 1)`. Saturates
at 1 for `|α| ≤ √(ln 2 / 9) ≈ 0.277` and decays to 0 for large `α`.
Always returns a value in `[0, 1]`.
"""
function _iddes_f_B(alpha::T) where {T}
    return min(T(2) * exp(-T(9) * alpha^2), one(T))
end

"""
    _iddes_f_d_tilde(f_dt, f_B) -> f_d_tilde

Modified IDDES shielding `f_d_tilde = max(1 − f_dt, f_B)`. Because
`f_dt ∈ [0, 1]` and `f_B ∈ [0, 1]`, `f_d_tilde ∈ [0, 1]` too.
"""
function _iddes_f_d_tilde(f_dt::T, f_B::T) where {T}
    return max(one(T) - f_dt, f_B)
end

"""
    _iddes_f_e(r_dt, r_dl, alpha, C_t, C_l) -> f_e

Elevating function that cancels log-layer mismatch in the blending
zone. Follows Shur 2008:

```
f_t  = tanh((C_t²·r_dt)³)
f_l  = tanh((C_l²·r_dl)^10)
f_e2 = 1 − max(f_t, f_l)
f_e1 = 2·exp(-11.09·α²)   if α ≥ 0
     = 2·exp( -9.0·α²)   if α <  0
f_e  = max(f_e1 − 1, 0) · f_e2
```

Always non-negative.
"""
function _iddes_f_e(
        r_dt::T, r_dl::T, alpha::T,
        C_t::T, C_l::T,
    ) where {T}
    f_t = tanh((C_t^2 * r_dt)^3)
    f_l = tanh((C_l^2 * r_dl)^10)
    f_e2 = one(T) - max(f_t, f_l)
    f_e1 = alpha >= zero(T) ?
        T(2) * exp(-T(11.09) * alpha^2) :
        T(2) * exp(-T(9.0) * alpha^2)
    return max(f_e1 - one(T), zero(T)) * f_e2
end

"""
    iddes_blended_length(d_wall, delta, h_max, nu, nu_t, S;
                         C_DES = 0.65, C_t = 1.63, C_l = 3.55, kappa = 0.41)
    -> (l_iddes, f_d_tilde, f_e)

Full IDDES hybrid length scale
```
L_IDDES = f_d_tilde · (1 + f_e) · L_RANS + (1 − f_d_tilde) · L_LES
```
with `L_RANS = d_wall` (SA convention) and `L_LES = C_DES · Δ`.
Returns the triple `(l, f_d_tilde, f_e)` for diagnostics; all V&V
tests and the production solver use this helper.
"""
function iddes_blended_length(
        d_wall::T, delta::T, h_max::T,
        nu::T, nu_t::T, S::T;
        C_DES::T = T(0.65),
        C_t::T = T(1.63),
        C_l::T = T(3.55),
        kappa::T = T(0.41),
    ) where {T}
    r_dt = _iddes_r_dt(nu_t, d_wall, S, kappa)
    r_dl = _iddes_r_dl(nu, d_wall, S, kappa)
    f_dt = _iddes_f_dt(r_dt)
    alpha = _iddes_alpha(d_wall, h_max)
    f_B = _iddes_f_B(alpha)
    f_d_tilde = _iddes_f_d_tilde(f_dt, f_B)
    f_e = _iddes_f_e(r_dt, r_dl, alpha, C_t, C_l)
    l_RANS = d_wall
    l_LES = C_DES * delta
    l_iddes = f_d_tilde * (one(T) + f_e) * l_RANS +
        (one(T) - f_d_tilde) * l_LES
    return l_iddes, f_d_tilde, f_e
end

# ── Production viscosity / solve hooks ───────────────────────────────

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::IDDES{T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    turbulent_viscosity!(nu_t, model.base_model, turb_state, mesh)
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::IDDES{T},
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    S_mag = compute_strain_rate(U, mesh)

    # Per-cell IDDES blended length scale
    l_iddes = Vector{T}(undef, nc)
    for c in 1:nc
        l, _, _ = iddes_blended_length(
            model.d_wall[c], model.delta[c], model.h_max[c],
            nu, turb_state.nu_t[c], S_mag[c];
            C_DES = model.C_DES, C_t = model.C_t,
            C_l = model.C_l, kappa = model.kappa,
        )
        l_iddes[c] = l
    end

    # Build a per-call SA model carrying the IDDES length scale as the
    # effective wall distance. Immutable — creates a fresh struct.
    sa = model.base_model
    modified = SpalartAllmaras{T}(
        sa.cb1, sa.cb2, sa.sigma, sa.kappa,
        sa.cw2, sa.cw3, sa.cv1, sa.ct3, sa.ct4,
        l_iddes,
    )
    solve_turbulence!(
        turb_state, modified, U, phi, nu, mesh, bcs_turb;
        dt = dt, linear_solver = linear_solver,
        solver_config = solver_config,
    )
    return nothing
end

# turbulence/wmles.jl — Equilibrium wall-modeled LES
#
# Algebraic log-law wall model applied at the first off-wall cell when the
# computed `y⁺` exceeds the log-layer threshold (typically ~30). The
# model itself is an LES SGS closure (Smagorinsky-like inside the
# domain) but replaces the near-wall viscous-sublayer resolution with an
# equilibrium log-law that yields `τ_w = ρ u_τ²`. Near-wall cells then
# see an effective SGS viscosity
#
#   ν_sgs_wall = ν · (y⁺/u⁺ − 1)
#
# identical to the `compute_nut_wall` expression used by the RANS wall
# functions. Far from walls we fall back to a standard Smagorinsky
# closure.
#
# This is the "equilibrium" flavour of WMLES — no ODE for the wall-layer
# velocity, just a closed-form log law via the Spalding Newton iteration
# already in `wall_functions.jl`.

"""
    EquilibriumWMLES{T} <: AbstractLESModel

Equilibrium wall-modeled LES with an algebraic log-law wall treatment.

Far from walls this model behaves as a classical Smagorinsky SGS
closure (`ν_sgs = (Cs·Δ)² |S|`). At near-wall cells whose computed
`y⁺` exceeds `y_plus_switch` (default 30 — the nominal edge of the
log layer), `ν_sgs` is overwritten by the Spalding-based wall value
`compute_nut_wall(U_par, y, ν)`. At smaller `y⁺` the Smagorinsky
value is kept (the sublayer is assumed to be resolved).

# Fields
- `Cs::T` — Smagorinsky constant for the bulk-flow branch (default 0.1)
- `delta::Vector{T}` — grid filter width per cell
- `wall_cells::Vector{Int}` — indices of cells adjacent to wall faces
- `wall_faces::Vector{Int}` — indices of the corresponding wall faces
  (parallel to `wall_cells`)
- `y_plus_switch::T` — log-layer threshold (default 30)

# Construction

```julia
wmles = EquilibriumWMLES(mesh, [:bottom, :top]; Cs = 0.1, y_plus_switch = 30.0)
```

The wall-patch list is identical to the one supplied to
`NoSlipWallBC` in the incompressible solver.
"""
struct EquilibriumWMLES{T} <: AbstractLESModel
    Cs::T
    delta::Vector{T}
    wall_cells::Vector{Int}
    wall_faces::Vector{Int}
    y_plus_switch::T
end

"""
    EquilibriumWMLES(mesh, wall_patches; Cs = 0.1, y_plus_switch = 30.0)

Construct an equilibrium WMLES model with per-cell filter width and a
precomputed list of wall-adjacent cells for fast lookup inside
`turbulent_viscosity!`.
"""
function EquilibriumWMLES(
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol};
        Cs::Real = 0.1,
        y_plus_switch::Real = 30.0,
    ) where {Dim, T}
    delta = compute_filter_width(mesh)
    wall_cells, wall_faces = _collect_wall_cells(mesh, wall_patches)
    return EquilibriumWMLES{T}(
        T(Cs), delta, wall_cells, wall_faces, T(y_plus_switch),
    )
end

"""
    _collect_wall_cells(mesh, wall_patches) -> (cells, faces)

Return parallel vectors of cell and face indices for every boundary
face whose tag appears in `wall_patches`. A cell listed once per wall
face it touches — duplicates are allowed and harmless for the
overwrite logic.
"""
function _collect_wall_cells(
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol},
    ) where {Dim, T}
    wall_set = Set(wall_patches)
    cells = Int[]
    faces = Int[]
    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        is_internal_face(mesh, f) && continue
        tag = _face_tag(mesh, f)
        tag in wall_set || continue
        push!(cells, owner(mesh, f))
        push!(faces, f)
    end
    return cells, faces
end

"""
    wmles_wall_nut(U_par, y, nu; y_plus_switch = 30) -> (nut, active)

Closed-form WMLES wall viscosity. Returns the Spalding-based
`compute_nut_wall` value when the computed `y⁺` exceeds
`y_plus_switch`, and a sentinel `(0.0, false)` otherwise (caller
keeps the bulk-flow Smagorinsky value). Splitting this from
`turbulent_viscosity!` lets the V&V suite test the primitive in
isolation.
"""
function wmles_wall_nut(
        U_par::T, y::T, nu::T;
        y_plus_switch::T = T(30),
    ) where {T}
    if U_par <= zero(T) || y <= zero(T)
        return zero(T), false
    end
    u_tau = spalding_u_tau(U_par, y, nu)
    y_plus = y * u_tau / max(nu, T(1.0e-20))
    if y_plus < y_plus_switch
        return zero(T), false
    end
    u_plus = max(U_par / max(u_tau, T(1.0e-20)), T(1.0e-10))
    nut = nu * max(y_plus / u_plus - one(T), zero(T))
    return nut, true
end

"""
    wmles_wall_shear(U_par, y, nu, rho) -> τ_w

Closed-form equilibrium wall shear `τ_w = ρ · u_τ²` using the
Spalding iteration. Reported in viscous units (density scales
separately). Used by the post-processing layer and the WMLES V&V
tests.
"""
function wmles_wall_shear(U_par::T, y::T, nu::T, rho::T) where {T}
    if U_par <= zero(T) || y <= zero(T)
        return zero(T)
    end
    u_tau = spalding_u_tau(U_par, y, nu)
    return rho * u_tau^2
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::EquilibriumWMLES{T},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    # Bulk-flow Smagorinsky branch — same form as `Smagorinsky`.
    S_mag = compute_strain_rate(U, mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        nu_t[c] = (model.Cs * model.delta[c])^2 * S_mag[c]
    end

    # Near-wall log-law overwrite. We need a laminar viscosity — for the
    # SGS model there's no problem-level handle, so we pull it from the
    # first nonzero filter-width cell as a stand-in. Callers that want a
    # problem-consistent ν should invoke `wmles_wall_nut` directly from
    # the solver wrapper with the problem's ν; the default below is for
    # standalone LES experiments.
    nu_est = T(1.0e-5)
    for (c, f) in zip(model.wall_cells, model.wall_faces)
        y, U_par = _wall_projection(mesh, c, f, U.internal[c])
        nut_wall, active = wmles_wall_nut(
            U_par, y, nu_est;
            y_plus_switch = model.y_plus_switch,
        )
        if active
            nu_t[c] = nut_wall
        end
    end
    return nothing
end

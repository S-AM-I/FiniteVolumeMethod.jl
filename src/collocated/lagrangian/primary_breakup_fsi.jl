# lagrangian/primary_breakup_fsi.jl — Primary-breakup ↔ ALE-FSI handshake
#
# Lightweight coupling between the Lagrangian primary-breakup models
# (`KHACTBreakup`, `LISABreakup`) and the ALE mesh-motion infrastructure
# (`MeshMotionState`). Given:
#
#   - a shared ALE region specified by a list of interface face tags
#   - a `ParticleTracker` ready to accept injected droplets
#   - a primary-breakup criterion
#
# the handshake walks every ALE interface face, reconstructs the local
# slip velocity from the mesh sweep flux `phi_mesh`, evaluates the
# breakup correlation, and — if the criterion triggers — seeds
# Lagrangian particles at the face centre with velocity drawn from that
# face's interface flow. The mass that leaves the continuum phase is
# accumulated into a per-cell source vector `mass_source` so the caller
# can apply it to the fluid continuity equation as an explicit sink.
#
# The design is intentionally minimal — it is a "handshake" not a full
# two-way coupled primary-atomisation solver. The criterion is:
#
#   |U_slip| > U_crit    AND    breakup timescale τ_b < dt
#
# both evaluated at the face centre. When a face triggers, a single
# droplet of diameter `d_parent` is seeded (one per face per call) and
# its mass `m = ρ_l · π/6 · d_parent³` is subtracted from the owner
# cell's continuity residual. Full DPM coupling (multiple child
# droplets sampled from a Rosin-Rammler distribution, explicit
# child-radius tracking, etc.) is deferred to a later wave.

using LinearAlgebra: norm

"""
    PrimaryBreakupFSIResult{Dim, T}

Return value of [`couple_primary_breakup_fsi!`](@ref). Carries
diagnostics and the mass-deficit vector so the caller can apply it
to the fluid continuity equation.

# Fields
- `n_injected::Int` — number of Lagrangian particles created this call
- `mass_source::Vector{T}` — per-cell mass source to subtract from the
  continuity residual (length `ncells`). Positive entries represent
  mass leaving the continuous phase.
- `triggered_faces::Vector{Int}` — indices of interface faces whose
  criterion triggered breakup during this call
- `total_mass_released::T` — sum of `mass_source` (≥ 0)
"""
struct PrimaryBreakupFSIResult{Dim, T}
    n_injected::Int
    mass_source::Vector{T}
    triggered_faces::Vector{Int}
    total_mass_released::T
end

"""
    _face_interface_velocity(ale_state, mesh, f) -> SVector{Dim, T}

Reconstruct the face-centred interface velocity vector from the ALE
mesh sweep flux:

```
u_f = (phi_mesh[f] / A_f) · n_f
```

where `phi_mesh` is the face sweep flux stored on `ale_state` and
`n_f` is the unit outward face normal. Returns a zero vector when
`A_f` is degenerate.
"""
function _face_interface_velocity(
        ale_state::MeshMotionState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        f::Int,
    ) where {Dim, T}
    A_f = mesh.face_areas[f]
    if A_f <= zero(T)
        return zero(SVector{Dim, T})
    end
    n = SVector{Dim, T}(ntuple(d -> mesh.face_normals[d, f], Val(Dim)))
    u_n = ale_state.phi_mesh[f] / A_f
    return u_n * n
end

"""
    _breakup_timescale(model, d_parent, U_slip_mag, rho_f, sigma, mu_l, rho_l, h_sheet)
        -> (τ_b, d_child)

Dispatch on the concrete primary-breakup model and return the
break-up timescale `τ_b [s]` and the predicted child diameter
`d_child [m]`. `Inf` timescale means the correlation returned "no
break-up".

For `KHACTBreakup` we use the droplet-diameter form (`d_parent` is the
local liquid-column diameter); for `LISABreakup` we use the sheet
form with thickness `h_sheet`. If `h_sheet ≤ 0` the LISA path falls
back to the KH-ACT formula with `d_parent`.
"""
function _breakup_timescale(
        model::KHACTBreakup{T},
        d_parent::T, U_slip_mag::T,
        rho_f::T, sigma::T,
        mu_l::T, rho_l::T, h_sheet::T,
    ) where {T}
    d_child, τ_b = kh_act_breakup(
        d_parent, U_slip_mag, rho_f, rho_l, mu_l, sigma;
        model = model,
    )
    return τ_b, d_child
end

function _breakup_timescale(
        model::LISABreakup{T},
        d_parent::T, U_slip_mag::T,
        rho_f::T, sigma::T,
        mu_l::T, rho_l::T, h_sheet::T,
    ) where {T}
    # LISA requires a sheet thickness; fall back to d_parent as a
    # thickness surrogate when the caller has not supplied one.
    h = h_sheet > zero(T) ? h_sheet : d_parent
    d_child, τ_b = lisa_breakup(
        h, U_slip_mag, rho_f, rho_l, sigma;
        model = model,
    )
    return τ_b, d_child
end

"""
    couple_primary_breakup_fsi!(tracker, breakup_model, ale_state, mesh,
        rho_f, sigma, dt; kwargs...) -> PrimaryBreakupFSIResult

Handshake coupling between a Lagrangian particle tracker and an ALE
mesh-motion state for primary atomisation. Walks every interface
face (tags listed in `interface_patches`), reconstructs the local
slip velocity from `ale_state.phi_mesh`, evaluates the primary
break-up criterion, and on trigger:

1. Injects a single Lagrangian droplet at the face centre, with
   velocity drawn from the interface flow field
   (`u_f = (phi_mesh / A_f) · n_f`).
2. Accumulates the droplet mass `m = ρ_l · π/6 · d_parent³` into the
   owner-cell entry of the returned `mass_source` vector (caller
   applies as a continuity sink).

A face triggers when `|U_slip| > U_crit` *and* `τ_b < dt` — i.e.
both a velocity threshold and a timescale threshold must be
satisfied.

# Required arguments
- `tracker::ParticleTracker{Dim, T}` — mutated in-place (new particles
  appended)
- `breakup_model::AbstractPrimaryBreakupModel` — `KHACTBreakup` or
  `LISABreakup`
- `ale_state::MeshMotionState{Dim, T}` — read-only on this call;
  supplies `phi_mesh`
- `mesh::UnstructuredFVMMesh{Dim, T}`
- `rho_f::T` — continuous-phase (gas) density
- `sigma::T` — surface tension
- `dt::T` — current time step; breakup must complete within this window

# Keyword arguments (defaulted)
- `interface_patches::Vector{Symbol} = [:interface]` — face tags to
  iterate
- `rho_l::T = T(1000)` — liquid density
- `mu_l::T = T(1.0e-3)` — liquid dynamic viscosity
- `d_parent::T = T(1.0e-4)` — parent-droplet / liquid-column diameter
- `h_sheet::T = zero(T)` — LISA sheet thickness (ignored by KH-ACT)
- `U_crit::T = zero(T)` — slip-speed threshold below which no break-up
  is attempted even if `τ_b < dt`

Mass conservation guarantee: when no face triggers, `mass_source` is
all-zero and `tracker.particles` is unchanged. When faces trigger,
`total_mass_released = sum(mass_source) = n_injected · m_per_drop`,
an exact equality.
"""
function couple_primary_breakup_fsi!(
        tracker::ParticleTracker{Dim, T},
        breakup_model::AbstractPrimaryBreakupModel,
        ale_state::MeshMotionState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        rho_f::T, sigma::T, dt::T;
        interface_patches::Vector{Symbol} = Symbol[:interface],
        rho_l::T = T(1000),
        mu_l::T = T(1.0e-3),
        d_parent::T = T(1.0e-4),
        h_sheet::T = zero(T),
        U_crit::T = zero(T),
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    mass_source = zeros(T, nc)
    triggered = Int[]
    n_injected = 0

    patch_set = Set(interface_patches)
    nf = size(mesh.face_cells, 2)
    m_drop = rho_l * T(pi) / T(6) * d_parent^3

    for f in 1:nf
        is_internal_face(mesh, f) && continue
        tag = _face_tag(mesh, f)
        tag in patch_set || continue

        # Reconstruct slip velocity at the face centre.
        U_slip = _face_interface_velocity(ale_state, mesh, f)
        U_slip_mag = norm(U_slip)
        if U_slip_mag <= U_crit
            continue
        end

        τ_b, _ = _breakup_timescale(
            breakup_model, d_parent, U_slip_mag,
            rho_f, sigma, mu_l, rho_l, h_sheet,
        )
        if !(isfinite(τ_b) && τ_b < dt)
            continue
        end

        # Trigger — seed a droplet at the face centre, with velocity
        # drawn from the interface flow field.
        push!(triggered, f)
        pos = face_center(mesh, f)
        # Use the same injection helper that the cone/flat-fan
        # injectors in injection.jl use to guarantee a consistent
        # particle record.
        _push_particle!(
            tracker, pos, U_slip;
            diameter = d_parent,
            density = rho_l,
        )
        n_injected += 1

        # Apply mass deficit to owner cell.
        P = owner(mesh, f)
        mass_source[P] += m_drop
    end

    total_mass = sum(mass_source)
    return PrimaryBreakupFSIResult{Dim, T}(
        n_injected, mass_source, triggered, total_mass,
    )
end

# pressure_based/eos_coupling.jl — Density update + face-viscosity coupling
#
# Stage 3 compressible extension of the pressure-based solver family.
# Provides the glue between an `AbstractThermoModel` and the existing
# incompressible SIMPLE/PIMPLE machinery:
#
#   * `update_density!(rho, model, p, T)` writes ρ = ρ(p, T) per cell
#   * `update_viscosity!(mu, model, T)`  writes μ = μ(T)    per cell
#   * `face_density(model, p_P, p_N, T_P, T_N, w)` interpolates ρ on a face
#   * `density_flux_update!` transforms volumetric φ → mass flux ρφ for
#     the continuity equation of a compressible step
#
# These helpers intentionally do NOT touch the matrix structure — they
# populate plain Julia arrays that the compressible SIMPLE/PIMPLE loops
# feed back into `assemble_momentum!` / `assemble_pressure!` via the
# existing `nu_eff` / `body_force` extension points.

using LinearAlgebra: dot

# ── Per-cell density update ─────────────────────────────────────────

"""
    update_density!(rho, model, p, T)

Write `rho[c] = density_at(model, p[c], T[c])` for every interior cell.

- `rho::Vector{FT}`            — destination density array (length `nc`)
- `model::AbstractThermoModel` — thermodynamic model
- `p::Vector{FT}`              — cell pressures
- `T::Vector{FT}`              — cell temperatures
"""
function update_density!(
        rho::Vector{FT},
        model::AbstractThermoModel,
        p::AbstractVector{FT},
        T::AbstractVector{FT},
    ) where {FT}
    n = length(rho)
    @inbounds for c in 1:n
        rho[c] = FT(density_at(model, p[c], T[c]))
    end
    return nothing
end

"""
    update_density!(rho, model, p, T_const::Real)

Isothermal convenience: use a single scalar temperature for every cell.
"""
function update_density!(
        rho::Vector{FT},
        model::AbstractThermoModel,
        p::AbstractVector{FT},
        T_const::Real,
    ) where {FT}
    Tc = FT(T_const)
    n = length(rho)
    @inbounds for c in 1:n
        rho[c] = FT(density_at(model, p[c], Tc))
    end
    return nothing
end

# ── Per-cell viscosity update ───────────────────────────────────────

"""
    update_viscosity!(mu, model, T)

Write `mu[c] = viscosity_at(model, T[c])` for every interior cell.
Used when molecular viscosity is temperature-dependent (Sutherland law
or a tabulated lookup).
"""
function update_viscosity!(
        mu::Vector{FT},
        model::AbstractThermoModel,
        T::AbstractVector{FT},
    ) where {FT}
    n = length(mu)
    @inbounds for c in 1:n
        mu[c] = FT(viscosity_at(model, T[c]))
    end
    return nothing
end

function update_viscosity!(
        mu::Vector{FT},
        model::AbstractThermoModel,
        T_const::Real,
    ) where {FT}
    Tc = FT(T_const)
    n = length(mu)
    @inbounds for c in 1:n
        mu[c] = FT(viscosity_at(model, Tc))
    end
    return nothing
end

# ── Compressibility ψ = ∂ρ/∂p|_T ────────────────────────────────────

"""
    psi_at(model::AbstractThermoModel, p, T) -> FT

Isothermal compressibility `ψ = ∂ρ/∂p|_T` [s²/m²] used by the
compressible pressure equation (`ddt(ψ p)` term).  Analytic for
`IdealGas` (`ψ = 1/(R T)`); other models fall back to a central finite
difference of `density_at` with step `δ = 1e-4 · max(|p|, 1)` (adequate
because ψ only enters as a Picard-frozen linearization coefficient —
mass conservation is enforced through the matching `ρ ← ρ* + ψ(p - p*)`
update, not through the accuracy of ψ itself).
"""
@inline function psi_at(model::AbstractThermoModel, p, T)
    delta = 1.0e-4 * max(abs(p), one(p))
    return (density_at(model, p + delta, T) - density_at(model, p - delta, T)) /
        (2 * delta)
end

@inline psi_at(m::IdealGas, p, T) = 1 / (m.R * max(T, eps(typeof(float(T)))))

"""
    update_psi!(psi, model, p, T)

Write `psi[c] = psi_at(model, p[c], T[c])` for every interior cell.
"""
function update_psi!(
        psi::Vector{FT},
        model::AbstractThermoModel,
        p::AbstractVector{FT},
        T::AbstractVector{FT},
    ) where {FT}
    @inbounds for c in eachindex(psi)
        psi[c] = FT(psi_at(model, p[c], T[c]))
    end
    return nothing
end

# ── Face-centered density ───────────────────────────────────────────

"""
    face_density(model, p_P, p_N, T_P, T_N, w) -> FT

Linear face interpolation of density using the owner-weight `w`:

```
ρ_f = w · ρ(p_P, T_P) + (1 - w) · ρ(p_N, T_N)
```

This matches the standard OpenFOAM linear face scheme and is consistent
with how the existing `gradient`, `interpolation`, and `face_weight`
helpers interpolate scalar fields.
"""
@inline function face_density(
        model::AbstractThermoModel,
        p_P, p_N, T_P, T_N, w,
    )
    rho_P = density_at(model, p_P, T_P)
    rho_N = density_at(model, p_N, T_N)
    return w * rho_P + (one(w) - w) * rho_N
end

# ── Mass-flux update ────────────────────────────────────────────────

"""
    update_mass_flux!(phi_m, phi, rho_f)

Turn a volumetric face flux `phi_f = U·S` into a mass flux
`phi_m_f = ρ_f · phi_f` for the compressible continuity equation.

- `phi_m::Vector{FT}`  — destination mass-flux array (length = nfaces)
- `phi::Vector{FT}`    — volumetric face flux (`state.phi.values`)
- `rho_f::Vector{FT}`  — face densities (length = nfaces)

`phi_m[f]` has units `kg / s`. Positive values point from owner to
neighbour, matching the sign convention of `FaceFluxField`.
"""
function update_mass_flux!(
        phi_m::Vector{FT},
        phi::AbstractVector{FT},
        rho_f::AbstractVector{FT},
    ) where {FT}
    length(phi_m) == length(phi) == length(rho_f) ||
        error("update_mass_flux!: size mismatch")
    @inbounds for f in eachindex(phi_m)
        phi_m[f] = rho_f[f] * phi[f]
    end
    return nothing
end

# ── Face density field ──────────────────────────────────────────────

"""
    compute_face_densities!(rho_f, model, mesh, p, T, state)

Populate `rho_f::Vector{FT}` (length = nfaces) with face-interpolated
density using the EOS `model`. Internal faces use the standard linear
owner-weight interpolation; boundary faces evaluate the EOS with the
owner-cell `(p, T)` — i.e. a zero-gradient extrapolation of ρ (which
is the standard choice when pressure and temperature BCs are separate).

Signature kept deliberately explicit so callers can pre-allocate
`rho_f` once and reuse it across SIMPLE iterations.
"""
function compute_face_densities!(
        rho_f::Vector{FT},
        model::AbstractThermoModel,
        mesh::UnstructuredFVMMesh{Dim, FT},
        p::AbstractVector{FT},
        T::AbstractVector{FT},
    ) where {Dim, FT}
    nf = size(mesh.face_cells, 2)
    length(rho_f) == nf || error("compute_face_densities!: rho_f length ≠ nfaces")
    @inbounds for f in 1:nf
        P = owner(mesh, f)
        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)
            rho_f[f] = FT(face_density(model, p[P], p[N], T[P], T[N], w))
        else
            rho_f[f] = FT(density_at(model, p[P], T[P]))
        end
    end
    return nothing
end

# ── Pressure-work / dilatation source ───────────────────────────────

"""
    pressure_work_source(U, p, mesh, c) -> FT

Per-cell dilatation work `U·∇p · V_c`. For compressible energy coupling
(rhoPimpleFoam-style) the enthalpy equation picks up this source on
the RHS. Uses the Green-Gauss pressure gradient already in scope via
`gradient(p, mesh)` at the callsite — this helper just exposes the
combination for readability.
"""
@inline function pressure_work_source(
        u_c::SVector{Dim, FT},
        grad_p_c::SVector{Dim, FT},
        V_c::FT,
    ) where {Dim, FT}
    return dot(u_c, grad_p_c) * V_c
end

"""
    add_pressure_work!(eq, U, p, mesh)

Add the pressure-work source `U · ∇p · V_c` to the RHS of the energy
equation for every interior cell. Used when coupling the compressible
continuity update to the temperature transport.
"""
function add_pressure_work!(
        eq::CollocatedEquation{FT},
        U::CollocatedVectorField{Dim, FT},
        p::CollocatedScalarField{FT},
        mesh::UnstructuredFVMMesh{Dim, FT},
    ) where {Dim, FT}
    grad_p = gradient(p, mesh)
    nc = length(mesh.cell_volumes)
    @inbounds for c in 1:nc
        eq.b[c] += pressure_work_source(U.internal[c], grad_p[c], mesh.cell_volumes[c])
    end
    return nothing
end

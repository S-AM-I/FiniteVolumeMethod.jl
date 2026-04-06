# postprocessing/forces.jl — Integrated forces and coefficients
#
# Computes pressure and viscous forces on boundary patches, and
# aerodynamic coefficients (Cd, Cl) from the integrated forces.

using LinearAlgebra: norm, dot

"""
    compute_forces(p, U, nu, mesh, patch)

Compute pressure and viscous forces on boundary `patch`.

Pressure force: `F_p = -sum_f p_f * S_f` (outward-pointing)
Viscous force: `F_v = sum_f tau_w_f * A_f`

Returns `(pressure = SVector, viscous = SVector)`.
"""
function compute_forces(
        p::CollocatedScalarField{T},
        U::CollocatedVectorField{Dim, T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    pbmap = build_boundary_map(p)
    tau_w = compute_wall_shear_stress(U, nu, mesh, patch)

    F_pressure = zero(SVector{Dim, T})
    F_viscous = zero(SVector{Dim, T})

    for (i, f) in enumerate(faces)
        S_f = face_normal_area(mesh, f)
        p_f = p.boundary[pbmap[f]]
        A_f = mesh.face_areas[f]

        F_pressure = F_pressure - p_f * S_f
        F_viscous = F_viscous + tau_w[i] * A_f
    end

    return (pressure = F_pressure, viscous = F_viscous)
end

"""
    force_coefficients(pressure_force, viscous_force; rho_ref, U_ref, A_ref,
        drag_direction, lift_direction)

Compute aerodynamic force coefficients from integrated forces.

- `Cd = (F_total . drag_dir) / (q * A_ref)` where `q = 0.5 * rho * U^2`
- `Cl = (F_total . lift_dir) / (q * A_ref)`
- `Cd_pressure`, `Cd_viscous` for separate contributions

Returns `(Cd, Cl, Cd_pressure, Cd_viscous)` named tuple.
"""
function force_coefficients(
        pressure_force::SVector{Dim, T},
        viscous_force::SVector{Dim, T};
        rho_ref::T,
        U_ref::T,
        A_ref::T,
        drag_direction::SVector{Dim, T} = _default_drag_dir(Val(Dim), T),
        lift_direction::SVector{Dim, T} = _default_lift_dir(Val(Dim), T),
    ) where {Dim, T}
    q = T(0.5) * rho_ref * U_ref^2
    qA = q * A_ref

    F_total = pressure_force + viscous_force

    Cd = qA > zero(T) ? dot(F_total, drag_direction) / qA : zero(T)
    Cl = qA > zero(T) ? dot(F_total, lift_direction) / qA : zero(T)
    Cd_p = qA > zero(T) ? dot(pressure_force, drag_direction) / qA : zero(T)
    Cd_v = qA > zero(T) ? dot(viscous_force, drag_direction) / qA : zero(T)

    return (Cd = Cd, Cl = Cl, Cd_pressure = Cd_p, Cd_viscous = Cd_v)
end

_default_drag_dir(::Val{2}, ::Type{T}) where {T} = SVector{2, T}(one(T), zero(T))
_default_drag_dir(::Val{3}, ::Type{T}) where {T} = SVector{3, T}(one(T), zero(T), zero(T))
_default_lift_dir(::Val{2}, ::Type{T}) where {T} = SVector{2, T}(zero(T), one(T))
_default_lift_dir(::Val{3}, ::Type{T}) where {T} = SVector{3, T}(zero(T), one(T), zero(T))

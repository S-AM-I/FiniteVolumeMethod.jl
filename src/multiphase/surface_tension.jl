# multiphase/surface_tension.jl — Continuum Surface Force (CSF) model
#
# Computes interface curvature from the volume fraction gradient and
# produces a body force F_st = σ · κ · ∇α for the momentum equation.
#
# Also exposes static and dynamic contact-angle models for wall
# adhesion via normal-vector correction at wall faces. The correction
# is applied to the CSF interface normal in the tangent-normal plane
# of the wall so the computed curvature matches the prescribed
# equilibrium (or dynamic) contact angle.

using LinearAlgebra: norm, dot

# ── Contact-angle models ─────────────────────────────────────────────

"""
    AbstractContactAngleModel

Supertype for wall-adhesion models that prescribe an interface-wall
angle θ and rotate the interface normal accordingly.
"""
abstract type AbstractContactAngleModel end

"""
    StaticContactAngle{T}

Static (equilibrium) contact angle model.

At a wall face with outward unit normal `n_wall` and interface tangent
`t_wall` (computed in the tangent-normal plane), the interface normal
is rotated to

    n' = cos(θ_s) · n_wall + sin(θ_s) · t_wall

so that

- θ_s = 90° ⇒ `n' = n_wall` (no correction, CSF unchanged)
- θ_s = 0°  ⇒ `n' = t_wall` (complete wetting, tangent-aligned)
- θ_s = 180° ⇒ `n' = -t_wall` (complete dewetting, anti-tangent)

# Fields
- `theta_s::T` — equilibrium contact angle in **radians**
"""
struct StaticContactAngle{T} <: AbstractContactAngleModel
    theta_s::T
end

"""
    DynamicContactAngle{T}

Cox-Voinov dynamic contact angle model:

    θ³(Ca) = θ_s³ + 9 · Ca · ln(L / L_s)

with capillary number `Ca = μ · U_cl / σ`. At `Ca = 0` the result
reduces to `θ_s` (returned via the helper `cox_voinov_angle`).

# Fields
- `theta_s::T` — static contact angle (radians)
- `mu::T`     — characteristic dynamic viscosity (Pa·s)
- `sigma::T`  — surface tension coefficient (N/m)
- `L::T`      — outer length scale (macroscopic, e.g. capillary length)
- `L_s::T`    — slip / microscopic length scale
"""
struct DynamicContactAngle{T} <: AbstractContactAngleModel
    theta_s::T
    mu::T
    sigma::T
    L::T
    L_s::T
end

"""
    cox_voinov_angle(model::DynamicContactAngle{T}, U_cl::T) -> T

Cox-Voinov prediction of the dynamic contact angle given the
contact-line speed `U_cl`. Returns an angle in radians.

- `U_cl = 0` ⇒ `θ_s` (exact, modulo machine precision)
- Monotonically increasing in `|Ca|` for `ln(L/L_s) > 0`
"""
function cox_voinov_angle(model::DynamicContactAngle{T}, U_cl::T) where {T}
    Ca = model.mu * U_cl / model.sigma
    theta3 = model.theta_s^3 + T(9) * Ca * log(model.L / model.L_s)
    # cbrt is sign-preserving and exact at zero.
    return cbrt(theta3)
end

"""
    apply_contact_angle(
        n_interface::SVector{Dim, T},
        n_wall::SVector{Dim, T},
        model::AbstractContactAngleModel;
        U_cl::T = zero(T),
    ) -> SVector{Dim, T}

Rotate the interface normal `n_interface` to match the angle prescribed
by `model` on a wall with outward unit normal `n_wall`. The tangent
`t_wall` is chosen in the plane of the interface, orthogonal to `n_wall`.

The dynamic model uses `U_cl` (contact-line velocity); the static model
ignores it.
"""
function apply_contact_angle(
        n_interface::SVector{Dim, T},
        n_wall::SVector{Dim, T},
        model::StaticContactAngle{T};
        U_cl::T = zero(T),
    ) where {Dim, T}
    return _rotate_to_angle(n_interface, n_wall, model.theta_s)
end

function apply_contact_angle(
        n_interface::SVector{Dim, T},
        n_wall::SVector{Dim, T},
        model::DynamicContactAngle{T};
        U_cl::T = zero(T),
    ) where {Dim, T}
    theta = cox_voinov_angle(model, U_cl)
    return _rotate_to_angle(n_interface, n_wall, theta)
end

# Internal — rotate interface normal to target angle θ with n_wall.
# Tangent t_wall lies in the plane spanned by n_wall and n_interface,
# perpendicular to n_wall, and points "along the interface" (i.e. the
# component of n_interface orthogonal to n_wall, re-normalised).
@inline function _rotate_to_angle(
        n_interface::SVector{Dim, T},
        n_wall::SVector{Dim, T},
        theta::T,
    ) where {Dim, T}
    # Decompose n_interface = (n·n_w) n_w + t_comp ⇒ tangent component.
    nw_mag = norm(n_wall)
    if nw_mag < T(1.0e-12)
        return n_interface
    end
    n_w_hat = n_wall / nw_mag

    proj = dot(n_interface, n_w_hat)
    t_comp = n_interface - proj * n_w_hat
    t_mag = norm(t_comp)
    if t_mag < T(1.0e-12)
        # Interface normal collinear with wall normal — pick an arbitrary
        # tangent (doesn't matter: static θ=90° gives back n_wall).
        return cos(theta) * n_w_hat
    end
    t_w_hat = t_comp / t_mag
    return cos(theta) * n_w_hat + sin(theta) * t_w_hat
end

# ── Curvature & CSF force ─────────────────────────────────────────────

"""
    compute_curvature(alpha, mesh; contact_angle = nothing, wall_patches = Symbol[]) -> Vector{T}

Compute the interface curvature `κ = -div(∇α / |∇α|)` per cell.

If `contact_angle` is an `AbstractContactAngleModel` and `wall_patches`
lists the face tags treated as solid walls, the interface normal on
those boundary faces is rotated via `apply_contact_angle` before the
divergence is evaluated.

Steps:
1. Compute `∇α` via Green-Gauss gradient
2. Normalize to get interface normal `n̂ = ∇α / |∇α|`
3. Compute `div(n̂)` via face summation, applying wall rotation
4. `κ = -div(n̂)`
"""
function compute_curvature(
        alpha::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        contact_angle::Union{Nothing, AbstractContactAngleModel} = nothing,
        wall_patches::Vector{Symbol} = Symbol[],
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Step 1: gradient of alpha
    grad_alpha = gradient(alpha, mesh)

    # Step 2: normalize to get interface normal per cell
    n_hat = Vector{SVector{Dim, T}}(undef, nc)
    for c in 1:nc
        g_mag = norm(grad_alpha[c])
        if g_mag > T(1.0e-12)
            n_hat[c] = grad_alpha[c] / g_mag
        else
            n_hat[c] = zero(SVector{Dim, T})
        end
    end

    # Step 3: div(n_hat) via face summation
    div_n = zeros(T, nc)
    for f in 1:nf
        P = owner(mesh, f)
        S_f = face_normal_area(mesh, f)

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            # Interpolate n_hat to face
            n_f = w * n_hat[P] + (one(T) - w) * n_hat[N]
            flux = dot(n_f, S_f)

            div_n[P] += flux
            div_n[N] -= flux
        else
            # Boundary: use owner value (optionally rotated).
            n_face = n_hat[P]
            if contact_angle !== nothing &&
                    mesh.face_tags !== nothing &&
                    mesh.face_tags[f] in wall_patches
                A_f = mesh.face_areas[f]
                if A_f > T(0)
                    n_wall = S_f / A_f
                    n_face = apply_contact_angle(n_face, n_wall, contact_angle)
                end
            end
            flux = dot(n_face, S_f)
            div_n[P] += flux
        end
    end

    # Normalize by cell volume
    kappa = Vector{T}(undef, nc)
    for c in 1:nc
        div_n[c] /= mesh.cell_volumes[c]
        kappa[c] = -div_n[c]
    end

    return kappa
end

"""
    compute_surface_tension_force(
        alpha, props, mesh;
        contact_angle = nothing, wall_patches = Symbol[],
    ) -> Union{Nothing, Vector{SVector{Dim, T}}}

Compute the CSF surface tension body force: `F_st = σ · κ · ∇α`.

UNITS NOTE: the returned force is per unit VOLUME (dynamic, N/m³).
The VOF solver consumes it with a unit reference density (ρ_ref = 1,
`prob.density == 1` in `solve_vof`), under which dynamic force per
volume and kinematic force per unit mass coincide numerically.  If you
feed this force into a kinematic momentum equation with ρ_ref ≠ 1,
divide by the density first.

Returns `nothing` when `sigma == 0` (surface tension disabled).

When `contact_angle` is supplied, the curvature field is computed with
a wall-normal rotation applied to the CSF normal on the listed
`wall_patches` — giving the standard OpenFOAM `alphaContactAngle` wall
adhesion behaviour.
"""
function compute_surface_tension_force(
        alpha::CollocatedScalarField{T},
        props::TwoPhaseProperties{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        contact_angle::Union{Nothing, AbstractContactAngleModel} = nothing,
        wall_patches::Vector{Symbol} = Symbol[],
    ) where {Dim, T}
    if !has_surface_tension(props)
        return nothing
    end

    nc = length(mesh.cell_volumes)
    grad_alpha = gradient(alpha, mesh)
    kappa = compute_curvature(
        alpha, mesh;
        contact_angle = contact_angle, wall_patches = wall_patches,
    )

    force = Vector{SVector{Dim, T}}(undef, nc)
    for c in 1:nc
        force[c] = props.sigma * kappa[c] * grad_alpha[c]
    end

    return force
end

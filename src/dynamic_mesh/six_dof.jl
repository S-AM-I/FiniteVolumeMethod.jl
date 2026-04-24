# dynamic_mesh/six_dof.jl — Newton-Euler 6-DOF rigid body integrator
#
# Provides a mutable `RigidBody6DOF{T}` state plus the `advance_six_dof!`
# step function. The translational DOFs use a simple explicit Euler update
# (momentum form). The rotational DOFs are integrated in the body frame via
# the Euler equations I·ω̇ = τ − ω × (I·ω), with the orientation quaternion
# advanced by q̇ = 0.5 · ω ⊗ q and renormalized every step to prevent drift.
#
# All vectors are 3D. Inertia is stored as a 3×3 `SMatrix{3,3,T}` so
# diagonal / principal-axis inertias are supported without special-casing.
# The code deliberately re-implements quaternion helpers locally with the
# prefix `_sixdof_` to avoid clashes with quaternion helpers that may exist
# elsewhere in the package.

using LinearAlgebra: cross, dot, norm
using StaticArrays: SVector, SMatrix, @SVector, @SMatrix

@doc """
    RigidBody6DOF{T}

Mutable Newton-Euler rigid-body state.

# Fields
- `mass::T` — body mass
- `inertia::SMatrix{3, 3, T, 9}` — inertia tensor (body frame)
- `inertia_inv::SMatrix{3, 3, T, 9}` — cached inverse inertia tensor
- `position::SVector{3, T}` — centre-of-mass position (world frame)
- `velocity::SVector{3, T}` — linear velocity (world frame)
- `orientation::SVector{4, T}` — unit quaternion `(w, x, y, z)` mapping
  body → world
- `angular_velocity::SVector{3, T}` — angular velocity (body frame)

# Example
```julia
I3 = SMatrix{3,3}(1.0I)
body = RigidBody6DOF(1.0, I3)           # unit sphere at origin, at rest
advance_six_dof!(body, SVector(0.0, 0.0, -9.81), SVector(0.0, 0.0, 0.0), 0.01)
```
"""
mutable struct RigidBody6DOF{T}
    mass::T
    inertia::SMatrix{3, 3, T, 9}
    inertia_inv::SMatrix{3, 3, T, 9}
    position::SVector{3, T}
    velocity::SVector{3, T}
    orientation::SVector{4, T}
    angular_velocity::SVector{3, T}
end

@doc """
    RigidBody6DOF(mass, inertia; position = zero, velocity = zero,
                  orientation = identity_quat, angular_velocity = zero)

Construct a [`RigidBody6DOF`](@ref) with defaults placing the body at the
origin, at rest, aligned with the world axes. The inertia tensor is inverted
once at construction time and cached as `inertia_inv`.
"""
function RigidBody6DOF(
        mass::T,
        inertia::AbstractMatrix;
        position::AbstractVector = @SVector(zeros(T, 3)),
        velocity::AbstractVector = @SVector(zeros(T, 3)),
        orientation::AbstractVector = SVector{4, T}(one(T), zero(T), zero(T), zero(T)),
        angular_velocity::AbstractVector = @SVector(zeros(T, 3)),
    ) where {T}
    size(inertia) == (3, 3) || error("inertia tensor must be 3×3, got $(size(inertia))")
    mass > zero(T) || error("mass must be positive, got $mass")
    I3 = SMatrix{3, 3, T, 9}(inertia)
    I3_inv = inv(I3)
    return RigidBody6DOF{T}(
        mass, I3, I3_inv,
        SVector{3, T}(position),
        SVector{3, T}(velocity),
        _sixdof_normalize_quat(SVector{4, T}(orientation)),
        SVector{3, T}(angular_velocity),
    )
end

# ── Quaternion helpers (local, prefixed `_sixdof_` to avoid clashes) ──

"""Hamilton product of two quaternions q1 ⊗ q2 with (w, x, y, z) storage."""
function _sixdof_quat_mul(q1::SVector{4, T}, q2::SVector{4, T}) where {T}
    w1, x1, y1, z1 = q1[1], q1[2], q1[3], q1[4]
    w2, x2, y2, z2 = q2[1], q2[2], q2[3], q2[4]
    return SVector{4, T}(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
end

"""Normalize a quaternion, falling back to identity for a zero quaternion."""
function _sixdof_normalize_quat(q::SVector{4, T}) where {T}
    n = sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3] + q[4] * q[4])
    if n < eps(T)
        return SVector{4, T}(one(T), zero(T), zero(T), zero(T))
    end
    return q / n
end

"""Embed a 3-vector ω as a pure-imaginary quaternion (0, ωx, ωy, ωz)."""
function _sixdof_omega_as_quat(omega::SVector{3, T}) where {T}
    return SVector{4, T}(zero(T), omega[1], omega[2], omega[3])
end

@doc """
    advance_six_dof!(body::RigidBody6DOF{T}, force::SVector{3,T},
                     torque::SVector{3,T}, dt::T) where {T}

Advance `body` by one explicit Euler step under the applied world-frame
`force` and body-frame `torque`. The updates are

```
r_{n+1} = r_n + v_n · dt
v_{n+1} = v_n + (F_n / m) · dt
q̇        = 0.5 · ω_n ⊗ q_n
q_{n+1} = normalize(q_n + q̇ · dt)
ω̇        = I⁻¹ · (τ_n − ω_n × (I · ω_n))
ω_{n+1} = ω_n + ω̇ · dt
```

Returns `body` (mutated in place). Accepts any `AbstractVector` for
`force` and `torque`; they are converted to `SVector{3, T}` internally
where `T` is the body's element type.
"""
function advance_six_dof!(
        body::RigidBody6DOF{T},
        force::AbstractVector,
        torque::AbstractVector,
        dt::Real,
    ) where {T}
    F = SVector{3, T}(force)
    τ = SVector{3, T}(torque)
    Δt = T(dt)

    # Translation (world frame)
    body.position = body.position + body.velocity * Δt
    body.velocity = body.velocity + (F / body.mass) * Δt

    # Rotation — orientation update via quaternion kinematics
    omega_q = _sixdof_omega_as_quat(body.angular_velocity)
    qdot = T(0.5) * _sixdof_quat_mul(omega_q, body.orientation)
    body.orientation = _sixdof_normalize_quat(body.orientation + qdot * Δt)

    # Rotation — angular velocity via Euler's equation (body frame)
    I_omega = body.inertia * body.angular_velocity
    omega_dot = body.inertia_inv * (τ - cross(body.angular_velocity, I_omega))
    body.angular_velocity = body.angular_velocity + omega_dot * Δt

    return body
end

@doc """
    angular_momentum(body::RigidBody6DOF{T}) -> SVector{3, T}

Return the body-frame angular momentum `L = I · ω`.
"""
function angular_momentum(body::RigidBody6DOF{T}) where {T}
    return body.inertia * body.angular_velocity
end

@doc """
    kinetic_energy(body::RigidBody6DOF{T}) -> T

Return the translational + rotational kinetic energy
`½ m ‖v‖² + ½ ωᵀ I ω`.
"""
function kinetic_energy(body::RigidBody6DOF{T}) where {T}
    v = body.velocity
    omega = body.angular_velocity
    E_trans = T(0.5) * body.mass * dot(v, v)
    E_rot = T(0.5) * dot(omega, body.inertia * omega)
    return E_trans + E_rot
end

@doc """
    quaternion_norm(body::RigidBody6DOF) -> T

Return the 2-norm of the orientation quaternion. A well-maintained body
should have `quaternion_norm ≈ 1` after every step.
"""
function quaternion_norm(body::RigidBody6DOF{T}) where {T}
    q = body.orientation
    return sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3] + q[4] * q[4])
end

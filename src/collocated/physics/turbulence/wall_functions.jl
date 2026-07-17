# turbulence/wall_functions.jl — Wall function BC generation for turbulence models
#
# Generates boundary condition dictionaries for turbulence fields at wall
# and inlet patches. Reuses existing compute_friction_velocity, k_wall_value,
# epsilon_wall_value from src/physics/turbulence/k_epsilon.jl.

"""
    turbulence_inlet_bc(model::StandardKEpsilon, U_mag, intensity, length_scale)

Generate inlet BCs for k-ε from freestream conditions.

- `k_inlet = 1.5 * (U_mag * intensity)²`
- `ε_inlet = C_μ^0.75 * k^1.5 / length_scale`
"""
function turbulence_inlet_bc(
        model::StandardKEpsilon, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    k_inlet = T(1.5) * (U_mag * intensity)^2
    eps_inlet = model.C_mu^T(0.75) * k_inlet^T(1.5) / length_scale
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => DirichletBC(k_inlet),
        :epsilon => DirichletBC(eps_inlet),
    )
end

"""
    turbulence_inlet_bc(model::KOmega, U_mag, intensity, length_scale)

Generate inlet BCs for k-ω from freestream conditions.

- `k_inlet = 1.5 * (U_mag * intensity)²`
- `ω_inlet = k^0.5 / (C_μ^0.25 * length_scale)`
"""
function turbulence_inlet_bc(
        model::KOmega, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    k_inlet = T(1.5) * (U_mag * intensity)^2
    omega_inlet = sqrt(k_inlet) / (T(0.09)^T(0.25) * length_scale)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => DirichletBC(k_inlet),
        :omega => DirichletBC(omega_inlet),
    )
end

"""
    turbulence_inlet_bc(model::KOmegaSSTModel, U_mag, intensity, length_scale)

Generate inlet BCs for k-ω SST (same as k-ω).
"""
function turbulence_inlet_bc(
        model::KOmegaSSTModel, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    k_inlet = T(1.5) * (U_mag * intensity)^2
    omega_inlet = sqrt(k_inlet) / (model.coeffs.beta_star^T(0.25) * length_scale)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => DirichletBC(k_inlet),
        :omega => DirichletBC(omega_inlet),
    )
end

"""
    turbulence_inlet_bc(model::SpalartAllmaras, U_mag, intensity, length_scale)

Generate inlet BCs for SA. ν̃_inlet ≈ 3-5 * ν for freestream.
"""
function turbulence_inlet_bc(
        ::SpalartAllmaras, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    # SA freestream: nu_tilde ≈ 3 * nu_laminar is typical
    # Use intensity to scale: higher TI → higher nu_tilde
    nu_tilde_inlet = T(3) * intensity * U_mag * length_scale
    return Dict{Symbol, AbstractBoundaryCondition}(
        :nu_tilde => DirichletBC(nu_tilde_inlet),
    )
end

"""
    turbulence_wall_bc(model::StandardKEpsilon)

Generate wall BCs for k-ε (zero-gradient for k, fixed ε via wall function).
For now returns Neumann(0) for both — the wall function values are
set dynamically during the solve.
"""
function turbulence_wall_bc(::StandardKEpsilon)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => NeumannBC(0.0),
        :epsilon => NeumannBC(0.0),
    )
end

function turbulence_wall_bc(::Union{KOmega, KOmegaSSTModel})
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => NeumannBC(0.0),
        :omega => NeumannBC(0.0),
    )
end

function turbulence_wall_bc(::SpalartAllmaras)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :nu_tilde => DirichletBC(0.0),
    )
end

# ── Wall function computation ─────────────────────────────────────

# Log-law constants
const WF_KAPPA = 0.41       # von Karman constant
const WF_E = 9.793      # log-law constant
const WF_C_MU = 0.09       # k-epsilon model constant

"""
    spalding_u_tau(U_par, y, nu; max_iter = 20, tol = 1e-6) -> T

Solve for friction velocity `u_tau` using the Spalding law of the wall:
```
    y+ = u+ + (1/E)(e^(κ u+) - 1 - κu+ - (κu+)²/2 - (κu+)³/6)
```
Uses Newton iteration starting from the log-law estimate.
"""
function spalding_u_tau(
        U_par::T, y::T, nu::T;
        max_iter::Int = 20,
        tol::T = T(1.0e-6),
    ) where {T}
    kappa = T(WF_KAPPA)
    E_wf = T(WF_E)

    # Initial estimate from log law: u_tau = kappa * U_par / log(E * y * U_par / nu + 1)
    y_star = max(y * U_par / nu, T(1))
    u_plus = log(E_wf * y_star) / kappa
    u_plus = max(u_plus, T(1))
    u_tau = U_par / u_plus

    # Newton iteration on Spalding equation:
    # f(u+) = u+ + (1/E)(exp(κ*u+) - 1 - κ*u+ - (κ*u+)^2/2 - (κ*u+)^3/6) - y+
    for _ in 1:max_iter
        u_tau = max(u_tau, T(1.0e-14))
        u_plus = U_par / u_tau
        y_plus = y * u_tau / nu

        ku = kappa * u_plus
        exp_ku = exp(min(ku, T(50)))  # cap to avoid overflow
        f = u_plus + (exp_ku - one(T) - ku - ku^2 / 2 - ku^3 / 6) / E_wf - y_plus
        # df/du_tau = df/du+ * du+/du_tau + df/dy+ * dy+/du_tau
        df_dup = one(T) + (
            kappa * exp_ku - kappa - kappa^2 * u_plus -
                kappa^3 * u_plus^2 / 2
        ) / E_wf
        dup_dut = -U_par / u_tau^2
        dyp_dut = y / nu
        df_dut = df_dup * dup_dut - dyp_dut

        abs(df_dut) < T(1.0e-20) && break
        u_tau_new = u_tau - f / df_dut
        u_tau_new = max(u_tau_new, T(1.0e-14))

        if abs(u_tau_new - u_tau) < tol * u_tau
            u_tau = u_tau_new
            break
        end
        u_tau = u_tau_new
    end

    return u_tau
end

"""
    compute_nut_wall(U_par, y, nu) -> T

Compute turbulent viscosity `nu_t` at a wall-adjacent cell using the
Spalding wall function (equivalent to OpenFOAM `nutUSpaldingWallFunction`).

```
    nu_t = nu * (y+ / u+ - 1)
```
where `y+` and `u+` are from the converged Spalding iteration.
"""
function compute_nut_wall(U_par::T, y::T, nu::T) where {T}
    u_tau = spalding_u_tau(U_par, y, nu)
    y_plus = y * u_tau / nu
    u_plus = max(U_par / u_tau, T(1.0e-10))
    nut = nu * max(y_plus / u_plus - one(T), zero(T))
    return nut
end

"""
    equilibrium_k_wall(u_tau) -> T

Equilibrium TKE at wall-adjacent cell: `k = u_tau² / sqrt(C_mu)`.
"""
equilibrium_k_wall(u_tau::T) where {T} = u_tau^2 / sqrt(T(WF_C_MU))

"""
    equilibrium_epsilon_wall(u_tau, y, nu) -> T

Equilibrium dissipation at wall-adjacent cell: `ε = u_tau³ / (κ y)`.
"""
function equilibrium_epsilon_wall(u_tau::T, y::T, nu::T) where {T}
    return u_tau^3 / (T(WF_KAPPA) * max(y, T(1.0e-20)))
end

"""
    equilibrium_omega_wall(u_tau, y, nu) -> T

Equilibrium specific dissipation: `ω = u_tau / (C_mu^0.25 * κ * y)`.
"""
function equilibrium_omega_wall(u_tau::T, y::T, nu::T) where {T}
    return u_tau / (T(WF_C_MU)^T(0.25) * T(WF_KAPPA) * max(y, T(1.0e-20)))
end

# ── Apply wall functions to turbulence state ──────────────────────

"""
    apply_wall_functions!(
        turb_state, model, U, mesh, nu, wall_patches,
    )

After solving the turbulence transport equations, update near-wall
cell values using equilibrium wall functions:

- Compute `u_tau` at each wall face via the Spalding law
- Set `k` and `ε` (or `ω`) at wall-adjacent cells to equilibrium values
- Update `nu_t` at wall-adjacent cells

This enforces the log-law boundary condition implicitly without
requiring explicit Dirichlet BCs on k and ε.
"""
# Stage 4d: project the cell-center-to-wall offset and the cell-center
# velocity onto the wall-normal / wall-tangential axes of face `f`.
#
# Returns `(y, U_par)` where:
# - `y = |(x_c - x_f) · n̂|` is the wall-normal distance from cell `c`
#   to boundary face `f`. On a Cartesian mesh this is identical to the
#   straight-line distance; on skewed cells it is strictly smaller and
#   is the physically-correct wall-normal coordinate for the log law.
# - `U_par = |U_cell - (U_cell · n̂) n̂|` is the wall-tangential
#   velocity magnitude. On a Cartesian no-slip wall with flow parallel
#   to the wall, this equals `|U_cell|` (the old formula); on skewed
#   cells or cells with residual wall-normal velocity during an
#   iterative solve, it removes the spurious normal-component
#   contribution.
#
# Both values are robust to the outward-vs-inward normal convention.
@inline function _wall_projection(
        mesh::UnstructuredFVMMesh{Dim, T}, c::Int, f::Int,
        U_cell::SVector{Dim, T},
    ) where {Dim, T}
    # Unit normal to face `f`. face_normal_area stores A·n̂.
    A_f = mesh.face_areas[f]
    S_f = face_normal_area(mesh, f)
    n_hat = S_f / A_f

    x_c = cell_center(mesh, c)
    x_f = face_center(mesh, f)
    d = x_c - x_f
    y = abs(dot(d, n_hat))

    U_normal = dot(U_cell, n_hat) * n_hat
    U_par_vec = U_cell - U_normal
    U_par = norm(U_par_vec)

    return y, U_par
end

function apply_wall_functions!(
        turb_state::RANSTurbulenceState{T},
        model::StandardKEpsilon,
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        nu::T,
        wall_patches::Vector{Symbol},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    wall_set = Set(wall_patches)

    k_field = turb_state.fields[:k]
    eps_field = turb_state.fields[:epsilon]

    for f in 1:nf
        is_internal_face(mesh, f) && continue
        tag = _face_tag(mesh, f)
        tag in wall_set || continue

        c = owner(mesh, f)
        y, U_par = _wall_projection(mesh, c, f, U.internal[c])

        u_tau = spalding_u_tau(U_par, y, nu)
        k_field.internal[c] = equilibrium_k_wall(u_tau)
        eps_field.internal[c] = equilibrium_epsilon_wall(u_tau, y, nu)
        turb_state.nu_t[c] = compute_nut_wall(U_par, y, nu)
    end

    return nothing
end

function apply_wall_functions!(
        turb_state::RANSTurbulenceState{T},
        model::Union{KOmega, KOmegaSSTModel},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        nu::T,
        wall_patches::Vector{Symbol},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    wall_set = Set(wall_patches)

    k_field = turb_state.fields[:k]
    omega_field = turb_state.fields[:omega]

    for f in 1:nf
        is_internal_face(mesh, f) && continue
        tag = _face_tag(mesh, f)
        tag in wall_set || continue

        c = owner(mesh, f)
        y, U_par = _wall_projection(mesh, c, f, U.internal[c])

        u_tau = spalding_u_tau(U_par, y, nu)
        k_field.internal[c] = equilibrium_k_wall(u_tau)
        omega_field.internal[c] = equilibrium_omega_wall(u_tau, y, nu)
        turb_state.nu_t[c] = compute_nut_wall(U_par, y, nu)
    end

    return nothing
end

function apply_wall_functions!(
        turb_state::RANSTurbulenceState{T},
        model::SpalartAllmaras,
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        nu::T,
        wall_patches::Vector{Symbol},
    ) where {Dim, T}
    # SA: nu_tilde = 0 at wall (already set by Dirichlet BC)
    # Just update nu_t at wall-adjacent cells — same skew-penalty
    # projection as the k-ε / k-ω branches so SA doesn't see a
    # spuriously-large y from the raw centre-to-centre offset on
    # skewed cells.
    nf = size(mesh.face_cells, 2)
    wall_set = Set(wall_patches)

    for f in 1:nf
        is_internal_face(mesh, f) && continue
        tag = _face_tag(mesh, f)
        tag in wall_set || continue

        c = owner(mesh, f)
        y, U_par = _wall_projection(mesh, c, f, U.internal[c])
        turb_state.nu_t[c] = compute_nut_wall(U_par, y, nu)
    end

    return nothing
end

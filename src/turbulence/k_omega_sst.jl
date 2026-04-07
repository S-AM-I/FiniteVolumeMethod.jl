# turbulence/k_omega_sst.jl — k-ω SST turbulence model (Menter 1994)
#
# Blends k-ω (near wall) with k-ε (far field) via blending functions
# F1 and F2. Includes the SST viscosity limiter for adverse pressure
# gradients. Wraps the existing KappaOmegaSST coefficients struct.

"""
    KOmegaSSTModel{T} <: AbstractRANSModel

k-ω SST (Shear Stress Transport) turbulence model.

Wraps `KappaOmegaSST{T}` coefficients and adds the blending/limiter
logic. Requires wall distance `d_wall` per cell.

# Fields
- `coeffs::KappaOmegaSST{T}` — model coefficients
- `d_wall::Vector{T}` — wall distance per cell (precomputed)
- `nu::T` — laminar kinematic viscosity (for F2 computation)
"""
struct KOmegaSSTModel{T} <: AbstractRANSModel
    coeffs::KappaOmegaSST{T}
    d_wall::Vector{T}
    nu::T
end

function KOmegaSSTModel(
        mesh::UnstructuredFVMMesh{Dim, T}, wall_patches::Vector{Symbol};
        coeffs = KappaOmegaSST(),
        nu::Real = 1.0e-5,
    ) where {Dim, T}
    d_wall = compute_wall_distance(mesh, wall_patches)
    return KOmegaSSTModel{T}(coeffs, T.(d_wall), T(nu))
end

n_turbulence_fields(::KOmegaSSTModel) = 2
turbulence_field_names(::KOmegaSSTModel) = (:k, :omega)

# ── Blending functions ───────────────────────────────────────────────

"""Compute F1 blending function (0 = k-ε far field, 1 = k-ω near wall)."""
function _sst_F1(
        k::T, omega::T, nu::T, d::T, coeffs::KappaOmegaSST{T},
        grad_k_dot_grad_omega::T
    ) where {T}
    d_safe = max(d, T(1.0e-10))
    omega_safe = max(omega, T(1.0e-10))
    k_safe = max(k, T(1.0e-10))

    arg1_a = sqrt(k_safe) / (coeffs.beta_star * omega_safe * d_safe)
    arg1_b = T(500) * nu / (d_safe^2 * omega_safe)
    arg1_ab = max(arg1_a, arg1_b)

    CDkw = max(T(2) * coeffs.sigma_omega2 / omega_safe * grad_k_dot_grad_omega, T(1.0e-10))
    arg1_c = T(4) * coeffs.sigma_omega2 * k_safe / (CDkw * d_safe^2)

    arg1 = min(arg1_ab, arg1_c)
    return tanh(arg1^4)
end

"""Compute F2 blending function for the SST viscosity limiter."""
function _sst_F2(k::T, omega::T, nu::T, d::T, coeffs::KappaOmegaSST{T}) where {T}
    d_safe = max(d, T(1.0e-10))
    omega_safe = max(omega, T(1.0e-10))
    k_safe = max(k, T(1.0e-10))

    arg2_a = T(2) * sqrt(k_safe) / (coeffs.beta_star * omega_safe * d_safe)
    arg2_b = T(500) * nu / (d_safe^2 * omega_safe)
    arg2 = max(arg2_a, arg2_b)
    return tanh(arg2^2)
end

"""Blend a constant: phi = F1*phi_1 + (1-F1)*phi_2."""
_blend(phi1::T, phi2::T, F1::T) where {T} = F1 * phi1 + (one(T) - F1) * phi2

# ── Interface implementation ─────────────────────────────────────────

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::KOmegaSSTModel{T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    k_field = turb_state.fields[:k]
    omega_field = turb_state.fields[:omega]
    co = model.coeffs
    nc = length(mesh.cell_volumes)

    # SST viscosity limiter: nu_t = a1*k / max(a1*omega, S*F2)
    # Estimate strain rate from existing nu_t: S ≈ nu_t * omega / k
    # (from the equilibrium relation nu_t = k/omega and production = dissipation)
    for c in 1:nc
        k_val = max(k_field.internal[c], T(1.0e-10))
        omega_val = max(omega_field.internal[c], T(1.0e-10))

        # F2 with correct laminar viscosity
        F2 = _sst_F2(k_val, omega_val, model.nu, model.d_wall[c], co)

        # Estimate S from current nu_t: in equilibrium, nu_t*S^2 ≈ beta_star*k*omega
        # so S ≈ sqrt(beta_star * k * omega / max(nu_t, eps))
        nu_t_old = max(nu_t[c], T(1.0e-10))
        S_est = sqrt(co.beta_star * k_val * omega_val / nu_t_old)

        # SST limiter
        nu_t[c] = co.a1 * k_val / max(co.a1 * omega_val, S_est * F2)
    end
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::KOmegaSSTModel{T},
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
    co = model.coeffs
    k_field = turb_state.fields[:k]
    omega_field = turb_state.fields[:omega]

    # Production
    S_mag = compute_strain_rate(U, mesh)
    P_k = Vector{T}(undef, nc)
    for c in 1:nc
        P_k[c] = turb_state.nu_t[c] * S_mag[c]^2
    end

    # Compute grad(k) and grad(omega) for cross-diffusion and F1
    grad_k = gradient(k_field, mesh)
    grad_omega = gradient(omega_field, mesh)

    # Compute F1 per cell
    F1 = Vector{T}(undef, nc)
    for c in 1:nc
        gk_dot_gw = dot(grad_k[c], grad_omega[c])
        F1[c] = _sst_F1(
            k_field.internal[c], omega_field.internal[c],
            nu, model.d_wall[c], co, gk_dot_gw
        )
    end

    # Blended constants
    sigma_k_blend = Vector{T}(undef, nc)
    sigma_omega_blend = Vector{T}(undef, nc)
    beta_blend = Vector{T}(undef, nc)
    alpha_blend = Vector{T}(undef, nc)
    for c in 1:nc
        sigma_k_blend[c] = _blend(co.sigma_k1, co.sigma_k2, F1[c])
        sigma_omega_blend[c] = _blend(co.sigma_omega1, co.sigma_omega2, F1[c])
        beta_blend[c] = _blend(co.beta1, co.beta2, F1[c])
        # alpha from beta, beta_star, sigma_omega, kappa
        alpha1 = co.beta1 / co.beta_star - co.sigma_omega1 * co.kappa^2 / sqrt(co.beta_star)
        alpha2 = co.beta2 / co.beta_star - co.sigma_omega2 * co.kappa^2 / sqrt(co.beta_star)
        alpha_blend[c] = _blend(alpha1, alpha2, F1[c])
    end

    # ── k equation ───────────────────────────────────────────────
    k_eq = CollocatedEquation(mesh)
    bcs_k = get(bcs_turb, :k, Dict{Symbol, AbstractBoundaryCondition}())

    assemble_convection!(k_eq, phi, mesh, bcs_k)

    gamma_k = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_k[c] = nu + sigma_k_blend[c] * turb_state.nu_t[c]
    end
    assemble_laplacian!(k_eq, gamma_k, mesh, bcs_k)

    if dt !== nothing
        assemble_ddt_euler!(k_eq, one(T), k_field.internal, mesh, dt)
    end

    for c in 1:nc
        omega_val = max(omega_field.internal[c], T(1.0e-10))
        k_eq.b[c] += P_k[c] * mesh.cell_volumes[c]
        k_eq.A[c, c] += co.beta_star * omega_val * mesh.cell_volumes[c]
    end

    lp_k = to_linear_problem(k_eq)
    sol_k = _dispatch_solve(lp_k, linear_solver, solver_config, :k)
    for c in 1:nc
        k_field.internal[c] = max(sol_k.u[c], T(1.0e-10))
    end

    # ── ω equation ───────────────────────────────────────────────
    omega_eq = CollocatedEquation(mesh)
    bcs_omega = get(bcs_turb, :omega, Dict{Symbol, AbstractBoundaryCondition}())

    assemble_convection!(omega_eq, phi, mesh, bcs_omega)

    gamma_omega = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_omega[c] = nu + sigma_omega_blend[c] * turb_state.nu_t[c]
    end
    assemble_laplacian!(omega_eq, gamma_omega, mesh, bcs_omega)

    if dt !== nothing
        assemble_ddt_euler!(omega_eq, one(T), omega_field.internal, mesh, dt)
    end

    for c in 1:nc
        k_safe = max(k_field.internal[c], T(1.0e-10))
        omega_val = max(omega_field.internal[c], T(1.0e-10))
        # Production
        omega_eq.b[c] += alpha_blend[c] * (omega_val / k_safe) * P_k[c] * mesh.cell_volumes[c]
        # Destruction
        omega_eq.A[c, c] += beta_blend[c] * omega_val * mesh.cell_volumes[c]
        # Cross-diffusion (explicit, only in k-ε region where F1 < 1)
        gk_dot_gw = dot(grad_k[c], grad_omega[c])
        cd_term = T(2) * (one(T) - F1[c]) * co.sigma_omega2 / omega_val * gk_dot_gw
        omega_eq.b[c] += max(cd_term, zero(T)) * mesh.cell_volumes[c]
    end

    lp_omega = to_linear_problem(omega_eq)
    sol_omega = _dispatch_solve(lp_omega, linear_solver, solver_config, :omega)
    for c in 1:nc
        omega_field.internal[c] = max(sol_omega.u[c], T(1.0e-10))
    end

    return nothing
end

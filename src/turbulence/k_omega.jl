# turbulence/k_omega.jl — Standard k-ω turbulence model (Wilcox 1988)
#
# Two-equation model with better near-wall behavior than k-ε.
# Turbulent viscosity: ν_t = k / ω

"""
    KOmega{T} <: AbstractRANSModel

Standard k-ω turbulence model (Wilcox 1988).

# Fields
- `beta_star::T` — k destruction coefficient (default 0.09)
- `alpha::T` — ω production coefficient (default 5/9)
- `beta::T` — ω destruction coefficient (default 3/40)
- `sigma_k::T` — k diffusion Prandtl number (default 0.5)
- `sigma_omega::T` — ω diffusion Prandtl number (default 0.5)
"""
struct KOmega{T} <: AbstractRANSModel
    beta_star::T
    alpha::T
    beta::T
    sigma_k::T
    sigma_omega::T
end

function KOmega(;
        beta_star = 0.09, alpha = 5.0 / 9.0, beta = 3.0 / 40.0,
        sigma_k = 0.5, sigma_omega = 0.5,
    )
    T = promote_type(
        typeof(beta_star), typeof(alpha), typeof(beta),
        typeof(sigma_k), typeof(sigma_omega),
    )
    return KOmega{T}(T(beta_star), T(alpha), T(beta), T(sigma_k), T(sigma_omega))
end

n_turbulence_fields(::KOmega) = 2
turbulence_field_names(::KOmega) = (:k, :omega)

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::KOmega,
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    k_field = turb_state.fields[:k]
    omega_field = turb_state.fields[:omega]
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        k_val = max(k_field.internal[c], T(1.0e-10))
        omega_val = max(omega_field.internal[c], T(1.0e-10))
        nu_t[c] = k_val / omega_val
    end
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::KOmega,
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    k_field = turb_state.fields[:k]
    omega_field = turb_state.fields[:omega]

    # Production
    S_mag = compute_strain_rate(U, mesh)
    P_k = Vector{T}(undef, nc)
    for c in 1:nc
        P_k[c] = turb_state.nu_t[c] * S_mag[c]^2
    end

    # ── k equation ───────────────────────────────────────────────
    k_eq = CollocatedEquation(mesh)
    bcs_k = get(bcs_turb, :k, Dict{Symbol, AbstractBoundaryCondition}())

    assemble_convection!(k_eq, phi, mesh, bcs_k)

    gamma_k = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_k[c] = nu + model.sigma_k * turb_state.nu_t[c]
    end
    assemble_laplacian!(k_eq, gamma_k, mesh, bcs_k)

    if dt !== nothing
        assemble_ddt_euler!(k_eq, one(T), k_field.internal, mesh, dt)
    end

    # Source: S_C = P_k, S_P = -beta_star * omega
    for c in 1:nc
        omega_val = max(omega_field.internal[c], T(1.0e-10))
        k_eq.b[c] += P_k[c] * mesh.cell_volumes[c]
        k_eq.A[c, c] += model.beta_star * omega_val * mesh.cell_volumes[c]
    end

    lp_k = to_linear_problem(k_eq)
    sol_k = _solve_linear(lp_k, linear_solver)
    for c in 1:nc
        k_field.internal[c] = max(sol_k.u[c], T(1.0e-10))
    end

    # ── ω equation ───────────────────────────────────────────────
    omega_eq = CollocatedEquation(mesh)
    bcs_omega = get(bcs_turb, :omega, Dict{Symbol, AbstractBoundaryCondition}())

    assemble_convection!(omega_eq, phi, mesh, bcs_omega)

    gamma_omega = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_omega[c] = nu + model.sigma_omega * turb_state.nu_t[c]
    end
    assemble_laplacian!(omega_eq, gamma_omega, mesh, bcs_omega)

    if dt !== nothing
        assemble_ddt_euler!(omega_eq, one(T), omega_field.internal, mesh, dt)
    end

    # Source: S_C = alpha*(omega/k)*P_k, S_P = -beta*omega
    for c in 1:nc
        k_safe = max(k_field.internal[c], T(1.0e-10))
        omega_val = max(omega_field.internal[c], T(1.0e-10))
        omega_eq.b[c] += model.alpha * (omega_val / k_safe) * P_k[c] * mesh.cell_volumes[c]
        omega_eq.A[c, c] += model.beta * omega_val * mesh.cell_volumes[c]
    end

    lp_omega = to_linear_problem(omega_eq)
    sol_omega = _solve_linear(lp_omega, linear_solver)
    for c in 1:nc
        omega_field.internal[c] = max(sol_omega.u[c], T(1.0e-10))
    end

    return nothing
end

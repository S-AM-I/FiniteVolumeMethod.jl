# turbulence/k_epsilon_rans.jl — Standard k-ε model for collocated solver
#
# Provides turbulent_viscosity! and solve_turbulence! methods for the
# existing StandardKEpsilon type. Assembles k and ε transport equations
# using Phase 0 operators (convection, Laplacian, source linearization).

# ── Interface implementation ─────────────────────────────────────────

n_turbulence_fields(::StandardKEpsilon) = 2
turbulence_field_names(::StandardKEpsilon) = (:k, :epsilon)

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::StandardKEpsilon,
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    k_field = turb_state.fields[:k]
    eps_field = turb_state.fields[:epsilon]
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        k_val = max(k_field.internal[c], T(1.0e-10))
        eps_val = max(eps_field.internal[c], T(1.0e-10))
        nu_t[c] = model.C_mu * k_val^2 / eps_val
    end
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::StandardKEpsilon,
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
    k_field = turb_state.fields[:k]
    eps_field = turb_state.fields[:epsilon]

    # Compute production. Strain-rate magnitude is |S| = √(2 S_ij S_ij)
    # — a full tensor contraction; production is ν_t · |S|².
    S_mag = compute_strain_rate(U, mesh)

    # Stage 4a: Durbin realizability cap. When `model.realizability_alpha > 0`
    # we enforce ν_t ≤ α · k / |S| before computing production. This keeps
    # the eddy viscosity bounded in regions of strong strain and suppresses
    # the non-physical ν_t spikes that break convergence on flows like the
    # Sandia-flame near-nozzle region or backward-facing step just past
    # reattachment.
    if model.realizability_alpha > zero(T)
        k_field_internal = k_field.internal
        for c in 1:nc
            k_val = max(k_field_internal[c], T(1.0e-10))
            s_val = max(S_mag[c], T(1.0e-10))
            nu_t_cap = model.realizability_alpha * k_val / s_val
            turb_state.nu_t[c] = min(turb_state.nu_t[c], nu_t_cap)
        end
    end

    P_k = Vector{T}(undef, nc)
    for c in 1:nc
        P_k[c] = turb_state.nu_t[c] * S_mag[c]^2
    end

    # ── k equation ───────────────────────────────────────────────
    k_eq = CollocatedEquation(mesh)
    bcs_k = get(bcs_turb, :k, Dict{Symbol, AbstractBoundaryCondition}())

    # Convection
    assemble_convection!(k_eq, phi, mesh, bcs_k)

    # Diffusion: gamma_k = nu + nu_t / sigma_k
    gamma_k = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_k[c] = nu + turb_state.nu_t[c] / model.sigma_k
    end
    assemble_laplacian!(k_eq, gamma_k, mesh, bcs_k)

    # Temporal term
    if dt !== nothing
        assemble_ddt_euler!(k_eq, one(T), k_field.internal, mesh, dt)
    end

    # Source: S_C = P_k, S_P = -eps/k (linearized destruction)
    for c in 1:nc
        k_safe = max(k_field.internal[c], T(1.0e-10))
        k_eq.b[c] += P_k[c] * mesh.cell_volumes[c]
        add_diag!(k_eq, c, eps_field.internal[c] / k_safe * mesh.cell_volumes[c])
    end

    # Solve k
    lp_k = to_linear_problem(k_eq)
    sol_k = _dispatch_solve(lp_k, linear_solver, solver_config, :k)
    for c in 1:nc
        k_field.internal[c] = max(sol_k.u[c], T(1.0e-10))
    end

    # ── ε equation ───────────────────────────────────────────────
    eps_eq = CollocatedEquation(mesh)
    bcs_eps = get(bcs_turb, :epsilon, Dict{Symbol, AbstractBoundaryCondition}())

    # Convection
    assemble_convection!(eps_eq, phi, mesh, bcs_eps)

    # Diffusion: gamma_eps = nu + nu_t / sigma_epsilon
    gamma_eps = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_eps[c] = nu + turb_state.nu_t[c] / model.sigma_epsilon
    end
    assemble_laplacian!(eps_eq, gamma_eps, mesh, bcs_eps)

    # Temporal term
    if dt !== nothing
        assemble_ddt_euler!(eps_eq, one(T), eps_field.internal, mesh, dt)
    end

    # Source: S_C = C1*(eps/k)*P_k, S_P = -C2*(eps/k) (linearized)
    for c in 1:nc
        k_safe = max(k_field.internal[c], T(1.0e-10))
        eps_by_k = eps_field.internal[c] / k_safe
        eps_eq.b[c] += model.C1_epsilon * eps_by_k * P_k[c] * mesh.cell_volumes[c]
        add_diag!(eps_eq, c, model.C2_epsilon * eps_by_k * mesh.cell_volumes[c])
    end

    # Solve ε
    lp_eps = to_linear_problem(eps_eq)
    sol_eps = _dispatch_solve(lp_eps, linear_solver, solver_config, :epsilon)
    for c in 1:nc
        eps_field.internal[c] = max(sol_eps.u[c], T(1.0e-10))
    end

    return nothing
end

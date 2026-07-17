# turbulence/k_epsilon_rans.jl — Standard k-ε model for collocated solver
#
# Provides turbulent_viscosity! and solve_turbulence! methods for the
# existing StandardKEpsilon type. Assembles k and ε transport equations
# using Phase 0 operators (convection, Laplacian, source linearization).
#
# Correctness upgrades (v3.0 Wave 1 — production-grade):
#   1. Durbin realizability cap:  ν_t ← min(C_μ k²/ε, C_T k / |S|)
#      with C_T ≈ 0.6 (Durbin 1996). Bounds the Reynolds stresses by
#      the Schwarz inequality and suppresses non-physical spikes in
#      strong-shear regions. The cap is driven by `model.realizability_alpha`
#      — if the user leaves it at zero, `_durbin_C_T(model)` falls back to
#      the Durbin 1996 default 0.6 so the cap is active out-of-the-box.
#   2. Full-tensor production: P_k = 2 ν_t S_ij S_ij, assembled through
#      `_sym_self_magnitude_sq` from `dynamic_smagorinsky.jl`. Algebraically
#      identical to `ν_t · |S|²` under the  `|S| = √(2 S_ij S_ij)` convention
#      (`strain_rate.jl`) but written in the tensor form so the origin of
#      each term is explicit — and so swapping in a non-isotropic ν_t later
#      is a one-line change rather than an algebraic audit.

# ── Interface implementation ─────────────────────────────────────────

n_turbulence_fields(::StandardKEpsilon) = 2
turbulence_field_names(::StandardKEpsilon) = (:k, :epsilon)

"""
    _durbin_C_T(model) -> T

Durbin realizability coefficient for the eddy-viscosity cap
`ν_t ≤ C_T · k / |S|`. Honours `model.realizability_alpha` when set
(>0) and otherwise returns the Durbin 1996 default 0.6.
"""
@inline function _durbin_C_T(model::StandardKEpsilon{T}) where {T}
    return model.realizability_alpha > zero(T) ?
        model.realizability_alpha : T(0.6)
end

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

"""
    _apply_durbin_cap!(nu_t, k_field, S_mag, C_T)

Enforce `ν_t ≤ C_T · k / |S|` pointwise. Writes the capped eddy
viscosity back into `nu_t`. Safe against `|S| → 0` (no cap applied
when the local strain vanishes — the unbounded formula is already
finite in that limit).

`k_field` is duck-typed: any object exposing a real-typed `internal`
vector is accepted (production code passes a `CollocatedScalarField`;
the V&V suite passes a lightweight stand-in).
"""
function _apply_durbin_cap!(
        nu_t::Vector{T}, k_field,
        S_mag::Vector{T}, C_T::T,
    ) where {T}
    nc = length(nu_t)
    k_int = k_field.internal
    for c in 1:nc
        k_val = max(k_int[c], T(1.0e-10))
        s_val = S_mag[c]
        if s_val > T(1.0e-14)
            nu_t_cap = C_T * k_val / s_val
            nu_t[c] = min(nu_t[c], nu_t_cap)
        end
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

    # Fail fast with a clear message if any turbulence BCs are missing
    # (previously this defaulted to an empty Dict and errored mid-solve
    # at the first boundary face).
    _validate_turbulence_bcs(bcs_turb, mesh, model)

    # Compute the velocity gradient tensor ONCE; both the strain-rate
    # magnitude (Durbin cap) and the full-tensor production are derived
    # from it (previously the gradients were reconstructed twice).
    grad_U = _compute_velocity_gradient_tensor(U, mesh)
    S_mag = Vector{T}(undef, nc)
    S_mag_sq_cells = Vector{T}(undef, nc)
    for c in 1:nc
        S_comp = _strain_components(grad_U, c, Val(Dim))
        # |S|² = 2 S_ij S_ij (reduced-component form)
        s2 = max(_sym_self_magnitude_sq(S_comp, Val(Dim)), zero(T))
        S_mag_sq_cells[c] = s2
        S_mag[c] = sqrt(s2)
    end

    # Durbin realizability cap (v3.0). Enforces
    #   ν_t ≤ C_T · k / |S|   (C_T ≈ 0.6, Durbin 1996)
    # which is equivalent to requiring the Reynolds stress tensor from
    # the Boussinesq closure to satisfy the Schwarz inequality
    # (realizability). Active by default via `_durbin_C_T`; users can
    # override via `StandardKEpsilon(; realizability_alpha = ...)`.
    _apply_durbin_cap!(turb_state.nu_t, k_field, S_mag, _durbin_C_T(model))

    # Full-tensor production P_k = 2 ν_t S_ij S_ij.
    P_k = Vector{T}(undef, nc)
    for c in 1:nc
        P_k[c] = turb_state.nu_t[c] * S_mag_sq_cells[c]
    end

    # ── k equation ───────────────────────────────────────────────
    k_eq = _cached_equation!(turb_state, :k, mesh)
    bcs_k = bcs_turb[:k]

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
    eps_eq = _cached_equation!(turb_state, :epsilon, mesh)
    bcs_eps = bcs_turb[:epsilon]

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

# ── Realizability hook (see interface.jl) ────────────────────────────

"""
    _apply_realizability!(turb_state, model::StandardKEpsilon, U, mesh)

Re-apply the Durbin cap after `turbulent_viscosity!` recomputes
`nu_t = C_mu k²/ε` from the transport fields.  Without this, the cap
applied inside `solve_turbulence!` was discarded by the uncapped
recompute in `_update_turbulence!`, so the momentum equation never saw
the realizability-limited eddy viscosity.
"""
function _apply_realizability!(
        turb_state::RANSTurbulenceState{T},
        model::StandardKEpsilon,
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    S_mag = compute_strain_rate(U, mesh)
    _apply_durbin_cap!(turb_state.nu_t, turb_state.fields[:k], S_mag, _durbin_C_T(model))
    return nothing
end

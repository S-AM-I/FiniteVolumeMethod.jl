# turbulence/spalart_allmaras.jl — Spalart-Allmaras one-equation turbulence model
#
# Single transport equation for modified turbulent viscosity ν̃.
# Good near-wall behavior without wall functions. Requires wall distance.

"""
    SpalartAllmaras{T} <: AbstractRANSModel

Spalart-Allmaras one-equation turbulence model.

# Fields
Standard SA constants with default values.
"""
struct SpalartAllmaras{T} <: AbstractRANSModel
    cb1::T      # 0.1355
    cb2::T      # 0.622
    sigma::T    # 2/3
    kappa::T    # 0.41
    cw2::T      # 0.3
    cw3::T      # 2.0
    cv1::T      # 7.1
    ct3::T      # 1.2
    ct4::T      # 0.5
    d_wall::Vector{T}  # wall distance per cell
end

function SpalartAllmaras(
        mesh::UnstructuredFVMMesh{Dim, T}, wall_patches::Vector{Symbol};
        cb1 = 0.1355, cb2 = 0.622, sigma = 2.0 / 3.0, kappa = 0.41,
        cw2 = 0.3, cw3 = 2.0, cv1 = 7.1, ct3 = 1.2, ct4 = 0.5,
    ) where {Dim, T}
    d_wall = compute_wall_distance(mesh, wall_patches)
    Tc = promote_type(typeof(cb1), typeof(cb2), typeof(sigma), T)
    return SpalartAllmaras{Tc}(
        Tc(cb1), Tc(cb2), Tc(sigma), Tc(kappa),
        Tc(cw2), Tc(cw3), Tc(cv1), Tc(ct3), Tc(ct4), Tc.(d_wall),
    )
end

n_turbulence_fields(::SpalartAllmaras) = 1
turbulence_field_names(::SpalartAllmaras) = (:nu_tilde,)

# ── SA helper functions ──────────────────────────────────────────────

_sa_chi(nu_tilde::T, nu::T) where {T} = nu_tilde / max(nu, T(1.0e-15))

function _sa_fv1(chi::T, cv1::T) where {T}
    chi3 = chi^3
    return chi3 / (chi3 + cv1^3)
end

function _sa_fv2(chi::T, cv1::T) where {T}
    fv1 = _sa_fv1(chi, cv1)
    return one(T) - chi / (one(T) + chi * fv1)
end

function _sa_fw(r::T, cw2::T, cw3::T) where {T}
    g = r + cw2 * (r^6 - r)
    cw3_6 = cw3^6
    return g * ((one(T) + cw3_6) / (g^6 + cw3_6))^(one(T) / 6)
end

# ── Interface implementation ─────────────────────────────────────────

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::SpalartAllmaras{T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nt_field = turb_state.fields[:nu_tilde]
    nc = length(mesh.cell_volumes)
    # Compute fv1 using a typical laminar nu estimate (1e-5 for air, 1e-6 for water).
    # The exact value matters little when nu_tilde >> nu (turbulent region).
    nu_est = T(1.0e-5)
    for c in 1:nc
        nt = max(nt_field.internal[c], zero(T))
        chi = nt / max(nu_est, T(1.0e-15))
        chi3 = chi^3
        fv1 = chi3 / (chi3 + model.cv1^3)
        nu_t[c] = nt * fv1
    end
    return nothing
end

"""
    turbulent_viscosity_sa!(nu_t, model, turb_state, mesh, nu)

SA-specific version that takes laminar viscosity for correct fv1 computation.
"""
function turbulent_viscosity_sa!(
        nu_t::Vector{T},
        model::SpalartAllmaras{T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        nu::T,
    ) where {Dim, T}
    nt_field = turb_state.fields[:nu_tilde]
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        nt = max(nt_field.internal[c], zero(T))
        chi = _sa_chi(nt, nu)
        fv1 = _sa_fv1(chi, model.cv1)
        nu_t[c] = nt * fv1
    end
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::SpalartAllmaras{T},
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
    nt_field = turb_state.fields[:nu_tilde]

    # Strain rate
    S_mag = compute_strain_rate(U, mesh)

    # Derived constant
    cw1 = model.cb1 / model.kappa^2 + (one(T) + model.cb2) / model.sigma

    # ── ν̃ equation ────────────────────────────────────────────────
    nt_eq = CollocatedEquation(mesh)
    bcs_nt = get(bcs_turb, :nu_tilde, Dict{Symbol, AbstractBoundaryCondition}())

    # Convection
    assemble_convection!(nt_eq, phi, mesh, bcs_nt)

    # Diffusion: (1/sigma) * div((nu + nu_tilde) * grad(nu_tilde))
    gamma_nt = Vector{T}(undef, nc)
    for c in 1:nc
        nt = max(nt_field.internal[c], zero(T))
        gamma_nt[c] = (nu + nt) / model.sigma
    end
    assemble_laplacian!(nt_eq, gamma_nt, mesh, bcs_nt)

    # Temporal term
    if dt !== nothing
        assemble_ddt_euler!(nt_eq, one(T), nt_field.internal, mesh, dt)
    end

    # Source terms (production - destruction, linearized)
    for c in 1:nc
        nt = max(nt_field.internal[c], T(1.0e-10))
        d = max(model.d_wall[c], T(1.0e-10))
        chi = _sa_chi(nt, nu)
        fv1 = _sa_fv1(chi, model.cv1)
        fv2 = _sa_fv2(chi, model.cv1)

        # Modified vorticity
        S_tilde = S_mag[c] + nt / (model.kappa^2 * d^2) * fv2
        S_tilde = max(S_tilde, T(1.0e-10))

        # Production: cb1 * S_tilde * nu_tilde (implicit in nu_tilde)
        # Treat as: S_C = cb1 * S_tilde (coefficient on nu_tilde, goes to diagonal with negative sign)
        # Actually production is positive, so: b += cb1 * S_tilde * nt * V
        # But for linearization, we want it implicit: A[c,c] -= cb1 * S_tilde * V
        # (negative because it's a source, reducing A makes the diagonal smaller → source)
        # OpenFOAM convention: positive source → subtract from diagonal
        # Here: production goes to RHS: b[c] += cb1 * S_tilde * nt * V_c
        nt_eq.b[c] += model.cb1 * S_tilde * nt * mesh.cell_volumes[c]

        # Destruction: -cw1 * fw * (nu_tilde/d)^2 — linearize as implicit
        r_val = min(nt / (S_tilde * model.kappa^2 * d^2), T(10))
        fw = _sa_fw(r_val, model.cw2, model.cw3)
        # D = cw1 * fw * (nt/d^2) * nt → S_P = cw1 * fw * nt / d^2
        add_diag!(nt_eq, c, cw1 * fw * nt / d^2 * mesh.cell_volumes[c])
    end

    # cb2/sigma * |grad(nu_tilde)|^2 term (explicit source)
    grad_nt = gradient(nt_field, mesh)
    for c in 1:nc
        grad_sq = dot(grad_nt[c], grad_nt[c])
        nt_eq.b[c] += model.cb2 / model.sigma * grad_sq * mesh.cell_volumes[c]
    end

    # Solve
    lp = to_linear_problem(nt_eq)
    sol = _dispatch_solve(lp, linear_solver, solver_config, :nu_tilde)
    for c in 1:nc
        nt_field.internal[c] = max(sol.u[c], zero(T))
    end

    # Update nu_t with correct fv1
    turbulent_viscosity_sa!(turb_state.nu_t, model, turb_state, mesh, nu)

    return nothing
end

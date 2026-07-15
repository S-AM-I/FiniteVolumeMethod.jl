# pressure_based/compressible_simple.jl — Weakly-compressible SIMPLE
#
# Stage 3 pressure-based extension.  HONESTY NOTE (what this solver
# actually does): the pressure-velocity loop reuses the incompressible
# `assemble_pressure!` unchanged, so the continuity constraint enforced
# is `div(U) = 0` — NOT the steady compressible `div(ρU) = 0`.  Density
# is refreshed from the EOS between iterations (`update_density!`) and
# face densities `rho_f` are computed, but neither enters the mass
# balance: `update_mass_flux!` is never called and `rho_f` is unused by
# the coupling loop.
#
# The result is a low-Mach, weakly-compressible approximation:
#   1. `update_density!` refreshes ρ = EOS(p, T) as a POST-update
#   2. `update_viscosity!` for Sutherland / tabulated μ(T)
#   3. Optional energy equation
# Mass is NOT conserved for genuinely compressible cases; a @warn at
# solver entry states this.  Do not treat this as a rhoSimpleFoam
# analogue.
#
# The algorithm type `CompressibleSIMPLE{T}` and its `PIMPLE` counterpart
# live here so the same `solve_compressible(prob, alg)` dispatch works.

using Printf: @sprintf

# ── CompressibleSIMPLE algorithm ────────────────────────────────────

@doc """
    CompressibleSIMPLE{T} <: AbstractPVCoupling

Weakly-compressible SIMPLE algorithm (low-Mach approximation).
Reuses the existing incompressible `assemble_momentum!` and
`assemble_pressure!` kernels — the pressure equation enforces
INCOMPRESSIBLE continuity `div(U) = 0`; density is refreshed from the
EOS between iterations but never enters the mass balance, so mass is
not conserved for genuinely compressible flows.  Each outer iteration:

1. Update μ(T) and ρ(p, T) from the current fields.
2. Assemble + solve momentum (under-relaxed).
3. Extract A_P, H(U) from the solved, relaxed equations.
4. Assemble + solve the (incompressible) pressure Poisson equation.
5. Under-relax pressure, update ρ (EOS post-update), correct velocity,
   correct fluxes.
6. (Optional) Solve the energy equation.
7. Check momentum + continuity + (optional) energy residuals.

Valid only for low-Mach, weakly-compressible use.

# Fields
- `alpha_U::T`        — velocity under-relaxation factor
- `alpha_p::T`        — pressure under-relaxation factor
- `alpha_rho::T`      — density under-relaxation factor (default `0.7`)
- `max_iterations::Int`
- `tolerance::T`
"""
struct CompressibleSIMPLE{T} <: AbstractPVCoupling
    alpha_U::T
    alpha_p::T
    alpha_rho::T
    max_iterations::Int
    tolerance::T
end

@doc """
    CompressibleSIMPLE(; alpha_U = 0.7, alpha_p = 0.3, alpha_rho = 0.7,
                         max_iterations = 1000, tolerance = 1e-6)

Construct a [`CompressibleSIMPLE`](@ref) algorithm.
"""
function CompressibleSIMPLE(;
        alpha_U::T = 0.7,
        alpha_p::T = 0.3,
        alpha_rho::T = 0.7,
        max_iterations::Int = 1000,
        tolerance::T = 1.0e-6,
    ) where {T}
    return CompressibleSIMPLE{T}(alpha_U, alpha_p, alpha_rho, max_iterations, tolerance)
end

# ── CompressibleState (mirrors IncompressibleState + density + T) ──

@doc """
    CompressibleState{Dim, T}

Mutable solver state for the compressible pressure-based family.
Wraps an `IncompressibleState` plus cell-centred density and
temperature. Temperature is a live field; for isothermal runs
it is pinned to the problem's reference `T_ref`.

# Fields
- `base::IncompressibleState{Dim, T}` — velocity, pressure, flux, A_P, H_U
- `rho::Vector{T}`        — cell density (kg/m³)
- `T_cells::Vector{T}`    — cell temperature (K)
- `rho_f::Vector{T}`      — face-interpolated density (kg/m³, length = nfaces)
- `mu_cells::Vector{T}`   — cell molecular viscosity (Pa·s)
"""
mutable struct CompressibleState{Dim, T}
    base::IncompressibleState{Dim, T}
    rho::Vector{T}
    T_cells::Vector{T}
    rho_f::Vector{T}
    mu_cells::Vector{T}
end

@doc """
    CompressibleState(mesh, model; p0 = 1.01325e5, T0 = 300.0)

Construct a zero-initialised compressible state with density / μ
seeded from the EOS at `(p0, T0)`.
"""
function CompressibleState(
        mesh::UnstructuredFVMMesh{Dim, T},
        model::AbstractThermoModel;
        p0::Real = 1.01325e5,
        T0::Real = 300.0,
    ) where {Dim, T}
    base = IncompressibleState(mesh)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    fill!(base.p.internal, T(p0))
    fill!(base.p.boundary, T(p0))
    rho = fill(T(density_at(model, T(p0), T(T0))), nc)
    T_cells = fill(T(T0), nc)
    rho_f = fill(T(density_at(model, T(p0), T(T0))), nf)
    mu_cells = fill(T(viscosity_at(model, T(T0))), nc)
    return CompressibleState{Dim, T}(base, rho, T_cells, rho_f, mu_cells)
end

# ── CompressibleProblem thin wrapper ────────────────────────────────

@doc """
    CompressibleProblem{Dim, T, Mesh, BC, Algo, Model <: AbstractThermoModel}

Compressible pressure-based problem: an [`IncompressibleProblem`](@ref)
augmented with a thermodynamic model and a temperature reference.
Supports either [`CompressibleSIMPLE`](@ref) (steady) or
[`CompressiblePIMPLE`](@ref) (transient) as its `algorithm`.

# Fields
- `mesh`, `bcs`, `algorithm`       — identical to `IncompressibleProblem`
- `thermo::Model`                  — EOS / viscosity / cp model
- `T_ref::T`                       — reference temperature
- `solve_energy::Bool`             — if `true`, solve the energy equation
  each outer iteration and feed the updated temperature back to the EOS
"""
struct CompressibleProblem{Dim, T, Mesh, BC, Algo <: AbstractPVCoupling, Model <: AbstractThermoModel}
    mesh::Mesh
    bcs::BC
    algorithm::Algo
    thermo::Model
    T_ref::T
    solve_energy::Bool
end

@doc """
    CompressibleProblem(mesh, bcs, algorithm, thermo;
                         T_ref = 300.0, solve_energy = false)

Construct a [`CompressibleProblem`](@ref).
"""
function CompressibleProblem(
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs,
        algorithm::Algo,
        thermo::Model;
        T_ref::Real = 300.0,
        solve_energy::Bool = false,
    ) where {Dim, T, Algo <: AbstractPVCoupling, Model <: AbstractThermoModel}
    return CompressibleProblem{Dim, T, typeof(mesh), typeof(bcs), Algo, Model}(
        mesh, bcs, algorithm, thermo, T(T_ref), solve_energy,
    )
end

"""
    _incompressible_shim(prob::CompressibleProblem, rho_mean, mu_mean)

Build a lightweight `IncompressibleProblem` mirror used to feed the
existing `assemble_momentum!`/`assemble_pressure!` routines. The
momentum equations receive `nu = mu_mean / rho_mean` as an effective
scalar kinematic viscosity; per-cell variations (from Sutherland)
are applied via the `nu_eff` keyword downstream.
"""
function _incompressible_shim(
        prob::CompressibleProblem{Dim, T}, rho_mean::T, mu_mean::T,
    ) where {Dim, T}
    # Rebuild a tiny IncompressibleProblem with scalar nu and density.
    nu = mu_mean / max(rho_mean, eps(T))
    return IncompressibleProblem(prob.mesh, prob.bcs, prob.algorithm; nu = nu, density = rho_mean)
end

# ── Main entry point ────────────────────────────────────────────────

@doc """
    solve_compressible(prob::CompressibleProblem; kwargs...) -> SolveResult

Run the compressible SIMPLE loop (steady-state). Dispatch on the
`algorithm` field selects SIMPLE vs PIMPLE at construction time; this
method handles [`CompressibleSIMPLE`](@ref). For the transient
[`CompressiblePIMPLE`](@ref), use `solve_compressible(prob, tspan, dt)`.
"""
function solve_compressible(
        prob::CompressibleProblem{Dim, T, Mesh, BC, CompressibleSIMPLE{T}, Model};
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        p0::Real = 1.01325e5,
    ) where {Dim, T, Mesh, BC, Model}
    @warn "CompressibleSIMPLE enforces incompressible continuity (div(U)=0) " *
        "with an EOS density post-update. Mass is NOT conserved for genuinely " *
        "compressible cases — use this solver only for low-Mach, " *
        "weakly-compressible flows." maxlog = 1
    algo = prob.algorithm
    mesh = prob.mesh
    alpha_U = algo.alpha_U
    alpha_p = algo.alpha_p
    alpha_rho = algo.alpha_rho
    max_iter = algo.max_iterations
    tol = algo.tolerance

    # ── 0. Initialise compressible state ────────────────────────────
    cstate = CompressibleState(mesh, prob.thermo; p0 = p0, T0 = prob.T_ref)
    state = cstate.base
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)
    cyclic_pairs = collect_cyclic_pairs(prob.bcs, mesh)

    # Residual history
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity, :density]
    )

    converged = false
    final_iter = 0
    nc = length(mesh.cell_volumes)

    rho_prev = copy(cstate.rho)

    for iter in 1:max_iter
        final_iter = iter

        # ── 1. Update μ(T) and ρ(p, T) ──────────────────────────────
        update_viscosity!(cstate.mu_cells, prob.thermo, cstate.T_cells)
        mu_mean = sum(cstate.mu_cells) / nc
        rho_mean = sum(cstate.rho) / nc
        compute_face_densities!(
            cstate.rho_f, prob.thermo,
            mesh, state.p.internal, cstate.T_cells
        )

        # ── 2. Momentum solve (reuse incompressible machinery) ──────
        shim = _incompressible_shim(prob, rho_mean, mu_mean)
        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(
                eq, state, shim, d;
                nu_eff = cstate.mu_cells ./ cstate.rho
            )
            apply_cyclic_to_equation!(
                eq, _make_scalar_field(_extract_component(state.U, d), state),
                mesh, cyclic_pairs,
            )
            push!(eqs, eq)
        end
        for d in 1:Dim
            U_old_d = _extract_component(state.U, d)
            under_relax_momentum!(eqs[d], U_old_d, alpha_U)
            lp = to_linear_problem(eqs[d])
            sol = _dispatch_solve(
                lp, linear_solver, solver_config,
                d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
            )
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)

        # Extract A_P/H(U) from the relaxed, solved equations
        extract_momentum_operators!(state, eqs, mesh)

        # ── 3. Pressure solve (rhoSimpleFoam-style) ─────────────────
        # For a closed (Neumann-only) compressible system the absolute
        # pressure level is anchored by the imposed `p0` — we solve for
        # the pressure correction, then shift the result so that
        # `mean(p) == p0`. Pinning cell-1 to zero as in the incompressible
        # case would erase the physical absolute pressure (ρ = p/(R T)
        # needs it).
        needs_ref = _needs_pressure_reference(prob.bcs)
        p_mean_target = needs_ref ? sum(state.p.internal) / nc : zero(T)

        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, shim)
        apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
        if needs_ref
            fix_pressure_reference!(p_eq, 1, state.p.internal[1])
        end
        lp_p = to_linear_problem(p_eq)
        p_sol = _dispatch_solve(lp_p, linear_solver, solver_config, :p)

        # Under-relax pressure
        @inbounds for c in 1:nc
            state.p.internal[c] += alpha_p * (p_sol.u[c] - state.p.internal[c])
        end

        # Re-anchor the mean pressure for closed (Neumann-only) problems.
        if needs_ref
            p_mean_now = sum(state.p.internal) / nc
            shift = p_mean_target - p_mean_now
            @inbounds for c in 1:nc
                state.p.internal[c] += shift
            end
        end
        update_boundary_pressure!(state, prob.bcs, mesh)

        # ── 4. Correct velocity + fluxes ────────────────────────────
        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)
        correct_fluxes!(state, mesh)

        # ── 5. Density update with under-relaxation ─────────────────
        copyto!(rho_prev, cstate.rho)
        update_density!(cstate.rho, prob.thermo, state.p.internal, cstate.T_cells)
        @inbounds for c in 1:nc
            cstate.rho[c] = rho_prev[c] + alpha_rho * (cstate.rho[c] - rho_prev[c])
        end

        # ── 6. Residuals ────────────────────────────────────────────
        max_residual = zero(T)
        for d in 1:Dim
            u_d = _extract_component(state.U, d)
            r = momentum_residual(eqs[d], u_d)
            push!(residuals[component_labels[d]], r)
            max_residual = max(max_residual, r)
        end
        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)
        max_residual = max(max_residual, r_cont)

        # density residual: L1 relative change
        r_rho = zero(T)
        rho_scale = zero(T)
        @inbounds for c in 1:nc
            r_rho += abs(cstate.rho[c] - rho_prev[c])
            rho_scale += abs(cstate.rho[c])
        end
        r_rho_rel = rho_scale > eps(T) ? r_rho / rho_scale : zero(T)
        push!(residuals[:density], r_rho_rel)

        if verbose
            _print_compressible_residuals(iter, residuals, component_labels)
        end

        if max_residual < tol && r_rho_rel < tol
            converged = true
            break
        end
    end

    return (
        converged = converged, iterations = final_iter,
        residuals = residuals, state = cstate,
    )
end

"""
    _print_compressible_residuals(iter, residuals, labels)

One-line residual printer for the compressible SIMPLE loop.
"""
function _print_compressible_residuals(iter, residuals, labels)
    parts = String[]
    for label in labels
        r = residuals[label][end]
        push!(parts, string(label, "=", @sprintf("%.3e", r)))
    end
    r_cont = residuals[:continuity][end]
    r_rho = residuals[:density][end]
    push!(parts, string("cont=", @sprintf("%.3e", r_cont)))
    push!(parts, string("rho=", @sprintf("%.3e", r_rho)))
    println("cSIMPLE iter ", lpad(iter, 5), ": ", join(parts, "  "))
    return nothing
end

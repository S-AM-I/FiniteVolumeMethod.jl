# pressure_based/compressible_pimple.jl — Compressible (subsonic) PIMPLE
#
# Transient counterpart of `CompressibleSIMPLE`. Structure mirrors the
# incompressible `_pimple_step!`: outer SIMPLE-like passes with inner
# PISO correctors.  The pressure equation is the linearized compressible
# continuity `ddt(ψ p) + div(ρ_f φ) = 0` (rhoPimpleFoam subsonic branch):
#
#   * implicit `ψ V/dt` diagonal (ψ = ∂ρ/∂p from `psi_at`) — this also
#     removes the Neumann null space, so closed domains need NO pressure
#     reference or mean anchor; the absolute pressure level is set by the
#     mass content, which is the physical behaviour;
#   * density-weighted H/A mass flux via `compute_face_densities!` +
#     `update_mass_flux!`;
#   * conservative linearized density update `ρ ← ρ* + ψ(p_new − p*)` in
#     the FINAL outer pass, which makes Σ ρ V telescoping — total mass in
#     a closed adiabatic box is conserved to linear-solver tolerance.
#     Non-final outer passes refresh ρ = EOS(p, T) (Picard).
#
# The momentum equation stays kinematic with the `-(1/ρ)∇p` pressure
# source, which reproduces the correct isothermal acoustics
# (c² = 1/ψ = R T for an ideal gas); the ddt(ρ U) density variation in
# the momentum temporal term is neglected (low-Mach momentum
# approximation, documented limitation for strongly compressible flow).

using Printf: @sprintf

# ── CompressiblePIMPLE algorithm ────────────────────────────────────

@doc """
    CompressiblePIMPLE{T} <: AbstractPVCoupling

Transient compressible pressure-based coupling (subsonic).  Combines
outer SIMPLE-style under-relaxation with inner PISO correctors on the
linearized compressible continuity `ddt(ψ p) + div(ρ_f φ) = 0`, with a
conservative density update in the final pass so total mass is
conserved to linear-solver tolerance in closed domains.  For transonic
or supersonic flow the density-based stack in `src/hyperbolic/` is
appropriate (the `fvm::div(phid, p)` transonic term is not implemented).

# Fields
- `n_outer::Int`       — number of outer iterations per time step
- `n_correctors::Int`  — number of inner pressure correctors
- `alpha_U::T`
- `alpha_p::T`
- `alpha_rho::T`
- `tolerance::T`
"""
struct CompressiblePIMPLE{T} <: AbstractPVCoupling
    n_outer::Int
    n_correctors::Int
    alpha_U::T
    alpha_p::T
    alpha_rho::T
    tolerance::T
end

@doc """
    CompressiblePIMPLE(; n_outer = 2, n_correctors = 1,
                         alpha_U = 0.7, alpha_p = 0.3, alpha_rho = 0.7,
                         tolerance = 1e-6)

Construct a [`CompressiblePIMPLE`](@ref) algorithm.
"""
function CompressiblePIMPLE(;
        n_outer::Int = 2,
        n_correctors::Int = 1,
        alpha_U::T = 0.7,
        alpha_p::T = 0.3,
        alpha_rho::T = 0.7,
        tolerance::T = 1.0e-6,
    ) where {T}
    return CompressiblePIMPLE{T}(n_outer, n_correctors, alpha_U, alpha_p, alpha_rho, tolerance)
end

# ── Compressible PIMPLE step ────────────────────────────────────────

"""
    _compressible_pimple_step!(cstate, prob, dt; kwargs...)

Advance the compressible state by one time step of size `dt` using
the [`CompressiblePIMPLE`](@ref) algorithm.  See the file header for the
discretization; the final outer pass uses the conservative linearized
density update so `Σ ρ V` is preserved to linear-solver tolerance in
closed domains.

`ws` is an optional pre-allocated `(eqs, p_eq)` workspace (from
`_make_incompressible_workspace`-style construction) reused across time
steps; `psi_work`/`p_star_work`/`rho_old_work` are per-cell scratch
vectors allocated once by the driver.
"""
function _compressible_pimple_step!(
        cstate::CompressibleState{Dim, T},
        prob::CompressibleProblem{Dim, T, Mesh, BC, CompressiblePIMPLE{T}, Model},
        dt::T;
        linear_solver = nothing,
        solver_config = nothing,
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}} = Vector{Vector{Tuple{Int, Int}}}(),
        ws = nothing,
        psi_work::Union{Nothing, Vector{T}} = nothing,
        p_star_work::Union{Nothing, Vector{T}} = nothing,
        rho_old_work::Union{Nothing, Vector{T}} = nothing,
    ) where {Dim, T, Mesh, BC, Model}
    algo = prob.algorithm
    mesh = prob.mesh
    n_outer = algo.n_outer
    n_correctors = algo.n_correctors
    alpha_U = algo.alpha_U
    alpha_p = algo.alpha_p

    state = cstate.base
    nc = length(mesh.cell_volumes)

    # Workspace (equations pre-allocated once per solve when driven by
    # `solve_compressible`; standalone calls allocate here).
    if ws === nothing
        cell_pairs = _cyclic_cell_pairs(mesh, cyclic_pairs)
        eqs = [CollocatedEquation(mesh; extra_cell_pairs = cell_pairs) for _ in 1:Dim]
        p_eq = CollocatedEquation(mesh; extra_cell_pairs = cell_pairs)
    else
        eqs, p_eq = ws
    end
    psi = psi_work === nothing ? Vector{T}(undef, nc) : psi_work
    p_star = p_star_work === nothing ? Vector{T}(undef, nc) : p_star_work
    rho_old = rho_old_work === nothing ? Vector{T}(undef, nc) : rho_old_work

    # Old-time snapshots for the ddt terms (velocity AND density)
    _snapshot_old_time!(state)
    copyto!(rho_old, cstate.rho)

    for outer in 1:n_outer
        is_final = (outer == n_outer)

        # Property updates: μ(T), ρ_f(p, T), ψ(p, T)
        update_viscosity!(cstate.mu_cells, prob.thermo, cstate.T_cells)
        mu_mean = sum(cstate.mu_cells) / nc
        rho_mean = sum(cstate.rho) / nc
        compute_face_densities!(
            cstate.rho_f, prob.thermo,
            mesh, state.p.internal, cstate.T_cells
        )
        update_psi!(psi, prob.thermo, state.p.internal, cstate.T_cells)

        shim = _incompressible_shim(prob, rho_mean, mu_mean)

        # Momentum assemble + solve (kinematic, -(1/ρ)∇p source)
        for d in 1:Dim
            reset!(eqs[d])
            assemble_momentum!(
                eqs[d], state, shim, d; dt = dt,
                nu_eff = cstate.mu_cells ./ cstate.rho,
                rho_p = cstate.rho,
            )
            apply_cyclic_to_equation!(
                eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
                mesh, cyclic_pairs,
            )
        end
        for d in 1:Dim
            if !is_final
                U_old_d = _extract_component(state.U, d)
                under_relax_momentum!(eqs[d], U_old_d, alpha_U)
            end
            lp = to_linear_problem(eqs[d])
            sol = _dispatch_solve(
                lp, linear_solver, solver_config,
                d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
            )
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)

        # Extract A_P/H(U) from the (relaxed) solved equations
        extract_momentum_operators!(state, eqs, mesh; rho_p = cstate.rho)

        # PISO inner corrector loop on the compressible continuity
        # ddt(ψ p) + div(ρ_f φ) = 0.  The ψ V/dt diagonal removes the
        # Neumann null space — no pressure reference / mean anchor needed
        # (anchoring would destroy mass conservation).
        for k in 1:n_correctors
            copyto!(p_star, state.p.internal)   # linearization point p*

            reset!(p_eq)
            assemble_pressure_compressible!(
                p_eq, state, shim, cstate.rho, cstate.rho_f, mesh;
                psi = psi, rho_old = rho_old, dt = dt,
            )
            apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
            lp_p = to_linear_problem(p_eq)
            p_sol = _dispatch_solve(lp_p, linear_solver, solver_config, :p)

            if !is_final
                @inbounds for c in 1:nc
                    state.p.internal[c] += alpha_p * (p_sol.u[c] - state.p.internal[c])
                end
            else
                @inbounds for c in 1:nc
                    state.p.internal[c] = p_sol.u[c]
                end
            end
            update_boundary_pressure!(state, prob.bcs, mesh)

            if is_final
                # Conservative update: ρ ← ρ* + ψ(p_new − p*) matches the
                # assembled linearization exactly → Σ ρ V telescopes.
                _conservative_density_update!(
                    cstate.rho, psi, state.p.internal, p_star,
                )
            end

            correct_velocity!(state, mesh; rho_p = cstate.rho)
            update_boundary_velocity!(state, prob.bcs, mesh)
            update_boundary_cyclic!(state, mesh, cyclic_pairs)
            _correct_fluxes_compressible!(cstate, mesh)
        end

        # Non-final outers: Picard refresh ρ = EOS(p, T) so properties
        # track the intermediate pressure (mass balance is restored by
        # the final conservative update).
        if !is_final
            update_density!(cstate.rho, prob.thermo, state.p.internal, cstate.T_cells)
        end
    end

    return nothing
end

# ── Transient driver ────────────────────────────────────────────────

@doc """
    solve_compressible(prob, tspan, dt; kwargs...) -> NamedTuple

Transient compressible PIMPLE solver. Advances the state from
`tspan[1]` to `tspan[2]` with fixed time step `dt`, calling
`_compressible_pimple_step!` each time.

# Keyword Arguments
- `linear_solver`, `solver_config`, `verbose` — as in `solve_simple`
- `p0` — uniform initial absolute pressure (Pa)
- `p_init` — optional per-cell initial pressure field overriding `p0`
  (e.g. a pressure perturbation for acoustic tests); density is seeded
  consistently from the EOS

# Returns
A named tuple with fields:
- `converged::Bool`
- `iterations::Int`         — number of time steps taken
- `residuals::Dict`         — residual history; `:total_mass` records
  `Σ ρ V` after every step (conserved in closed domains)
- `state::CompressibleState`
"""
function solve_compressible(
        prob::CompressibleProblem{Dim, T, Mesh, BC, CompressiblePIMPLE{T}, Model},
        tspan::Tuple{T, T},
        dt::T;
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        p0::Real = 1.01325e5,
        p_init::Union{Nothing, Vector{T}} = nothing,
    ) where {Dim, T, Mesh, BC, Model}
    mesh = prob.mesh
    t_start, t_end = tspan
    nc = length(mesh.cell_volumes)

    cstate = CompressibleState(mesh, prob.thermo; p0 = p0, T0 = prob.T_ref)
    state = cstate.base
    if p_init !== nothing
        length(p_init) == nc || throw(ArgumentError("p_init must have length ncells = $nc"))
        copyto!(state.p.internal, p_init)
        update_density!(cstate.rho, prob.thermo, state.p.internal, cstate.T_cells)
    end
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)
    cyclic_pairs = collect_cyclic_pairs(prob.bcs, mesh)

    # Workspaces: equations + per-cell scratch, allocated once per solve.
    cell_pairs = _cyclic_cell_pairs(mesh, cyclic_pairs)
    eqs = [CollocatedEquation(mesh; extra_cell_pairs = cell_pairs) for _ in 1:Dim]
    p_eq = CollocatedEquation(mesh; extra_cell_pairs = cell_pairs)
    psi_work = Vector{T}(undef, nc)
    p_star_work = Vector{T}(undef, nc)
    rho_old_work = Vector{T}(undef, nc)

    residuals = Dict{Symbol, Vector{T}}(
        :continuity => T[], :density => T[], :total_mass => T[],
    )

    t = t_start
    n_steps = 0
    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)
        _compressible_pimple_step!(
            cstate, prob, dt_actual;
            linear_solver = linear_solver,
            solver_config = solver_config,
            cyclic_pairs = cyclic_pairs,
            ws = (eqs, p_eq),
            psi_work = psi_work, p_star_work = p_star_work,
            rho_old_work = rho_old_work,
        )
        t += dt_actual
        n_steps += 1

        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)
        push!(residuals[:density], sum(cstate.rho) / length(cstate.rho))
        push!(residuals[:total_mass], total_mass(cstate, mesh))

        if verbose
            println(
                "cPIMPLE step ", lpad(n_steps, 6),
                "  t=", @sprintf("%.4e", t),
                "  cont=", @sprintf("%.3e", r_cont),
                "  mass=", @sprintf("%.10e", residuals[:total_mass][end])
            )
        end
    end

    r_hist = residuals[:continuity]
    converged = isempty(r_hist) || isfinite(r_hist[end])

    return (
        converged = converged, iterations = n_steps,
        residuals = residuals, state = cstate,
    )
end

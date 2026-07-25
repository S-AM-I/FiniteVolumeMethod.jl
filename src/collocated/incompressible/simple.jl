# incompressible/simple.jl — SIMPLE steady-state solver loop
#
# Implements the Semi-Implicit Method for Pressure-Linked Equations
# (SIMPLE) algorithm for steady-state incompressible Navier-Stokes.
# Each outer iteration assembles momentum equations with under-relaxation,
# solves the pressure Poisson equation, and applies velocity/flux corrections.

using Printf: @sprintf

# ── Shared SIMPLE outer-iteration core ──────────────────────────────

@doc """
    _simple_outer_step!(state, prob, eqs, p_eq, cyclic_pairs, residuals, labels;
                        nu_eff, body_force, scheme, blend, kwargs...) -> max_res

Run one SIMPLE outer iteration on `state` in place: assemble and
under-relax the momentum equations, extract `A_P`/`H(U)` from the relaxed
solve, solve the pressure correction, under-relax and correct velocity and
fluxes, and append the momentum and continuity residuals to `residuals`.
Returns the maximum residual for this iteration.

The physics coupling enters only through `nu_eff` (effective viscosity from
a turbulence model) and `body_force` (e.g. buoyancy), both forwarded to
[`assemble_momentum!`](@ref).  Every physics-specific steady loop
(turbulent, thermal, radiation, combustion) calls this core and then runs
its own scalar-transport solves around it, so the momentum/pressure
discretisation lives in exactly one place — including cyclic coupling and
equation reuse, which the physics loops previously lacked.

Convergence is NOT decided here: the caller owns the tolerance test and the
first-iteration guard, because the coupled fields it advances afterwards
must feed back before convergence is meaningful.
"""
function _simple_outer_step!(
        state::IncompressibleState{Dim, T},
        prob::AnyIncompressibleProblem{Dim, T},
        eqs::Vector{CollocatedEquation{T}},
        p_eq::CollocatedEquation{T},
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}},
        residuals::Dict{Symbol, Vector{T}},
        component_labels;
        nu_eff::Union{T, Vector{T}} = prob.nu,
        body_force::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
        linear_solver = nothing,
        solver_config = nothing,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = prob.model.porous_zones,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = prob.model.mrf_zones,
    ) where {Dim, T}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh
    alpha_U = algo.alpha_U
    alpha_p = algo.alpha_p

    # ── 1. Assemble momentum equations ──────────────────────────────
    for d in 1:Dim
        reset!(eqs[d])
        assemble_momentum!(
            eqs[d], state, prob, d;
            nu_eff = nu_eff, body_force = body_force,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
            scheme = scheme, blend = blend,
        )
        apply_cyclic_to_equation!(
            eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
            mesh, cyclic_pairs,
        )
    end

    # ── 2. Under-relax + solve momentum ─────────────────────────────
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

    # ── 3. Update boundary velocity ─────────────────────────────────
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_cyclic!(state, mesh, cyclic_pairs)

    # ── 4. Extract operators (A_P, H_U) from the RELAXED, solved
    # momentum equations — standard SIMPLE ordering, so that D = V/A_P
    # in the pressure equation is consistent with the velocity actually
    # produced by the momentum solve.
    extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)

    # ── 5. Assemble + solve pressure ────────────────────────────────
    reset!(p_eq)
    assemble_pressure!(p_eq, state, prob; mrf_zones = mrf_zones)
    apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
    if _needs_pressure_reference(prob.bcs)
        fix_pressure_reference!(p_eq, 1, zero(T))
    end
    lp_p = to_linear_problem(p_eq)
    p_sol = _dispatch_solve(lp_p, linear_solver, solver_config, :p)

    # ── 6. Under-relax pressure ─────────────────────────────────────
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        state.p.internal[c] += alpha_p * (p_sol.u[c] - state.p.internal[c])
    end

    # ── 7. Update boundary pressure ─────────────────────────────────
    update_boundary_pressure!(state, prob.bcs, mesh)

    # ── 8. Correct velocity + fluxes ────────────────────────────────
    correct_velocity!(state, mesh; porous_zones = porous_zones)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_cyclic!(state, mesh, cyclic_pairs)
    correct_fluxes!(state, mesh; porous_zones = porous_zones)
    if mrf_zones !== nothing
        mrf_make_relative!(state.phi.values, mesh, mrf_zones)
    end

    # ── 9. Compute residuals ────────────────────────────────────────
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

    return max_residual
end

# ── SIMPLE solver ──────────────────────────────────────────────────

@doc """
    solve_simple(prob::AnyIncompressibleProblem{Dim, T}; kwargs...) -> SolveResult{Dim, T}

Solve a steady-state incompressible Navier-Stokes problem using the
[`SIMPLE`](@ref) pressure-velocity coupling algorithm.

The algorithm iterates:
1. Assemble and solve momentum equations (one per spatial dimension)
   with under-relaxation.
2. Assemble and solve the pressure Poisson equation.
3. Under-relax the pressure field.
4. Correct cell velocities and face fluxes from the new pressure.
5. Check convergence via momentum and continuity residuals.

# Arguments
- `prob` — [`IncompressibleProblem`](@ref) with a [`SIMPLE`](@ref) algorithm

# Keyword Arguments
- `linear_solver` — solver algorithm for `LinearProblem` (default: `nothing` → backslash)
- `verbose::Bool` — print residuals each iteration (default: `false`)
- `porous_zones` — optional `Vector{PorousZone{T}}` Darcy-Forchheimer
  zones added to the momentum equations (implicit diagonal treatment;
  see [`assemble_momentum!`](@ref))
- `mrf_zones` — optional `Vector{MRFZone{T}}` rotating reference-frame
  zones (absolute-velocity MRF formulation; frame source in momentum,
  relative flux in continuity via [`mrf_make_relative!`](@ref))
- `scheme::ConvectionScheme` — momentum convection scheme
  (default `CONV_UPWIND`; `CONV_LINEAR` / `CONV_BLENDED` reduce the
  first-order smearing at higher Re — see [`assemble_momentum!`](@ref))
- `blend::T` — blending factor for `CONV_BLENDED` (0 = upwind, 1 = central)

# Returns
A [`SolveResult`](@ref) containing convergence status, iteration count,
residual history, and the final [`IncompressibleState`](@ref).
"""
function solve_simple(
        prob::AnyIncompressibleProblem{Dim, T};
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = prob.model.porous_zones,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = prob.model.mrf_zones,
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
    ) where {Dim, T}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh
    max_iter = algo.max_iterations
    tol = algo.tolerance

    # Initialize state and apply boundary conditions
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    # Pre-compute cyclic face pairs (empty vector if no CyclicBC) and the
    # reusable equation workspace with those couplings in the sparsity.
    cyclic_pairs = collect_cyclic_pairs(prob.bcs, mesh)
    eqs, p_eq = _make_incompressible_workspace(prob, cyclic_pairs)

    # Residual history
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    converged = false
    final_iter = 0

    for iter in 1:max_iter
        final_iter = iter

        # Laminar flow: no effective-viscosity or body-force coupling.
        max_residual = _simple_outer_step!(
            state, prob, eqs, p_eq, cyclic_pairs, residuals, component_labels;
            scheme = scheme, blend = blend,
            linear_solver = linear_solver, solver_config = solver_config,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )

        if verbose
            _print_simple_residuals(iter, residuals, component_labels)
        end

        # Never declare convergence on the FIRST outer iteration: the
        # residuals of the startup iterate are degenerate whenever the
        # initial fields solve the momentum equations trivially (e.g. a
        # buoyancy-driven cavity starts with U = 0, uniform T ⇒ zero
        # body force ⇒ exactly zero residuals), and coupled quantities
        # (temperature, turbulence) have not yet fed back into momentum.
        if iter > 1 && max_residual < tol
            converged = true
            break
        end
    end

    return SolveResult{Dim, T}(converged, final_iter, residuals, state)
end

# ── Internal helpers ───────────────────────────────────────────────

"""
    _velocity_labels(::Val{Dim}) -> NTuple{Dim, Symbol}

Return canonical residual label symbols for each velocity component.
"""
_velocity_labels(::Val{2}) = (:Ux, :Uy)
_velocity_labels(::Val{3}) = (:Ux, :Uy, :Uz)

"""
    _solve_linear(lp, linear_solver)

Solve a `LinearProblem`, dispatching to the given solver or the default.
"""
function _solve_linear(lp, linear_solver)
    if linear_solver === nothing
        return solve(lp)
    else
        return solve(lp, linear_solver)
    end
end

"""
    _print_simple_residuals(iter, residuals, labels)

Print a one-line summary of the current iteration's residuals.
"""
function _print_simple_residuals(iter, residuals, labels)
    parts = String[]
    for label in labels
        r = residuals[label][end]
        push!(parts, string(label, "=", @sprintf("%.3e", r)))
    end
    r_cont = residuals[:continuity][end]
    push!(parts, string("cont=", @sprintf("%.3e", r_cont)))
    println("SIMPLE iter ", lpad(iter, 5), ": ", join(parts, "  "))
    return nothing
end

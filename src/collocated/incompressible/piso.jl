# incompressible/piso.jl — PISO transient step function
#
# Implements the Pressure Implicit with Splitting of Operators (PISO)
# algorithm as a single-time-step advancement function.  PISO is a
# non-iterative transient scheme with one momentum predictor followed
# by multiple pressure corrector steps.

# ── PISO step ──────────────────────────────────────────────────────

@doc """
    _piso_step!(state, prob, dt, n_correctors; linear_solver = nothing)

Advance the incompressible solver state by one time step `dt` using the
[`PISO`](@ref) algorithm.

The algorithm proceeds as:
1. **Momentum predictor**: Assemble the transient momentum equations
   (with `dt`) and solve for an intermediate velocity — no under-relaxation.
2. **Pressure corrector loop** (`n_correctors` passes):
   a. Assemble and solve the pressure Poisson equation.
   b. Assign the new pressure to `state.p`.
   c. Correct cell velocities, update boundary conditions, and
      recompute Rhie-Chow face fluxes.
   d. If more corrector steps remain, re-extract the momentum operators
      from a fresh momentum assembly so that the next pressure equation
      uses updated `A_P` and `H(U)`.

# Arguments
- `state::IncompressibleState{Dim, T}` — solver state (modified in-place)
- `prob::AnyIncompressibleProblem{Dim, T}` — problem definition
- `dt::T` — time step size
- `n_correctors::Int` — number of pressure correction steps (typically 2)

# Keyword Arguments
- `linear_solver` — solver algorithm for `LinearProblem` (default: `nothing`)
- `porous_zones` — optional Darcy-Forchheimer zones (see [`assemble_momentum!`](@ref))
- `mrf_zones` — optional MRF zones (see [`assemble_momentum!`](@ref))
"""
function _piso_step!(
        state::IncompressibleState{Dim, T},
        prob::AnyIncompressibleProblem{Dim, T},
        dt::T,
        n_correctors::Int;
        linear_solver = nothing,
        solver_config = nothing,
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}} = Vector{Vector{Tuple{Int, Int}}}(),
        t::T = zero(T),
        ws = nothing,
        nu_eff::Union{T, Vector{T}} = prob.nu,
        body_force::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = prob.model.porous_zones,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = prob.model.mrf_zones,
    ) where {Dim, T}
    mesh = prob.mesh

    # Snapshot the old-time velocity ONCE per time step: every momentum
    # assembly in this step (predictor + corrector re-assemblies) uses it
    # as the ddt old-time value.
    _snapshot_old_time!(state)

    # Equation workspace: reuse caller-provided equations when available.
    eqs, p_eq = ws === nothing ? _make_incompressible_workspace(prob, cyclic_pairs) : ws

    # ── 1. Momentum predictor (no under-relaxation) ─────────────────
    for d in 1:Dim
        reset!(eqs[d])
        assemble_momentum!(
            eqs[d], state, prob, d; dt = dt, t = t,
            nu_eff = nu_eff, body_force = body_force,
            scheme = scheme, blend = blend,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )
        # Apply cyclic coupling to momentum
        apply_cyclic_to_equation!(
            eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
            mesh, cyclic_pairs,
        )
    end

    for d in 1:Dim
        lp = to_linear_problem(eqs[d])
        sol = _dispatch_solve(
            lp, linear_solver, solver_config,
            d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
        )
        _set_component!(state.U, d, sol.u)
    end
    update_boundary_velocity!(state, prob.bcs, mesh; t = t)
    update_boundary_cyclic!(state, mesh, cyclic_pairs)

    # Extract operators AFTER the momentum solve so H(U) uses the solved
    # velocity (standard PISO ordering).
    extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)

    # ── 2. Pressure corrector loop ──────────────────────────────────
    for k in 1:n_correctors
        # 2a. Assemble + solve pressure
        reset!(p_eq)
        assemble_pressure!(p_eq, state, prob; mrf_zones = mrf_zones)
        apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        lp_p = to_linear_problem(p_eq)
        p_sol = _dispatch_solve(lp_p, linear_solver, solver_config, :p)

        # 2b. Direct assign (no under-relaxation in PISO)
        nc = length(mesh.cell_volumes)
        for c in 1:nc
            state.p.internal[c] = p_sol.u[c]
        end

        # 2c. Update boundary pressure
        update_boundary_pressure!(state, prob.bcs, mesh)

        # 2d. Correct velocity + fluxes
        correct_velocity!(state, mesh; porous_zones = porous_zones)
        update_boundary_velocity!(state, prob.bcs, mesh; t = t)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)
        correct_fluxes!(state, mesh; porous_zones = porous_zones)
        if mrf_zones !== nothing
            mrf_make_relative!(state.phi.values, mesh, mrf_zones)
        end

        # 2e. Re-assemble momentum + extract operators for next corrector
        # (ddt still against state.U_old — the time-step snapshot)
        if k < n_correctors
            for d in 1:Dim
                reset!(eqs[d])
                assemble_momentum!(
                    eqs[d], state, prob, d; dt = dt, t = t,
                    nu_eff = nu_eff, body_force = body_force,
                    scheme = scheme, blend = blend,
                    porous_zones = porous_zones, mrf_zones = mrf_zones,
                )
                apply_cyclic_to_equation!(
                    eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
                    mesh, cyclic_pairs,
                )
            end
            extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)
        end
    end

    return nothing
end

"""
    _make_incompressible_workspace(prob, cyclic_pairs) -> (eqs, p_eq)

Allocate the per-component momentum equations and the pressure equation
with the cyclic cross-couplings pre-allocated in the sparsity structure.
Allocate once per solve and pass to the step functions via `ws` to avoid
rebuilding the sparsity pattern every iteration.
"""
function _make_incompressible_workspace(
        prob::AnyIncompressibleProblem{Dim, T},
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}},
    ) where {Dim, T}
    mesh = prob.mesh
    cell_pairs = _cyclic_cell_pairs(mesh, cyclic_pairs)
    eqs = [CollocatedEquation(mesh; extra_cell_pairs = cell_pairs) for _ in 1:Dim]
    p_eq = CollocatedEquation(mesh; extra_cell_pairs = cell_pairs)
    return (eqs, p_eq)
end

# multiphase/solvers.jl — VOF transient solver wrapper
#
# Time-stepping loop: alpha transport → boundedness → mixture update →
# body forces → PISO/PIMPLE momentum+pressure with variable density.

using Printf: @sprintf

"""
    solve_vof(mesh, props, bcs_U, bcs_p, bcs_alpha, tspan, dt; kwargs...)

Solve a transient two-phase VOF flow problem.

Each time step:
1. Solve alpha transport with interface compression
2. Apply boundedness limiter
3. Update mixture properties (ρ, μ)
4. Compute body forces (gravity + surface tension)
5. PISO/PIMPLE step with variable density

# Arguments
- `mesh` — `UnstructuredFVMMesh`
- `props` — `TwoPhaseProperties`
- `bcs_U` — velocity boundary conditions
- `bcs_p` — pressure boundary conditions
- `bcs_alpha` — volume fraction boundary conditions
- `tspan` — `(t_start, t_end)`
- `dt` — time step size

# Keyword Arguments
- `alpha_init` — initial alpha: constant `T` or function `f(x) -> T`
- `g` — gravity vector (default: zero)
- `C_alpha` — compression coefficient (default: 1.0)
- `algorithm` — `PISO()` or `PIMPLE()` (default: `PISO()`)
- `linear_solver` — LinearSolve.jl algorithm
- `save_every` — save interval
- `verbose` — print progress

# Returns
`(SolveResult, VOFState)` tuple.
"""
function solve_vof(
        mesh::UnstructuredFVMMesh{Dim, T},
        props::TwoPhaseProperties{T},
        bcs_U::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_p::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_alpha::Dict{Symbol, <:AbstractBoundaryCondition},
        tspan::Tuple{T, T},
        dt::T;
        alpha_init::Union{T, Function} = zero(T),
        g::SVector{Dim, T} = zero(SVector{Dim, T}),
        C_alpha::T = one(T),
        algorithm::AbstractPVCoupling = PISO(),
        linear_solver = nothing,
        save_every::Int = 1,
        verbose::Bool = false,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    # Create incompressible problem (density=1 placeholder, actual rho handled via body force)
    prob = IncompressibleProblem(mesh, bcs_U, algorithm; nu = T(1.0e-3), density = one(T))

    # Initialize flow state
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, bcs_U, mesh)
    update_boundary_pressure!(state, bcs_p, mesh)

    # Initialize VOF state
    if alpha_init isa Function
        vof_state = VOFState(mesh, alpha_init, props)
        update_mixture_properties!(vof_state, props)
    else
        vof_state = VOFState(mesh; alpha_init = alpha_init)
        update_mixture_properties!(vof_state, props)
    end

    # Residual tracking
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    t_start, t_end = tspan
    t = t_start
    n_steps = 0

    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)

        # -- 1. Alpha transport -------------------------------------------
        alpha_eq = CollocatedEquation(mesh)
        assemble_alpha!(
            alpha_eq, vof_state.alpha, state.phi, mesh, bcs_alpha;
            dt = dt_actual, C_alpha = C_alpha,
        )
        alpha_sol = _solve_linear(to_linear_problem(alpha_eq), linear_solver)
        for c in 1:nc
            vof_state.alpha.internal[c] = alpha_sol.u[c]
        end

        # -- 2. Boundedness limiter ---------------------------------------
        clip_alpha!(vof_state.alpha, mesh)

        # -- 3. Update mixture properties ---------------------------------
        update_mixture_properties!(vof_state, props)

        # -- 4. Body forces (gravity + surface tension) -------------------
        body_force = Vector{SVector{Dim, T}}(undef, nc)
        for c in 1:nc
            body_force[c] = vof_state.rho[c] * g
        end

        # Surface tension
        F_st = compute_surface_tension_force(vof_state.alpha, props, mesh)
        if F_st !== nothing
            for c in 1:nc
                body_force[c] = body_force[c] + F_st[c]
            end
        end

        # -- 5. Kinematic viscosity per cell ------------------------------
        nu_eff = Vector{T}(undef, nc)
        for c in 1:nc
            nu_eff[c] = vof_state.mu[c] / vof_state.rho[c]
        end

        # -- 6. PISO/PIMPLE step with variable density -------------------
        if algorithm isa PISO
            _vof_piso_step!(
                state, prob, dt_actual, algorithm.n_correctors,
                nu_eff, body_force, vof_state.rho;
                linear_solver = linear_solver,
            )
        elseif algorithm isa PIMPLE
            _vof_pimple_step!(
                state, prob, dt_actual,
                nu_eff, body_force, vof_state.rho;
                linear_solver = linear_solver,
            )
        end

        t += dt_actual
        n_steps += 1

        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)

        if verbose && n_steps % max(1, round(Int, (t_end - t_start) / dt / 20)) == 0
            alpha_min = minimum(vof_state.alpha.internal)
            alpha_max = maximum(vof_state.alpha.internal)
            println(
                "Step ", lpad(n_steps, 6),
                "  t=", @sprintf("%.4e", t),
                "  cont=", @sprintf("%.3e", r_cont),
                "  α=[", @sprintf("%.4f", alpha_min), ",", @sprintf("%.4f", alpha_max), "]",
            )
        end
    end

    result = SolveResult{Dim, T}(true, n_steps, residuals, state)
    return (result, vof_state)
end

# -- VOF PISO step (variable density) ------------------------------------

function _vof_piso_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T, n_correctors::Int,
        nu_eff::Vector{T},
        body_force::Vector{SVector{Dim, T}},
        rho::Vector{T};
        linear_solver = nothing,
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Momentum predictor
    eqs = CollocatedEquation{T}[]
    for d in 1:Dim
        eq = CollocatedEquation(mesh)
        assemble_momentum!(
            eq, state, prob, d;
            dt = dt, nu_eff = nu_eff, body_force = body_force
        )
        push!(eqs, eq)
    end

    extract_momentum_operators!(state, eqs, mesh)

    for d in 1:Dim
        sol = _solve_linear(to_linear_problem(eqs[d]), linear_solver)
        _set_component!(state.U, d, sol.u)
    end
    update_boundary_velocity!(state, prob.bcs, mesh)

    # Pressure corrector with density-weighted diffusivity
    for k in 1:n_correctors
        p_eq = CollocatedEquation(mesh)

        # Density-weighted pressure diffusivity: D = V / (rho * A_P)
        D = Vector{T}(undef, nc)
        for c in 1:nc
            D[c] = mesh.cell_volumes[c] / (rho[c] * state.A_P[c])
        end

        bcs_p = expand_bcs_pressure(prob.bcs)
        assemble_laplacian!(p_eq, D, mesh, bcs_p)

        # RHS: divergence of H(U)/A_P flux (same as standard but density-weighted)
        phi_HbyA = compute_HbyA_flux(state, mesh)
        nf = size(mesh.face_cells, 2)
        for f in 1:nf
            P = owner(mesh, f)
            p_eq.b[P] -= phi_HbyA[f]
            if is_internal_face(mesh, f)
                N = neighbour(mesh, f)
                p_eq.b[N] += phi_HbyA[f]
            end
        end

        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end

        p_sol = _solve_linear(to_linear_problem(p_eq), linear_solver)
        for c in 1:nc
            state.p.internal[c] = p_sol.u[c]
        end

        update_boundary_pressure!(state, prob.bcs, mesh)
        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh)

        if k < n_correctors
            eqs_k = CollocatedEquation{T}[]
            for d in 1:Dim
                eq = CollocatedEquation(mesh)
                assemble_momentum!(
                    eq, state, prob, d;
                    dt = dt, nu_eff = nu_eff, body_force = body_force
                )
                push!(eqs_k, eq)
            end
            extract_momentum_operators!(state, eqs_k, mesh)
        end
    end

    return nothing
end

# -- VOF PIMPLE step (variable density) ----------------------------------

function _vof_pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T,
        nu_eff::Vector{T},
        body_force::Vector{SVector{Dim, T}},
        rho::Vector{T};
        linear_solver = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::PIMPLE{T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    for outer in 1:algo.n_outer
        is_final = (outer == algo.n_outer)

        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(
                eq, state, prob, d;
                dt = dt, nu_eff = nu_eff, body_force = body_force
            )
            push!(eqs, eq)
        end
        extract_momentum_operators!(state, eqs, mesh)

        for d in 1:Dim
            if !is_final
                U_old_d = _extract_component(state.U, d)
                under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            end
            sol = _solve_linear(to_linear_problem(eqs[d]), linear_solver)
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)

        for k in 1:algo.n_correctors
            p_eq = CollocatedEquation(mesh)

            D = Vector{T}(undef, nc)
            for c in 1:nc
                D[c] = mesh.cell_volumes[c] / (rho[c] * state.A_P[c])
            end

            bcs_p = expand_bcs_pressure(prob.bcs)
            assemble_laplacian!(p_eq, D, mesh, bcs_p)

            phi_HbyA = compute_HbyA_flux(state, mesh)
            nf = size(mesh.face_cells, 2)
            for f in 1:nf
                P = owner(mesh, f)
                p_eq.b[P] -= phi_HbyA[f]
                if is_internal_face(mesh, f)
                    N = neighbour(mesh, f)
                    p_eq.b[N] += phi_HbyA[f]
                end
            end

            if _needs_pressure_reference(prob.bcs)
                fix_pressure_reference!(p_eq, 1, zero(T))
            end

            p_sol = _solve_linear(to_linear_problem(p_eq), linear_solver)

            if !is_final
                for c in 1:nc
                    state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
                end
            else
                for c in 1:nc
                    state.p.internal[c] = p_sol.u[c]
                end
            end

            update_boundary_pressure!(state, prob.bcs, mesh)
            correct_velocity!(state, mesh)
            update_boundary_velocity!(state, prob.bcs, mesh)
            correct_fluxes!(state, mesh)
        end
    end

    return nothing
end

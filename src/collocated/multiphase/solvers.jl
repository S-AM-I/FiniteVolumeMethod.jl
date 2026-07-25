# multiphase/solvers.jl — VOF transient solver wrapper
#
# Time-stepping loop: alpha transport → boundedness → mixture update →
# body forces → PISO/PIMPLE momentum+pressure with variable density.

using Printf: @sprintf

# Phase-7 add-ons owned by Wave-1 Agent D. Loading via `include` here
# keeps the main layer file (`src/layers/discretization_assembly_kernels.jl`)
# unchanged while ensuring the symbols are available whenever the VOF
# solver is loaded.

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
- `use_mules` — enable MULES flux limiting in α-transport (default: true)
- `use_iso_advector` — use geometric isoAdvector reconstruction for the
  α-flux instead of the upwind/compressive blend (default: false)
- `contact_angle` — optional `AbstractContactAngleModel` for CSF wall
  adhesion (default: `nothing`)
- `wall_patches` — list of face-tag symbols treated as walls for the
  contact-angle correction (default: empty)
- `cavitation_model` — optional `AbstractCavitationVaporModel`
  (`KunzModel`, `SchnerrSauerModel`, `MerkleModel`).  When given (with
  `cavitation_props`), the vapour mass source `ṁ_v` is evaluated once per
  time step from the frozen pressure via `compute_vapor_source` and wired
  into (a) the α-transport equation as `-ṁ_v/ρ_l` (Patankar-implicit
  destruction, explicit condensation; see [`assemble_alpha!`](@ref)) and
  (b) the pressure equation as the volumetric dilatation source
  `ṁ_v (1/ρ_v - 1/ρ_l) V_c`.  CONVENTION: `alpha` is the LIQUID fraction,
  i.e. fluid 1 must be the liquid (`props.rho1 = ρ_l`) and fluid 2 the
  vapour (`props.rho2 = ρ_v`); the vapour fraction passed to the model is
  `1 - α`.  Default `nothing` — behaviour is bitwise identical to the
  cavitation-free solver.
- `cavitation_props` — `CavitationProperties(rho_l, rho_v, p_sat)`;
  required when `cavitation_model` is given.  `rho_l`/`rho_v` should
  match `props.rho1`/`props.rho2`.
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
        solver_config = nothing,
        use_mules::Bool = true,
        use_iso_advector::Bool = false,
        contact_angle::Union{Nothing, AbstractContactAngleModel} = nothing,
        wall_patches::Vector{Symbol} = Symbol[],
        cavitation_model::Union{Nothing, AbstractCavitationVaporModel{T}} = nothing,
        cavitation_props::Union{Nothing, CavitationProperties{T}} = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    if cavitation_model !== nothing && cavitation_props === nothing
        throw(ArgumentError("cavitation_model requires cavitation_props"))
    end

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

        # -- 0. Cavitation vapour mass source (frozen p, once per step) ---
        # Sign convention: positive ⇒ vapour produced ⇒ liquid destroyed.
        mdot_v = nothing
        psi_m = nothing
        if cavitation_model !== nothing
            alpha_v = Vector{T}(undef, nc)
            for c in 1:nc
                alpha_v[c] = one(T) - vof_state.alpha.internal[c]
            end
            mdot_v = compute_vapor_source(
                cavitation_model, state.p.internal, alpha_v, mesh,
                cavitation_props,
            )
            psi_m = _cavitation_dmdot_dp(
                cavitation_model, state.p.internal, alpha_v, mesh,
                cavitation_props,
            )
        end

        # -- 1. Alpha transport -------------------------------------------
        if use_iso_advector
            # Geometric interface reconstruction — bypasses the linear
            # system entirely; α advances by explicit Euler with the
            # reconstructed α-face flux.
            phi_alpha = FaceFluxField(:phi_alpha_iso, mesh; value = zero(T))
            assemble_isoadvector_flux!(
                phi_alpha, vof_state.alpha, state.U, mesh, dt_actual,
            )
            nf = size(mesh.face_cells, 2)
            for f in 1:nf
                F = phi_alpha.values[f] * dt_actual
                P = owner(mesh, f)
                vof_state.alpha.internal[P] -= F / mesh.cell_volumes[P]
                if is_internal_face(mesh, f)
                    N = neighbour(mesh, f)
                    vof_state.alpha.internal[N] += F / mesh.cell_volumes[N]
                end
            end
            # Explicit cavitation source on the geometric path (the
            # implicit Patankar variant lives in the linear-system path);
            # boundedness is restored by clip_alpha! below.
            if mdot_v !== nothing
                for c in 1:nc
                    vof_state.alpha.internal[c] -=
                        dt_actual * mdot_v[c] / cavitation_props.rho_l
                end
            end
        else
            alpha_eq = CollocatedEquation(mesh)
            assemble_alpha!(
                alpha_eq, vof_state.alpha, state.phi, mesh, bcs_alpha;
                dt = dt_actual, C_alpha = C_alpha, use_mules = use_mules,
                mdot_v = mdot_v,
                rho_l = cavitation_props === nothing ? one(T) : cavitation_props.rho_l,
            )
            alpha_sol = _dispatch_solve(to_linear_problem(alpha_eq), linear_solver, solver_config, :alpha)
            for c in 1:nc
                vof_state.alpha.internal[c] = alpha_sol.u[c]
            end
        end

        # -- 2. Boundedness limiter (post-solve safety net) ---------------
        clip_alpha!(vof_state.alpha, mesh)

        # -- 3. Update mixture properties ---------------------------------
        update_mixture_properties!(vof_state, props)

        # -- 4. Body forces (gravity + surface tension) -------------------
        # KINEMATIC form (force per unit mass): the momentum equation uses
        # kinematic convection/diffusion and the -(1/rho) grad(p) pressure
        # source (rho_p), so body forces must be accelerations.  gravity
        # is already an acceleration; the CSF surface-tension force (per
        # unit volume) is divided by the local mixture density.
        body_force = Vector{SVector{Dim, T}}(undef, nc)
        for c in 1:nc
            body_force[c] = g
        end

        # Surface tension (with optional wall-adhesion via contact angle)
        F_st = compute_surface_tension_force(
            vof_state.alpha, props, mesh;
            contact_angle = contact_angle, wall_patches = wall_patches,
        )
        if F_st !== nothing
            for c in 1:nc
                body_force[c] = body_force[c] + F_st[c] / vof_state.rho[c]
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
                linear_solver = linear_solver, solver_config = solver_config,
                mdot_v = mdot_v, psi_m = psi_m,
                cavitation_props = cavitation_props,
            )
        elseif algorithm isa PIMPLE
            _vof_pimple_step!(
                state, prob, dt_actual,
                nu_eff, body_force, vof_state.rho;
                linear_solver = linear_solver, solver_config = solver_config,
                mdot_v = mdot_v, psi_m = psi_m,
                cavitation_props = cavitation_props,
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

    # A transient run "converged" iff it completed with finite residuals
    # (converged used to be hardcoded true, masking NaN/Inf blow-ups).
    r_hist = residuals[:continuity]
    converged = isempty(r_hist) || isfinite(r_hist[end])

    result = SolveResult{Dim, T}(converged, n_steps, residuals, state)
    return (result, vof_state)
end

# -- Cavitation pressure-equation source ----------------------------------

"""
    _add_pressure_mass_transfer!(p_eq, mdot_v, psi_m, p_star, props, mesh)

Add the phase-change dilatation source to the pressure equation with
IMPLICIT pressure linearization (interPhaseChangeFoam `vDotP`-style).
Mixture continuity with mass transfer gives
`div(U) = ṁ_v(p) (1/ρ_v - 1/ρ_l)`; linearizing `ṁ_v(p) ≈ ṁ* + ψ_m (p - p*)`
about the current pressure `p*` and moving the `p`-proportional part to
the LHS yields

```
    -div(D ∇p) - ψ_m Δv V p = -div(φ_HbyA) + (ṁ* - ψ_m p*) Δv V
```

with `Δv = 1/ρ_v - 1/ρ_l`.  All shipped models have `∂ṁ_v/∂p ≤ 0`
(lower pressure ⇒ more vapour), so `-ψ_m Δv V ≥ 0` ADDS to the diagonal
— the Patankar-stable direction.  Without this implicit part the
`p → ṁ_v → div(U) → p` feedback diverges violently for any realistic
transfer coefficient.  `psi_m` entries are clamped to `≤ 0`.
"""
function _add_pressure_mass_transfer!(
        p_eq::CollocatedEquation{T},
        mdot_v::Vector{T},
        psi_m::Vector{T},
        p_star::AbstractVector{T},
        props::CavitationProperties{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    inv_dv = one(T) / props.rho_v - one(T) / props.rho_l
    nc = length(mesh.cell_volumes)
    @inbounds for c in 1:nc
        V_c = mesh.cell_volumes[c]
        psi_c = min(psi_m[c], zero(T))
        add_diag!(p_eq, c, -psi_c * inv_dv * V_c)
        p_eq.b[c] += (mdot_v[c] - psi_c * p_star[c]) * inv_dv * V_c
    end
    return nothing
end

"""
    _cavitation_dmdot_dp(model, p, alpha_v, mesh, props) -> Vector{T}

Central finite-difference `∂ṁ_v/∂p` per cell, used for the implicit
pressure linearization in [`_add_pressure_mass_transfer!`](@ref).  The
step is `δ = max(10⁻³ |p - p_sat|, 10⁻²)` Pa, small enough to resolve
the piecewise-linear Kunz/Merkle branches away from `p_sat` and safe at
the kink (where FD returns the average slope of the two branches).
"""
function _cavitation_dmdot_dp(
        model::AbstractCavitationVaporModel{T},
        p::AbstractVector{T},
        alpha_v::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        props::CavitationProperties{T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    psi_m = Vector{T}(undef, nc)
    @inbounds for c in 1:nc
        delta = max(T(1.0e-3) * abs(p[c] - props.p_sat), T(1.0e-2))
        m_hi = _vapor_source_cell(model, p[c] + delta, alpha_v[c], props)
        m_lo = _vapor_source_cell(model, p[c] - delta, alpha_v[c], props)
        psi_m[c] = (m_hi - m_lo) / (2 * delta)
    end
    return psi_m
end

"""
    _vof_correct_fluxes!(state, rho, mesh)

Density-consistent, pressure-equation-consistent flux update for the
VOF steps (OpenFOAM `phi = phiHbyA - pEqn.flux()` form):

```
    φ_f = interp(H/A_P)·S_f - D_f (p_N - p_P)/|d| |S_f|,
    D_f = harmonic{ V/(ρ A_P) }
```

matching the `D = V/(ρ A_P)` Laplacian of the VOF pressure equation.
The Rhie-Chow deferred-correction form is NOT used here: it subtracts
the linearly-interpolated Green-Gauss cell gradient, which is polluted
at large density ratios where the hydrostatic pressure slope is
discontinuous across the interface (the cell velocities are corrected
with the density-weighted gradient instead — see
`_rho_weighted_pressure_gradient`).  By construction `div(φ)` equals
the pressure-solve residual plus the explicit mass-transfer source.
"""
function _vof_correct_fluxes!(
        state::IncompressibleState{Dim, T},
        rho::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    phi_HbyA = compute_HbyA_flux(state, mesh)

    @inbounds for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            D_P = mesh.cell_volumes[P] / (rho[P] * state.A_P[P])
            D_N = mesh.cell_volumes[N] / (rho[N] * state.A_P[N])
            denom = w * D_N + (one(T) - w) * D_P
            D_f = denom > zero(T) ? D_P * D_N / denom : zero(T)

            _, d_mag = owner_neighbour_distance(mesh, f)
            snGrad = (state.p.internal[N] - state.p.internal[P]) / d_mag
            state.phi.values[f] = phi_HbyA[f] - D_f * snGrad * mesh.face_areas[f]
        else
            state.phi.values[f] = phi_HbyA[f]
        end
    end
    return nothing
end

# -- VOF PISO step (variable density) ------------------------------------

function _vof_piso_step!(
        state::IncompressibleState{Dim, T},
        prob::AnyIncompressibleProblem{Dim, T},
        dt::T, n_correctors::Int,
        nu_eff::Vector{T},
        body_force::Vector{SVector{Dim, T}},
        rho::Vector{T};
        linear_solver = nothing,
        solver_config = nothing,
        mdot_v::Union{Nothing, Vector{T}} = nothing,
        psi_m::Union{Nothing, Vector{T}} = nothing,
        cavitation_props::Union{Nothing, CavitationProperties{T}} = nothing,
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Old-time snapshot for the ddt term (once per time step)
    _snapshot_old_time!(state)

    # Momentum predictor
    eqs = CollocatedEquation{T}[]
    for d in 1:Dim
        eq = CollocatedEquation(mesh)
        assemble_momentum!(
            eq, state, prob, d;
            dt = dt, nu_eff = nu_eff, body_force = body_force,
            rho_p = rho,
        )
        push!(eqs, eq)
    end

    for d in 1:Dim
        sol = _dispatch_solve(
            to_linear_problem(eqs[d]), linear_solver, solver_config,
            d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
        )
        _set_component!(state.U, d, sol.u)
    end
    update_boundary_velocity!(state, prob.bcs, mesh)

    # Extract A_P/H(U) from the solved equations
    extract_momentum_operators!(state, eqs, mesh; rho_p = rho)

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

        # Cavitation dilatation (implicit p-linearization; see helper)
        if mdot_v !== nothing
            _add_pressure_mass_transfer!(
                p_eq, mdot_v, psi_m, state.p.internal, cavitation_props, mesh,
            )
        end

        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end

        p_sol = _dispatch_solve(to_linear_problem(p_eq), linear_solver, solver_config, :p)
        for c in 1:nc
            state.p.internal[c] = p_sol.u[c]
        end

        update_boundary_pressure!(state, prob.bcs, mesh)
        correct_velocity!(state, mesh; rho_p = rho)
        update_boundary_velocity!(state, prob.bcs, mesh)
        _vof_correct_fluxes!(state, rho, mesh)

        if k < n_correctors
            eqs_k = CollocatedEquation{T}[]
            for d in 1:Dim
                eq = CollocatedEquation(mesh)
                assemble_momentum!(
                    eq, state, prob, d;
                    dt = dt, nu_eff = nu_eff, body_force = body_force,
                    rho_p = rho,
                )
                push!(eqs_k, eq)
            end
            extract_momentum_operators!(state, eqs_k, mesh; rho_p = rho)
        end
    end

    return nothing
end

# -- VOF PIMPLE step (variable density) ----------------------------------

function _vof_pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::AnyIncompressibleProblem{Dim, T},
        dt::T,
        nu_eff::Vector{T},
        body_force::Vector{SVector{Dim, T}},
        rho::Vector{T};
        linear_solver = nothing,
        solver_config = nothing,
        mdot_v::Union{Nothing, Vector{T}} = nothing,
        psi_m::Union{Nothing, Vector{T}} = nothing,
        cavitation_props::Union{Nothing, CavitationProperties{T}} = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::PIMPLE{T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Old-time snapshot for the ddt term (shared by all outer iterations)
    _snapshot_old_time!(state)

    for outer in 1:algo.n_outer
        is_final = (outer == algo.n_outer)

        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(
                eq, state, prob, d;
                dt = dt, nu_eff = nu_eff, body_force = body_force,
                rho_p = rho,
            )
            push!(eqs, eq)
        end

        for d in 1:Dim
            if !is_final
                U_old_d = _extract_component(state.U, d)
                under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            end
            sol = _dispatch_solve(
                to_linear_problem(eqs[d]), linear_solver, solver_config,
                d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
            )
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)

        # Extract A_P/H(U) from the (relaxed) solved equations
        extract_momentum_operators!(state, eqs, mesh; rho_p = rho)

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

            # Cavitation dilatation (implicit p-linearization; see helper)
            if mdot_v !== nothing
                _add_pressure_mass_transfer!(
                    p_eq, mdot_v, psi_m, state.p.internal, cavitation_props, mesh,
                )
            end

            if _needs_pressure_reference(prob.bcs)
                fix_pressure_reference!(p_eq, 1, zero(T))
            end

            p_sol = _dispatch_solve(to_linear_problem(p_eq), linear_solver, solver_config, :p)

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
            correct_velocity!(state, mesh; rho_p = rho)
            update_boundary_velocity!(state, prob.bcs, mesh)
            _vof_correct_fluxes!(state, rho, mesh)
        end
    end

    return nothing
end

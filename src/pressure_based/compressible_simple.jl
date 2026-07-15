# pressure_based/compressible_simple.jl — Compressible (subsonic) SIMPLE
#
# Stage 3 pressure-based extension, upgraded to a real compressible
# pressure equation (rhoSimpleFoam-style, subsonic branch):
#
#   * The momentum equations stay in kinematic form but the pressure
#     source is `-(1/ρ_c) ∇p V_c` (`rho_p` kwarg of `assemble_momentum!`),
#     so `state.p` is the ABSOLUTE pressure.
#   * The pressure equation enforces steady MASS continuity
#     `div(ρ U) = 0`: the H/A flux is density-weighted through
#     `compute_face_densities!` + `update_mass_flux!`, and the Laplacian
#     coefficient `ρ_f (V/(ρ A_P))_f ≈ (V/A_P)_f` matches the velocity
#     correction `U = H/A_P - (V/(ρ A_P)) ∇p`.
#   * `update_density!` refreshes ρ = EOS(p, T); `update_viscosity!` for
#     Sutherland / tabulated μ(T).
#
# The transient counterpart (`compressible_pimple.jl`) adds the
# `ddt(ψ p)` compressibility term with ψ = ∂ρ/∂p and a conservative
# linearized density update, which makes total mass exactly telescoping
# (conserved to linear-solver tolerance) in closed domains.
#
# The algorithm type `CompressibleSIMPLE{T}` and its `PIMPLE` counterpart
# live here so the same `solve_compressible(prob, alg)` dispatch works.

using Printf: @sprintf

# ── CompressibleSIMPLE algorithm ────────────────────────────────────

@doc """
    CompressibleSIMPLE{T} <: AbstractPVCoupling

Steady compressible SIMPLE algorithm (subsonic, rhoSimpleFoam-style).
The pressure equation enforces steady MASS continuity `div(ρU) = 0`
with density-weighted H/A mass fluxes (`compute_face_densities!` +
`update_mass_flux!`); the momentum equations use the `-(1/ρ)∇p`
kinematic pressure source so `p` is the absolute pressure driving the
EOS.  Each outer iteration:

1. Update μ(T), ρ_f(p, T) from the current fields.
2. Assemble + solve momentum (under-relaxed, `rho_p = ρ`).
3. Extract A_P, H(U) from the solved, relaxed equations.
4. Assemble + solve the compressible pressure equation
   (`-div((V/A_P)∇p) = -div(ρ_f φ_HbyA)`).
5. Under-relax pressure, update ρ = EOS(p, T) (under-relaxed), correct
   velocity + fluxes, update the mass flux `phi_mass = ρ_f φ`.
6. Check momentum + continuity + density residuals.

The transonic `fvm::div(phid, p)` convective-pressure term is not
implemented — subsonic use only.

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
- `phi_mass::Vector{T}`   — face MASS flux `ρ_f φ_f` (kg/s, length = nfaces),
  maintained via [`update_mass_flux!`](@ref) after every flux correction
"""
mutable struct CompressibleState{Dim, T}
    base::IncompressibleState{Dim, T}
    rho::Vector{T}
    T_cells::Vector{T}
    rho_f::Vector{T}
    mu_cells::Vector{T}
    phi_mass::Vector{T}
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
    phi_mass = zeros(T, nf)
    return CompressibleState{Dim, T}(base, rho, T_cells, rho_f, mu_cells, phi_mass)
end

@doc """
    total_mass(cstate::CompressibleState, mesh) -> T

Total fluid mass `Σ_c ρ_c V_c` — the conserved quantity of the transient
compressible solver in closed domains.
"""
function total_mass(
        cstate::CompressibleState{Dim, T}, mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    m = zero(T)
    @inbounds for c in eachindex(cstate.rho)
        m += cstate.rho[c] * mesh.cell_volumes[c]
    end
    return m
end

# ── Compressible pressure equation ──────────────────────────────────

@doc """
    assemble_pressure_compressible!(
        p_eq, state, prob_shim, rho, rho_f, mesh;
        psi = nothing, rho_old = nothing, dt = nothing,
    )

Assemble the compressible pressure equation.

Steady (`psi === nothing`):
```
    -div((V/A_P) ∇p) = -div(ρ_f φ_HbyA)
```
Transient (ψ, ρ_old, dt given — PIMPLE):
```
    (ψ_c V_c / dt) p + -div((V/A_P) ∇p)
        = (ρ_old,c - ρ_c + ψ_c p_c) V_c / dt - div(ρ_f φ_HbyA)
```
which is the Newton linearization of `ddt(ρ) + div(ρU) = 0` about the
current `(ρ_c, p_c)`; the matching conservative density update is
`ρ ← ρ_c + ψ_c (p_new - p_c)` (see `_conservative_density_update!`).
Because the Laplacian rows sum to zero (Neumann walls) and the mass-flux
divergence telescopes, total mass `Σ ρ V` is conserved to linear-solver
tolerance in closed domains — no pressure reference or mean-anchor is
needed (the ψ diagonal removes the Neumann null space).

The Laplacian coefficient is `D_c = V_c / A_P[c]`: with the kinematic
momentum correction `U = H/A - (V/(ρA))∇p`, the face MASS-flux
correction coefficient is `ρ_f · (V/(ρA))_f ≈ (V/A)_f` (Picard-frozen
ρ), so this is the mass-continuity Laplacian.

Returns the density-weighted H/A mass flux vector (`ρ_f φ_HbyA`,
computed via [`update_mass_flux!`](@ref)) for optional reuse.
"""
function assemble_pressure_compressible!(
        p_eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob_shim::IncompressibleProblem{Dim, T},
        rho::Vector{T},
        rho_f::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        psi::Union{Nothing, Vector{T}} = nothing,
        rho_old::Union{Nothing, Vector{T}} = nothing,
        dt::Union{Nothing, T} = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Mass-continuity Laplacian coefficient D = V / A_P (see docstring).
    D = Vector{T}(undef, nc)
    for c in 1:nc
        D[c] = mesh.cell_volumes[c] / state.A_P[c]
    end
    bcs_p = expand_bcs_pressure(prob_shim.bcs)
    grad_p = gradient(state.p, mesh)
    assemble_laplacian!(
        p_eq, D, mesh, bcs_p;
        non_ortho_correction = true, grad_phi = grad_p,
    )

    # RHS: -div(ρ_f φ_HbyA) — the face densities are actually consumed
    # by the mass balance here (update_mass_flux! wired into the loop).
    phi_HbyA = compute_HbyA_flux(state, mesh)
    m_HbyA = Vector{T}(undef, nf)
    update_mass_flux!(m_HbyA, phi_HbyA, rho_f)
    for f in 1:nf
        P = owner(mesh, f)
        p_eq.b[P] -= m_HbyA[f]
        N = neighbour(mesh, f)
        if N != 0
            p_eq.b[N] += m_HbyA[f]
        end
    end

    # Transient compressibility: ddt(ρ) ≈ (ρ* + ψ(p^{n+1} - p*) - ρ_old)/dt
    if psi !== nothing
        dt === nothing && error("assemble_pressure_compressible!: psi requires dt")
        rho_old === nothing && error("assemble_pressure_compressible!: psi requires rho_old")
        for c in 1:nc
            V_dt = mesh.cell_volumes[c] / dt
            add_diag!(p_eq, c, psi[c] * V_dt)
            p_eq.b[c] += (rho_old[c] - rho[c] + psi[c] * state.p.internal[c]) * V_dt
        end
    end

    return m_HbyA
end

"""
    _correct_fluxes_compressible!(cstate, mesh)

Rhie-Chow flux correction with the compressible face coefficient
`D_f = V/(ρ A_P)` (harmonic, matching the `-(1/ρ)∇p` momentum form),
followed by the mass-flux update `phi_mass = ρ_f φ`.
"""
function _correct_fluxes_compressible!(
        cstate::CompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    state = cstate.base
    A_scaled = state.A_P .* cstate.rho
    rhie_chow_correction!(state.phi, state.U, state.p, A_scaled, mesh)
    update_mass_flux!(cstate.phi_mass, state.phi.values, cstate.rho_f)
    return nothing
end

"""
    _conservative_density_update!(rho, psi, p_new, p_star)

Linearized-EOS density update `ρ ← ρ + ψ (p_new - p*)`, matching the
`ddt(ψ p)` linearization in `assemble_pressure_compressible!` so the
per-step mass balance telescopes exactly.  For an isothermal ideal gas
this coincides with the exact EOS (`ρ = ψ p`).
"""
function _conservative_density_update!(
        rho::Vector{T}, psi::Vector{T},
        p_new::AbstractVector{T}, p_star::AbstractVector{T},
    ) where {T}
    @inbounds for c in eachindex(rho)
        rho[c] += psi[c] * (p_new[c] - p_star[c])
    end
    return nothing
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

Run the compressible SIMPLE loop (steady-state, subsonic). Dispatch on
the `algorithm` field selects SIMPLE vs PIMPLE at construction time;
this method handles [`CompressibleSIMPLE`](@ref). For the transient
[`CompressiblePIMPLE`](@ref), use `solve_compressible(prob, tspan, dt)`.

# Keyword Arguments
- `linear_solver`, `solver_config`, `verbose` — as in `solve_simple`
- `p0` — initial absolute pressure level (Pa); for closed (all-Neumann)
  domains the converged mean pressure is anchored to the initial mean
"""
function solve_compressible(
        prob::CompressibleProblem{Dim, T, Mesh, BC, CompressibleSIMPLE{T}, Model};
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        p0::Real = 1.01325e5,
    ) where {Dim, T, Mesh, BC, Model}
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

    # Equation workspace: allocate once, reset! + reassemble per iteration.
    cell_pairs = _cyclic_cell_pairs(mesh, cyclic_pairs)
    eqs = [CollocatedEquation(mesh; extra_cell_pairs = cell_pairs) for _ in 1:Dim]
    p_eq = CollocatedEquation(mesh; extra_cell_pairs = cell_pairs)

    for iter in 1:max_iter
        final_iter = iter

        # ── 1. Update μ(T) and ρ_f(p, T) ────────────────────────────
        update_viscosity!(cstate.mu_cells, prob.thermo, cstate.T_cells)
        mu_mean = sum(cstate.mu_cells) / nc
        rho_mean = sum(cstate.rho) / nc
        compute_face_densities!(
            cstate.rho_f, prob.thermo,
            mesh, state.p.internal, cstate.T_cells
        )

        # ── 2. Momentum solve (kinematic, -(1/ρ)∇p pressure source) ─
        shim = _incompressible_shim(prob, rho_mean, mu_mean)
        for d in 1:Dim
            reset!(eqs[d])
            assemble_momentum!(
                eqs[d], state, shim, d;
                nu_eff = cstate.mu_cells ./ cstate.rho,
                rho_p = cstate.rho,
            )
            apply_cyclic_to_equation!(
                eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
                mesh, cyclic_pairs,
            )
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
        extract_momentum_operators!(state, eqs, mesh; rho_p = cstate.rho)

        # ── 3. Compressible pressure solve: div(ρ_f φ) = 0 ──────────
        # For a closed (Neumann-only) steady system the absolute
        # pressure level is anchored by the imposed `p0` — we solve for
        # the pressure, then shift the result so the mean is preserved.
        # Pinning cell-1 to zero as in the incompressible case would
        # erase the physical absolute pressure (ρ = p/(R T) needs it).
        needs_ref = _needs_pressure_reference(prob.bcs)
        p_mean_target = needs_ref ? sum(state.p.internal) / nc : zero(T)

        reset!(p_eq)
        assemble_pressure_compressible!(
            p_eq, state, shim, cstate.rho, cstate.rho_f, mesh,
        )
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

        # ── 4. Correct velocity + fluxes (compressible D = V/(ρA)) ──
        correct_velocity!(state, mesh; rho_p = cstate.rho)
        update_boundary_velocity!(state, prob.bcs, mesh)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)
        _correct_fluxes_compressible!(cstate, mesh)

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

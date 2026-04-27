# ============================================================
# 1D Navier-Stokes Solver
# ============================================================
#
# Overrides compute_dt and hyperbolic_rhs! for problems with
# NavierStokesEquations{1}. The existing solve_hyperbolic time
# loop calls these via dispatch -- no changes needed there.

"""
    compute_dt(prob::HyperbolicProblem{<:NavierStokesEquations{1}}, U, t) -> dt

Compute the time step accounting for both hyperbolic and viscous stability:
- `dt_hyp = CFL * dx / max|lambda|`
- `dt_visc = 0.5 * rho_min * dx^2 / mu`
- `dt = CFL * min(dt_hyp, dt_visc)`
"""
function compute_dt(prob::HyperbolicProblem{<:NavierStokesEquations{1}}, U::AbstractVector, t)
    law = prob.law
    mesh = prob.mesh
    nc = ncells(mesh)
    cfl = prob.cfl
    dx = cell_volume(mesh, 1)
    mu = law.mu
    ng = _nghost_for_reconstruction(prob.reconstruction)

    lambda_max = zero(dx)
    rho_min = typeof(dx)(Inf)
    for i in 1:nc
        w = conserved_to_primitive(law, U[i + ng])
        lambda = max_wave_speed(law, w, 1)
        lambda_max = max(lambda_max, lambda)
        rho_min = min(rho_min, w[1])
    end

    dt_hyp = dx / lambda_max

    dt = dt_hyp
    if mu > 0
        dt_visc = 0.5 * rho_min * dx^2 / mu
        dt = min(dt_hyp, dt_visc)
    end

    dt = cfl * dt

    # Don't overshoot final time
    if t + dt > prob.final_time
        dt = prob.final_time - t
    end

    return dt
end

"""
    hyperbolic_rhs!(dU, U, prob::HyperbolicProblem{<:NavierStokesEquations{1}}, t)

Compute the RHS including both inviscid Riemann fluxes and viscous fluxes:
  `dU[i]/dt = -1/dx * (F_{i+1/2} - F_{i-1/2}) + 1/dx * (Fv_{i+1/2} - Fv_{i-1/2})`
"""
function hyperbolic_rhs!(
        dU::AbstractVector, U::AbstractVector,
        prob::HyperbolicProblem{<:NavierStokesEquations{1}}, t
    )
    law = prob.law
    mesh = prob.mesh
    nc = ncells(mesh)
    solver = prob.riemann_solver
    recon = prob.reconstruction
    dx = cell_volume(mesh, 1)
    ng = _nghost_for_reconstruction(recon)

    # Apply BCs to fill ghost cells
    apply_boundary_conditions!(U, prob, ng, t)

    # Update each interior cell
    for i in 1:nc
        # Left face (face i-1): between cell i-1 and cell i
        wL_left, wR_left = _reconstruct_face(recon, law, U, i - 1, nc)
        F_left = solve_riemann(solver, law, wL_left, wR_left, 1)

        # Right face (face i): between cell i and cell i+1
        wL_right, wR_right = _reconstruct_face(recon, law, U, i, nc)
        F_right = solve_riemann(solver, law, wL_right, wR_right, 1)

        # Inviscid contribution
        dU[i + ng] = -(F_right - F_left) / dx

        # Viscous contribution
        # Left face: primitive states of cells i-1 and i (padded: i+ng-1, i+ng)
        wL_visc_left = conserved_to_primitive(law, U[i + ng - 1])
        wR_visc_left = conserved_to_primitive(law, U[i + ng])
        Fv_left = viscous_flux_1d(law, wL_visc_left, wR_visc_left, dx)

        # Right face: primitive states of cells i and i+1 (padded: i+ng, i+ng+1)
        wL_visc_right = conserved_to_primitive(law, U[i + ng])
        wR_visc_right = conserved_to_primitive(law, U[i + ng + 1])
        Fv_right = viscous_flux_1d(law, wL_visc_right, wR_visc_right, dx)

        dU[i + ng] = dU[i + ng] + (Fv_right - Fv_left) / dx
    end

    return nothing
end

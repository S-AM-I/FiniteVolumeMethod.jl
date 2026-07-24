# dynamic_mesh/ale.jl — ALE flux correction and transient ALE solver
#
# Provides the ALE-corrected face flux (phi - phi_mesh) and a
# `solve_ale` wrapper that combines mesh motion with incompressible
# PISO/PIMPLE time stepping.

using Printf: @sprintf

@doc """
    ale_corrected_flux(phi::FaceFluxField{T}, phi_mesh::Vector{T}) -> FaceFluxField{T}

Compute the ALE-corrected face flux by subtracting the mesh sweep flux
from the fluid face flux: `phi_ale = phi - phi_mesh`.

The corrected flux is used in all convective transport operators so that
the face velocity is measured relative to the moving mesh.

# Arguments
- `phi` — fluid volumetric face flux
- `phi_mesh` — mesh sweep flux (from [`compute_mesh_flux!`](@ref))

# Returns
A new [`FaceFluxField`](@ref) with corrected values.
"""
function ale_corrected_flux(phi::FaceFluxField{T}, phi_mesh::Vector{T}) where {T}
    nf = length(phi.values)
    length(phi_mesh) == nf || error(
        "phi_mesh length ($(length(phi_mesh))) must match phi length ($nf)",
    )
    corrected = similar(phi.values)
    for f in 1:nf
        corrected[f] = phi.values[f] - phi_mesh[f]
    end
    return FaceFluxField{T}(:phi_ale, corrected)
end

# ── ALE transient solver ────────────────────────────────────────────

@doc """
    solve_ale(
        mesh, motion_solver, bcs_U, bcs_p, tspan, dt;
        bcs_displacement = nothing,
        nu = 1.0e-3,
        algorithm = PISO(),
        linear_solver = nothing,
        solver_config = nothing,
        verbose = false,
    )

Solve a transient incompressible Navier-Stokes problem on a moving
mesh using the ALE (Arbitrary Lagrangian-Eulerian) approach.

Each time step proceeds as:
1. Compute displacement via the motion solver.
2. Store old volumes, update mesh geometry, compute `phi_mesh`.
3. Compute ALE-corrected flux.
4. Advance one PISO/PIMPLE step with the corrected flux.

# Arguments
- `mesh` — `UnstructuredFVMMesh{Dim, T}` (modified in-place by mesh motion)
- `motion_solver` — [`SolidBodyMotion`](@ref) or [`LaplacianMotion`](@ref)
- `bcs_U` — velocity boundary conditions `Dict{Symbol, <:AbstractBoundaryCondition}`
- `bcs_p` — pressure boundary conditions `Dict{Symbol, <:AbstractBoundaryCondition}`
- `tspan` — `(t_start, t_end)`
- `dt` — time step size

# Keyword Arguments
- `bcs_displacement` — boundary conditions for displacement (required for
  [`LaplacianMotion`](@ref), ignored for [`SolidBodyMotion`](@ref))
- `nu` — kinematic viscosity (default `1e-3`)
- `algorithm` — pressure-velocity coupling: [`PISO`](@ref) or [`PIMPLE`](@ref)
- `linear_solver` — solver algorithm for `LinearProblem`
- `solver_config` — [`FVMSolverConfig`](@ref)
- `verbose` — print progress each time step

# Returns
A [`SolveResult`](@ref) with the final state and residual history.
"""
function solve_ale(
        mesh::UnstructuredFVMMesh{Dim, T},
        motion_solver::AbstractMotionSolver,
        bcs_U::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_p::Dict{Symbol, <:AbstractBoundaryCondition},
        tspan::Tuple{T, T},
        dt::T;
        bcs_displacement::Union{Nothing, Dict{Symbol, <:AbstractBoundaryCondition}} = nothing,
        nu::T = T(1.0e-3),
        algorithm::AbstractPVCoupling = PISO(),
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    t_start, t_end = tspan

    # Build merged BC dict for the incompressible solver
    bcs = _merge_ale_bcs(bcs_U, bcs_p)

    # Build incompressible problem
    prob = IncompressibleProblem(mesh, bcs, algorithm; nu = nu)

    # Initialize state
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, bcs, mesh)
    update_boundary_pressure!(state, bcs, mesh)

    # Initialize motion state
    motion_state = MeshMotionState(mesh)

    # Select step function
    step_fn! = _select_step_function(algorithm)

    # Residual tracking
    component_labels = Dim == 2 ? (:Ux, :Uy) : (:Ux, :Uy, :Uz)
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    # Time-stepping loop
    t = t_start
    n_steps = 0

    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)
        t += dt_actual
        n_steps += 1

        # 1. Compute displacement
        _compute_ale_displacement!(
            motion_state, motion_solver, mesh, t;
            bcs_displacement = bcs_displacement,
            linear_solver = linear_solver,
            solver_config = solver_config,
        )

        # 2. Update mesh geometry + compute phi_mesh
        update_mesh!(mesh, motion_state, dt_actual)

        # 3. ALE-corrected flux
        phi_ale = ale_corrected_flux(state.phi, motion_state.phi_mesh)
        # Store corrected flux back into state for the PISO/PIMPLE step
        copyto!(state.phi.values, phi_ale.values)

        # 4. PISO/PIMPLE step
        step_fn!(
            state, prob, dt_actual;
            linear_solver = linear_solver,
            solver_config = solver_config,
        )

        # Record residuals
        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)

        if verbose
            println(
                "ALE step ", lpad(n_steps, 6),
                "  t=", @sprintf("%.4e", t),
                "  cont=", @sprintf("%.3e", r_cont),
            )
        end
    end

    # A transient run "converged" iff it completed with finite residuals
    # (converged used to be hardcoded true, masking NaN/Inf blow-ups).
    r_hist = residuals[:continuity]
    converged = isempty(r_hist) || isfinite(r_hist[end])

    return SolveResult{Dim, T}(converged, n_steps, residuals, state)
end

# ── Internal helpers ────────────────────────────────────────────────

"""Dispatch displacement computation based on motion solver type."""
function _compute_ale_displacement!(
        motion_state, solver::SolidBodyMotion, mesh, t;
        bcs_displacement = nothing,
        linear_solver = nothing,
        solver_config = nothing,
    )
    compute_displacement!(motion_state, solver, mesh, t)
    return nothing
end

function _compute_ale_displacement!(
        motion_state, solver::LaplacianMotion, mesh, t;
        bcs_displacement = nothing,
        linear_solver = nothing,
        solver_config = nothing,
    )
    bcs_displacement === nothing && error(
        "bcs_displacement is required for LaplacianMotion",
    )
    compute_displacement!(
        motion_state, solver, mesh, bcs_displacement, t;
        linear_solver = linear_solver,
        solver_config = solver_config,
    )
    return nothing
end

"""
Merge velocity and pressure BCs into a single dict for IncompressibleProblem.

Velocity BCs are wrapped as FixedVelocityBC/NoSlipWallBC and pressure BCs
as FixedPressureBC.  For simplicity we pass through the raw BCs and rely
on the incompressible solver's BC expansion.
"""
function _merge_ale_bcs(
        bcs_U::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_p::Dict{Symbol, <:AbstractBoundaryCondition},
    )
    merged = Dict{Symbol, AbstractBoundaryCondition}()
    # Velocity BCs take precedence (they encode both U and p behavior)
    for (name, bc) in bcs_U
        merged[name] = bc
    end
    # Only add pressure BCs for patches not already covered
    for (name, bc) in bcs_p
        if !haskey(merged, name)
            merged[name] = bc
        end
    end
    return merged
end

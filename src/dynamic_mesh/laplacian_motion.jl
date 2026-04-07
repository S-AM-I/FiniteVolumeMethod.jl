# dynamic_mesh/laplacian_motion.jl — Diffusion-based mesh displacement
#
# Solves `div(gamma * grad(d_component)) = 0` per spatial dimension using
# the Phase 0 Laplacian assembly.  Boundary conditions prescribe
# displacement on moving boundaries (ParabolicDirichlet(value)) and
# zero displacement on fixed boundaries (ParabolicDirichlet(0)).

@doc """
    compute_displacement!(
        motion_state::MeshMotionState{Dim, T},
        solver::LaplacianMotion{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_displacement::Dict{Symbol, <:AbstractBoundaryCondition},
        t;
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}

Compute per-cell displacement for [`LaplacianMotion`](@ref) by solving
a Laplace equation per spatial dimension.

For each dimension `d`, the method assembles `div(gamma * grad(d_d)) = 0`
with boundary conditions from `bcs_displacement`, then solves the resulting
linear system.  Moving boundaries should prescribe
`ParabolicDirichlet(displacement_value)` and fixed boundaries should use
`ParabolicDirichlet(0)`.

# Arguments
- `motion_state` — mutable motion state (modified in-place)
- `solver` — Laplacian motion solver with diffusivity `gamma`
- `mesh` — the FVM mesh
- `bcs_displacement` — boundary conditions for displacement
  (keyed by patch name, one `Dict` used for all dimensions)
- `t` — current simulation time (unused, reserved for time-dependent BCs)

# Keyword Arguments
- `linear_solver` — algorithm for `LinearProblem` (default: `nothing`)
- `solver_config` — [`FVMSolverConfig`](@ref) for per-field solver selection
"""
function compute_displacement!(
        motion_state::MeshMotionState{Dim, T},
        solver::LaplacianMotion{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_displacement::Dict{Symbol, <:AbstractBoundaryCondition},
        t;
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    gamma = solver.gamma

    for d in 1:Dim
        # Assemble Laplace equation for dimension d
        eq = CollocatedEquation(mesh)
        assemble_laplacian!(eq, gamma, mesh, bcs_displacement)

        # Solve
        lp = to_linear_problem(eq)
        field_name = d == 1 ? :dx : (d == 2 ? :dy : :dz)
        sol = _dispatch_solve(lp, linear_solver, solver_config, field_name)

        # Store displacement component
        for c in 1:nc
            old = motion_state.displacement[c]
            motion_state.displacement[c] = Base.setindex(old, sol.u[c], d)
        end
    end

    return nothing
end

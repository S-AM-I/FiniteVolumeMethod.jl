# dynamic_mesh/laplacian_motion.jl — Diffusion-based mesh displacement
#
# Solves `div(gamma * grad(d_component)) = 0` per spatial dimension using
# the Phase 0 Laplacian assembly.  Boundary conditions prescribe
# displacement on moving boundaries (DirichletBC(value)) and
# zero displacement on fixed boundaries (DirichletBC(0)).

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
`DirichletBC(displacement_value)` and fixed boundaries should use
`DirichletBC(0)`.

# Arguments
- `motion_state` — mutable motion state (modified in-place)
- `solver` — Laplacian motion solver with diffusivity `gamma`
- `mesh` — the FVM mesh
- `bcs_displacement` — boundary conditions for displacement
  (keyed by patch name, one `Dict` used for all dimensions)
- `t` — current simulation time (unused, reserved for time-dependent BCs)

# Keyword Arguments
- `linear_solver` — algorithm for `LinearProblem` (default: `nothing`)
- `solver_config` — `FVMSolverConfig` for per-field solver selection
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

    # If gamma is scalar, optionally compute inverse-distance diffusivity
    # per cell for better quality near boundaries
    gamma_eff = gamma

    for d in 1:Dim
        # Assemble Laplace equation for dimension d
        eq = CollocatedEquation(mesh)
        assemble_laplacian!(eq, gamma_eff, mesh, bcs_displacement)

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

"""
    compute_distance_diffusivity(
        mesh, wall_patches; gamma_ref = 1.0, power = 2,
    ) -> Vector{T}

Compute a per-cell diffusivity based on inverse distance to wall boundaries.
Cells near walls get high diffusivity (preserving mesh quality near walls),
cells far from walls get low diffusivity (allowing more deformation).

    gamma[c] = gamma_ref / d_wall[c]^power

Useful as input to `assemble_laplacian!` for the Laplacian motion solver.

# Arguments
- `mesh` — `UnstructuredFVMMesh`
- `wall_patches` — list of wall/fixed boundary patch names
- `gamma_ref` — reference diffusivity scale (default 1.0)
- `power` — distance exponent (default 2 for 1/d²)
"""
function compute_distance_diffusivity(
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol};
        gamma_ref::T = one(T),
        power::Int = 2,
    ) where {Dim, T}
    d_wall = compute_wall_distance(mesh, wall_patches)
    nc = length(d_wall)
    gamma = Vector{T}(undef, nc)
    for c in 1:nc
        gamma[c] = gamma_ref / max(d_wall[c], T(1.0e-20))^power
    end
    return gamma
end

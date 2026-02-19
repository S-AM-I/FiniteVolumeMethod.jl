# ============================================================
# Dashboard Callbacks for Hyperbolic (Cell-Centered) Solvers
# ============================================================
#
# Creates callback functions that record FVMSnapshots during
# manual time-stepping loops (solve_hyperbolic, etc.).

"""
    hyperbolic_monitor(; interval=1, session_data, law, mesh)

Create a callback function for injection into `solve_hyperbolic` and related
cell-centered solvers. The callback is invoked as `cb(U, t, step, dt)` after
each time step.

# Keyword Arguments
- `interval::Int`: Record a snapshot every `interval` steps (default: every step).
- `session_data::FVMSessionData`: Session to append snapshots to.
- `law`: The conservation law (for computing conserved totals and variable names).
- `mesh`: The mesh (for computing conserved totals via cell volumes).

# Returns
A callable `(U, t, step, dt) -> nothing` suitable for the `callback` keyword
argument of `solve_hyperbolic`.
"""
function hyperbolic_monitor(;
        interval::Int = 1,
        session_data::FVMSessionData,
        law,
        mesh,
    )
    t_start = time()
    return function (U, t, step, dt)
        if step % interval != 0
            return nothing
        end
        wall = time() - t_start
        # Extract interior solution for totals computation
        U_interior = _extract_interior(U, mesh)
        totals = conserved_totals(law, U_interior, mesh)
        snap = FVMSnapshot(
            t, step, U_interior, 0.0, totals, dt, wall,
        )
        push!(session_data.snapshots, snap)
        return nothing
    end
end

# Extract interior cells from padded arrays
function _extract_interior(U::AbstractVector{<:SVector}, mesh::StructuredMesh1D)
    nc = ncells(mesh)
    return U[3:(nc + 2)]
end

function _extract_interior(U::AbstractMatrix{<:SVector}, mesh::StructuredMesh2D)
    nx, ny = mesh.nx, mesh.ny
    return U[3:(nx + 2), 3:(ny + 2)]
end

function _extract_interior(U::AbstractArray{<:SVector, 3}, mesh::StructuredMesh3D)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    return U[3:(nx + 2), 3:(ny + 2), 3:(nz + 2)]
end

# Unstructured meshes have no ghost cells in the solution vector
function _extract_interior(U::AbstractVector{<:SVector}, mesh::UnstructuredHyperbolicMesh)
    return U
end

"""
    create_session_data(prob) -> FVMSessionData

Convenience constructor: populate an `FVMSessionData` from a hyperbolic problem.
"""
function create_session_data(prob::HyperbolicProblem)
    return FVMSessionData(;
        problem_type = "HyperbolicProblem",
        law_name = string(typeof(prob.law)),
        mesh_info = mesh_to_dict(prob.mesh),
        variable_names = variable_names(prob.law),
        parameters = Dict{String, Any}(
            "cfl" => prob.cfl,
            "solver" => string(typeof(prob.riemann_solver)),
            "reconstruction" => string(typeof(prob.reconstruction)),
        ),
    )
end

function create_session_data(prob::HyperbolicProblem2D)
    return FVMSessionData(;
        problem_type = "HyperbolicProblem2D",
        law_name = string(typeof(prob.law)),
        mesh_info = mesh_to_dict(prob.mesh),
        variable_names = variable_names(prob.law),
        parameters = Dict{String, Any}(
            "cfl" => prob.cfl,
            "solver" => string(typeof(prob.riemann_solver)),
            "reconstruction" => string(typeof(prob.reconstruction)),
        ),
    )
end

function create_session_data(prob::HyperbolicProblem3D)
    return FVMSessionData(;
        problem_type = "HyperbolicProblem3D",
        law_name = string(typeof(prob.law)),
        mesh_info = mesh_to_dict(prob.mesh),
        variable_names = variable_names(prob.law),
        parameters = Dict{String, Any}(
            "cfl" => prob.cfl,
            "solver" => string(typeof(prob.riemann_solver)),
            "reconstruction" => string(typeof(prob.reconstruction)),
        ),
    )
end

function create_session_data(prob::UnstructuredHyperbolicProblem)
    return FVMSessionData(;
        problem_type = "UnstructuredHyperbolicProblem",
        law_name = string(typeof(prob.law)),
        mesh_info = mesh_to_dict(prob.mesh),
        variable_names = variable_names(prob.law),
        parameters = Dict{String, Any}(
            "cfl" => prob.cfl,
            "solver" => string(typeof(prob.riemann_solver)),
            "reconstruction" => string(typeof(prob.reconstruction)),
        ),
    )
end

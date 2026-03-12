# ============================================================
# Dashboard Callbacks for Hyperbolic (Cell-Centered) Solvers
# ============================================================

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
function FiniteVolumeMethod.hyperbolic_monitor(;
        interval::Int = 1,
        session_data::FiniteVolumeMethod.FVMSessionData,
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
        totals = FiniteVolumeMethod.conserved_totals(law, U_interior, mesh)
        snap = FiniteVolumeMethod.FVMSnapshot(
            t, step, U_interior, 0.0, totals, dt, wall,
        )
        push!(session_data.snapshots, snap)
        return nothing
    end
end

# Extract interior cells from padded arrays
function _extract_interior(U::AbstractVector{<:SVector}, mesh::FiniteVolumeMethod.StructuredMesh1D)
    nc = FiniteVolumeMethod.ncells(mesh)
    return U[3:(nc + 2)]
end

function _extract_interior(U::AbstractMatrix{<:SVector}, mesh::FiniteVolumeMethod.StructuredMesh2D)
    nx, ny = mesh.nx, mesh.ny
    return U[3:(nx + 2), 3:(ny + 2)]
end

function _extract_interior(U::AbstractArray{<:SVector, 3}, mesh::FiniteVolumeMethod.StructuredMesh3D)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    return U[3:(nx + 2), 3:(ny + 2), 3:(nz + 2)]
end

# Unstructured meshes have no ghost cells in the solution vector
function _extract_interior(U::AbstractVector{<:SVector}, mesh::FiniteVolumeMethod.UnstructuredHyperbolicMesh)
    return U
end

# ============================================================
# Session Data Constructors
# ============================================================

"""
    create_session_data(prob) -> FVMSessionData

Convenience constructor: populate an `FVMSessionData` from a hyperbolic problem.
"""
function FiniteVolumeMethod.create_session_data(prob::FiniteVolumeMethod.HyperbolicProblem)
    return FiniteVolumeMethod.FVMSessionData(;
        problem_type = "HyperbolicProblem",
        law_name = string(typeof(prob.law)),
        mesh_info = FiniteVolumeMethod.mesh_to_dict(prob.mesh),
        variable_names = FiniteVolumeMethod.variable_names(prob.law),
        parameters = Dict{String, Any}(
            "cfl" => prob.cfl,
            "solver" => string(typeof(prob.riemann_solver)),
            "reconstruction" => string(typeof(prob.reconstruction)),
        ),
    )
end

function FiniteVolumeMethod.create_session_data(prob::FiniteVolumeMethod.HyperbolicProblem2D)
    return FiniteVolumeMethod.FVMSessionData(;
        problem_type = "HyperbolicProblem2D",
        law_name = string(typeof(prob.law)),
        mesh_info = FiniteVolumeMethod.mesh_to_dict(prob.mesh),
        variable_names = FiniteVolumeMethod.variable_names(prob.law),
        parameters = Dict{String, Any}(
            "cfl" => prob.cfl,
            "solver" => string(typeof(prob.riemann_solver)),
            "reconstruction" => string(typeof(prob.reconstruction)),
        ),
    )
end

function FiniteVolumeMethod.create_session_data(prob::FiniteVolumeMethod.HyperbolicProblem3D)
    return FiniteVolumeMethod.FVMSessionData(;
        problem_type = "HyperbolicProblem3D",
        law_name = string(typeof(prob.law)),
        mesh_info = FiniteVolumeMethod.mesh_to_dict(prob.mesh),
        variable_names = FiniteVolumeMethod.variable_names(prob.law),
        parameters = Dict{String, Any}(
            "cfl" => prob.cfl,
            "solver" => string(typeof(prob.riemann_solver)),
            "reconstruction" => string(typeof(prob.reconstruction)),
        ),
    )
end

function FiniteVolumeMethod.create_session_data(prob::FiniteVolumeMethod.UnstructuredHyperbolicProblem)
    return FiniteVolumeMethod.FVMSessionData(;
        problem_type = "UnstructuredHyperbolicProblem",
        law_name = string(typeof(prob.law)),
        mesh_info = FiniteVolumeMethod.mesh_to_dict(prob.mesh),
        variable_names = FiniteVolumeMethod.variable_names(prob.law),
        parameters = Dict{String, Any}(
            "cfl" => prob.cfl,
            "solver" => string(typeof(prob.riemann_solver)),
            "reconstruction" => string(typeof(prob.reconstruction)),
        ),
    )
end

# ============================================================
# Dashboard Callbacks for Parabolic (Vertex-Centered) Solvers
# ============================================================

"""
    FVMMonitorCallback(; interval=1, session_data)

Create a `DiscreteCallback` that records snapshots of the parabolic solver state
into `session_data` every `interval` accepted time steps.

# Keyword Arguments
- `interval::Int`: Record every `interval`-th step (default: every step).
- `session_data::FVMSessionData`: Session to append snapshots to.

# Returns
A `DiscreteCallback` compatible with DifferentialEquations.jl solvers.

# Example
```julia
session = FVMSessionData(problem_type="FVMProblem", ...)
cb = FVMMonitorCallback(; interval=10, session_data=session)
sol = solve(prob, Tsit5(); callback=cb)
```
"""
function FiniteVolumeMethod.FVMMonitorCallback(;
        interval::Int = 1,
        session_data::FiniteVolumeMethod.FVMSessionData,
    )
    t_start = time()
    step_counter = Ref(0)

    condition = function (u, t, integrator)
        step_counter[] += 1
        return step_counter[] % interval == 0
    end

    affect! = function (integrator)
        wall = time() - t_start
        u = integrator.u
        t_val = integrator.t
        dt_val = integrator.dt
        totals = Dict{String, Float64}("total" => sum(u))
        snap = FiniteVolumeMethod.FVMSnapshot(
            Float64(t_val), step_counter[], copy(u), 0.0, totals, Float64(dt_val), wall,
        )
        push!(session_data.snapshots, snap)
        return nothing
    end

    return DiscreteCallback(condition, affect!; save_positions = (false, false))
end

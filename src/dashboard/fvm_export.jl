# ============================================================
# Dashboard Data Export — Core Types and Serialization
# ============================================================
#
# Provides snapshot and session types for exporting simulation
# data to the dashboard (JSON file or WebSocket).

"""
    FVMSnapshot{T}

A single time-step snapshot of the simulation state.

# Fields
- `time::Float64`: Current simulation time.
- `step::Int`: Time-step index.
- `solution::T`: Solution data (Vector for 1D, Matrix for 2D, etc.).
- `residual_norm::Float64`: L2 norm of the residual (dU).
- `conserved_totals::Dict{String,Float64}`: Integrated conserved quantities (mass, momentum, energy).
- `dt::Float64`: Time-step size used.
- `wall_time::Float64`: Wall-clock time elapsed since solver start.
"""
struct FVMSnapshot{T}
    time::Float64
    step::Int
    solution::T
    residual_norm::Float64
    conserved_totals::Dict{String, Float64}
    dt::Float64
    wall_time::Float64
end

"""
    FVMSessionData

Metadata and accumulated snapshots for a simulation session.

# Fields
- `problem_type::String`: E.g. `"HyperbolicProblem2D"`, `"FVMProblem"`.
- `law_name::String`: E.g. `"EulerEquations{2}"`, `"DiffusionEquation"`.
- `mesh_info::Dict{String,Any}`: Cell count, domain bounds, mesh type.
- `variable_names::Vector{String}`: E.g. `["rho", "vx", "vy", "P"]`.
- `parameters::Dict{String,Any}`: CFL, solver, reconstruction, etc.
- `snapshots::Vector{FVMSnapshot}`: Accumulated snapshots.
- `convergence_data::Vector{Dict{String,Any}}`: Optional convergence study results.
"""
mutable struct FVMSessionData
    problem_type::String
    law_name::String
    mesh_info::Dict{String, Any}
    variable_names::Vector{String}
    parameters::Dict{String, Any}
    snapshots::Vector{FVMSnapshot}
    convergence_data::Vector{Dict{String, Any}}
end

"""
    FVMSessionData(; problem_type, law_name, mesh_info, variable_names, parameters)

Construct an empty session (no snapshots yet).
"""
function FVMSessionData(;
        problem_type::String = "",
        law_name::String = "",
        mesh_info::Dict{String, Any} = Dict{String, Any}(),
        variable_names::Vector{String} = String[],
        parameters::Dict{String, Any} = Dict{String, Any}(),
    )
    return FVMSessionData(
        problem_type, law_name, mesh_info, variable_names, parameters,
        FVMSnapshot[], Dict{String, Any}[],
    )
end

# ============================================================
# Variable names for each conservation law
# ============================================================

"""
    variable_names(law) -> Vector{String}

Return human-readable names for the conserved variables of `law`.
"""
function variable_names end

variable_names(::EulerEquations{1}) = ["rho", "rho_v", "E"]
variable_names(::EulerEquations{2}) = ["rho", "rho_vx", "rho_vy", "E"]
variable_names(::EulerEquations{3}) = ["rho", "rho_vx", "rho_vy", "rho_vz", "E"]
variable_names(::IdealMHDEquations) = ["rho", "rho_vx", "rho_vy", "rho_vz", "E", "Bx", "By", "Bz"]
variable_names(::NavierStokesEquations{1}) = ["rho", "rho_v", "E"]
variable_names(::NavierStokesEquations{2}) = ["rho", "rho_vx", "rho_vy", "E"]
variable_names(::SRMHDEquations) = ["D", "Sx", "Sy", "Sz", "tau", "Bx", "By", "Bz"]
variable_names(::GRMHDEquations) = ["D", "Sx", "Sy", "Sz", "tau", "Bx", "By", "Bz"]
variable_names(::ShallowWaterEquations{1}) = ["h", "hv"]
variable_names(::ShallowWaterEquations{2}) = ["h", "hvx", "hvy"]
variable_names(::SRHydroEquations{1}) = ["D", "Sx", "tau"]
variable_names(::SRHydroEquations{2}) = ["D", "Sx", "Sy", "tau"]
variable_names(::TwoFluidEquations{1}) = ["rho_i", "rho_i_v", "E_i", "rho_e", "rho_e_v", "E_e"]
variable_names(::TwoFluidEquations{2}) = ["rho_i", "rho_i_vx", "rho_i_vy", "E_i", "rho_e", "rho_e_vx", "rho_e_vy", "E_e"]
variable_names(::ResistiveMHDEquations) = ["rho", "rho_vx", "rho_vy", "rho_vz", "E", "Bx", "By", "Bz"]
variable_names(::HallMHDEquations) = ["rho", "rho_vx", "rho_vy", "rho_vz", "E", "Bx", "By", "Bz"]

function variable_names(law::ReactiveEulerEquations{1, NS}) where {NS}
    base = ["rho", "rho_v", "E"]
    for name in law.species_names
        push!(base, "rho_Y_$(name)")
    end
    return base
end

function variable_names(law::ReactiveEulerEquations{2, NS}) where {NS}
    base = ["rho", "rho_vx", "rho_vy", "E"]
    for name in law.species_names
        push!(base, "rho_Y_$(name)")
    end
    return base
end

# ============================================================
# Mesh serialization
# ============================================================

"""
    mesh_to_dict(mesh::StructuredMesh1D) -> Dict{String,Any}

Serialize a 1D structured mesh to a dictionary.
"""
function mesh_to_dict(mesh::StructuredMesh1D)
    return Dict{String, Any}(
        "type" => "StructuredMesh1D",
        "ncells" => ncells(mesh),
        "xmin" => mesh.xmin,
        "xmax" => mesh.xmax,
        "dx" => mesh.dx,
    )
end

"""
    mesh_to_dict(mesh::StructuredMesh2D) -> Dict{String,Any}

Serialize a 2D structured mesh to a dictionary.
"""
function mesh_to_dict(mesh::StructuredMesh2D)
    return Dict{String, Any}(
        "type" => "StructuredMesh2D",
        "nx" => mesh.nx,
        "ny" => mesh.ny,
        "xmin" => mesh.xmin,
        "xmax" => mesh.xmax,
        "ymin" => mesh.ymin,
        "ymax" => mesh.ymax,
        "dx" => mesh.dx,
        "dy" => mesh.dy,
    )
end

"""
    mesh_to_dict(mesh::StructuredMesh3D) -> Dict{String,Any}

Serialize a 3D structured mesh to a dictionary.
"""
function mesh_to_dict(mesh::StructuredMesh3D)
    return Dict{String, Any}(
        "type" => "StructuredMesh3D",
        "nx" => mesh.nx,
        "ny" => mesh.ny,
        "nz" => mesh.nz,
        "xmin" => mesh.xmin,
        "xmax" => mesh.xmax,
        "ymin" => mesh.ymin,
        "ymax" => mesh.ymax,
        "zmin" => mesh.zmin,
        "zmax" => mesh.zmax,
        "dx" => mesh.dx,
        "dy" => mesh.dy,
        "dz" => mesh.dz,
    )
end

"""
    mesh_to_dict(mesh::UnstructuredHyperbolicMesh) -> Dict{String,Any}

Serialize an unstructured mesh to a dictionary.
"""
function mesh_to_dict(mesh::UnstructuredHyperbolicMesh)
    return Dict{String, Any}(
        "type" => "UnstructuredHyperbolicMesh",
        "ntriangles" => length(mesh.tri_centroids),
    )
end

"""
    mesh_to_dict(geom::FVMGeometry) -> Dict{String,Any}

Serialize a vertex-centered FVM geometry to a dictionary.
"""
function mesh_to_dict(geom::FVMGeometry)
    tri = geom.triangulation
    return Dict{String, Any}(
        "type" => "FVMGeometry",
        "num_triangles" => num_solid_triangles(tri),
        "num_vertices" => length(collect(each_solid_vertex(tri))),
    )
end

# ============================================================
# Conserved quantity integration
# ============================================================

"""
    conserved_totals(law, U::AbstractVector{<:SVector}, mesh::StructuredMesh1D)
        -> Dict{String,Float64}

Compute integrated conserved quantities over a 1D structured mesh.
`U` should be the interior solution (no ghost cells).
"""
function conserved_totals(law, U::AbstractVector{<:SVector}, mesh::StructuredMesh1D)
    names = variable_names(law)
    nv = nvariables(law)
    totals = zeros(nv)
    dx = mesh.dx
    for u in U
        for k in 1:nv
            totals[k] += u[k] * dx
        end
    end
    return Dict{String, Float64}(names[k] => totals[k] for k in 1:nv)
end

"""
    conserved_totals(law, U::AbstractMatrix{<:SVector}, mesh::StructuredMesh2D)
        -> Dict{String,Float64}

Compute integrated conserved quantities over a 2D structured mesh.
`U` should be the interior solution (no ghost cells).
"""
function conserved_totals(law, U::AbstractMatrix{<:SVector}, mesh::StructuredMesh2D)
    names = variable_names(law)
    nv = nvariables(law)
    totals = zeros(nv)
    dA = mesh.dx * mesh.dy
    for j in axes(U, 2), i in axes(U, 1)
        u = U[i, j]
        for k in 1:nv
            totals[k] += u[k] * dA
        end
    end
    return Dict{String, Float64}(names[k] => totals[k] for k in 1:nv)
end

"""
    conserved_totals(law, U::AbstractArray{<:SVector,3}, mesh::StructuredMesh3D)
        -> Dict{String,Float64}

Compute integrated conserved quantities over a 3D structured mesh.
"""
function conserved_totals(law, U::AbstractArray{<:SVector, 3}, mesh::StructuredMesh3D)
    names = variable_names(law)
    nv = nvariables(law)
    totals = zeros(nv)
    dV = mesh.dx * mesh.dy * mesh.dz
    for k3 in axes(U, 3), k2 in axes(U, 2), k1 in axes(U, 1)
        u = U[k1, k2, k3]
        for k in 1:nv
            totals[k] += u[k] * dV
        end
    end
    return Dict{String, Float64}(names[k] => totals[k] for k in 1:nv)
end

# ============================================================
# Snapshot serialization to Dict
# ============================================================

"""
    snapshot_to_dict(snap::FVMSnapshot) -> Dict{String,Any}

Convert a snapshot to a plain dictionary (for JSON serialization).
The solution field is converted to nested arrays of floats.
"""
function snapshot_to_dict(snap::FVMSnapshot)
    return Dict{String, Any}(
        "time" => snap.time,
        "step" => snap.step,
        "residual_norm" => snap.residual_norm,
        "conserved_totals" => snap.conserved_totals,
        "dt" => snap.dt,
        "wall_time" => snap.wall_time,
        "solution" => _solution_to_dict(snap.solution),
    )
end

# Solution conversion helpers
_solution_to_dict(U::AbstractVector{<:SVector}) = [collect(u) for u in U]
_solution_to_dict(U::AbstractMatrix{<:SVector}) = [[collect(U[i, j]) for i in axes(U, 1)] for j in axes(U, 2)]
_solution_to_dict(U::AbstractArray{<:SVector, 3}) = [[[collect(U[i, j, k]) for i in axes(U, 1)] for j in axes(U, 2)] for k in axes(U, 3)]
_solution_to_dict(U::AbstractVector{<:Real}) = collect(U)
_solution_to_dict(U::AbstractMatrix{<:Real}) = [collect(U[:, j]) for j in axes(U, 2)]

"""
    session_to_dict(session::FVMSessionData) -> Dict{String,Any}

Convert an entire session to a plain dictionary (for JSON serialization).
"""
function session_to_dict(session::FVMSessionData)
    return Dict{String, Any}(
        "version" => "1.0",
        "problem_type" => session.problem_type,
        "law" => session.law_name,
        "mesh" => session.mesh_info,
        "variables" => session.variable_names,
        "parameters" => session.parameters,
        "snapshots" => [snapshot_to_dict(s) for s in session.snapshots],
        "convergence" => session.convergence_data,
    )
end

# ============================================================
# Stubs for extension-provided functions
# ============================================================

"""
    export_session(session::FVMSessionData, filename::AbstractString)

Write the session data to a JSON file. Requires JSON3 to be loaded
(install via `using JSON3`).
"""
function export_session end

"""
    import_session(filename::AbstractString) -> Dict

Read a `.fvm-session.json` file. Requires JSON3 to be loaded.
"""
function import_session end

"""
    serve_dashboard(; port=8765, session_data=nothing)

Start a WebSocket server for live dashboard updates.
Requires HTTP and JSON3 to be loaded.
"""
function serve_dashboard end

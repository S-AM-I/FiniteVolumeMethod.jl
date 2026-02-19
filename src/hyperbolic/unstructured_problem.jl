# ============================================================
# Unstructured Hyperbolic Problem
# ============================================================
#
# Problem type for cell-centered FVM on unstructured triangular
# meshes. Each boundary segment can have its own BC.

"""
    UnstructuredHyperbolicProblem{Law,Mesh,RS,Recon,BCs,IC,FT}

Hyperbolic conservation law problem on an unstructured triangular mesh.

# Fields
- `law`: Conservation law (e.g. `EulerEquations{2}`).
- `mesh`: `UnstructuredHyperbolicMesh`.
- `riemann_solver`: Riemann solver (e.g. `HLLSolver()`).
- `reconstruction`: Reconstruction scheme (currently `NoReconstruction()`).
- `boundary_conditions`: Dict mapping segment ID → BC.
- `default_bc`: Fallback BC for segments not in the dict.
- `initial_condition`: Function `(x, y) → SVector{N}` of primitive variables.
- `initial_time`, `final_time`, `cfl`: Simulation parameters.
"""
struct UnstructuredHyperbolicProblem{Law, Mesh, RS, Recon, BCs, DBC, IC, FT}
    law::Law
    mesh::Mesh
    riemann_solver::RS
    reconstruction::Recon
    boundary_conditions::BCs
    default_bc::DBC
    initial_condition::IC
    initial_time::FT
    final_time::FT
    cfl::FT
end

"""
    UnstructuredHyperbolicProblem(
        law, mesh, solver, recon, bc, ic;
        initial_time=0.0, final_time, cfl=0.3
    )

Construct a problem with a single BC applied to all boundary segments.
"""
function UnstructuredHyperbolicProblem(
        law, mesh::UnstructuredHyperbolicMesh, solver, recon,
        bc, ic;
        initial_time = 0.0, final_time, cfl = 0.3
    )
    FT = typeof(final_time)
    return UnstructuredHyperbolicProblem(
        law, mesh, solver, recon,
        Dict{Int, typeof(bc)}(),
        bc, ic,
        FT(initial_time), FT(final_time), FT(cfl)
    )
end

"""
    UnstructuredHyperbolicProblem(
        law, mesh, solver, recon, bcs::Dict, default_bc, ic;
        initial_time=0.0, final_time, cfl=0.3
    )

Construct a problem with per-segment boundary conditions.
"""
function UnstructuredHyperbolicProblem(
        law, mesh::UnstructuredHyperbolicMesh, solver, recon,
        bcs::Dict, default_bc, ic;
        initial_time = 0.0, final_time, cfl = 0.3
    )
    FT = typeof(final_time)
    return UnstructuredHyperbolicProblem(
        law, mesh, solver, recon,
        bcs, default_bc, ic,
        FT(initial_time), FT(final_time), FT(cfl)
    )
end

function Base.show(io::IO, ::MIME"text/plain", prob::UnstructuredHyperbolicProblem)
    nc = prob.mesh.ntri
    t0 = prob.initial_time
    tf = prob.final_time
    law_name = nameof(typeof(prob.law))
    rs_name = nameof(typeof(prob.riemann_solver))
    return print(io, "UnstructuredHyperbolicProblem: $law_name with $rs_name on $nc cells, t ∈ ($t0, $tf)")
end

"""
    get_bc(prob::UnstructuredHyperbolicProblem, segment::Int)

Get the boundary condition for a given boundary segment.
"""
function get_bc(prob::UnstructuredHyperbolicProblem, segment::Int)
    return get(prob.boundary_conditions, segment, prob.default_bc)
end

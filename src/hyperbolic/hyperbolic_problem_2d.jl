"""
    HyperbolicProblem2D{Law, Mesh, RS, Rec, BC_L, BC_R, BC_B, BC_T, IC, FT}

A cell-centered finite volume problem for 2D hyperbolic conservation laws
on structured Cartesian meshes.

# Fields
- `law`: The conservation law (e.g., `EulerEquations{2}`).
- `mesh`: The 2D computational mesh (`StructuredMesh2D`).
- `riemann_solver`: The approximate Riemann solver.
- `reconstruction`: Reconstruction scheme.
- `bc_left`, `bc_right`, `bc_bottom`, `bc_top`: Boundary conditions on each side.
- `initial_condition`: Function `(x, y) -> SVector{N}` of primitive variables.
- `initial_time::FT`, `final_time::FT`: Time span.
- `cfl::FT`: CFL number for time step control.
"""
struct HyperbolicProblem2D{Law, Mesh, RS, Rec, BC_L, BC_R, BC_B, BC_T, IC, FT}
    law::Law
    mesh::Mesh
    riemann_solver::RS
    reconstruction::Rec
    bc_left::BC_L
    bc_right::BC_R
    bc_bottom::BC_B
    bc_top::BC_T
    initial_condition::IC
    initial_time::FT
    final_time::FT
    cfl::FT
end

function HyperbolicProblem2D(
        law, mesh, riemann_solver, reconstruction,
        bc_left, bc_right, bc_bottom, bc_top,
        initial_condition; initial_time = 0.0, final_time, cfl = 0.5
    )
    return HyperbolicProblem2D(
        law, mesh, riemann_solver, reconstruction,
        bc_left, bc_right, bc_bottom, bc_top,
        initial_condition,
        initial_time, final_time, cfl
    )
end

function Base.show(io::IO, ::MIME"text/plain", prob::HyperbolicProblem2D)
    nc = ncells(prob.mesh)
    t0 = prob.initial_time
    tf = prob.final_time
    nx, ny = prob.mesh.nx, prob.mesh.ny
    law_name = nameof(typeof(prob.law))
    rs_name = nameof(typeof(prob.riemann_solver))
    return print(io, "HyperbolicProblem2D: $law_name with $rs_name on $(nx)×$(ny) cells, t ∈ ($t0, $tf)")
end

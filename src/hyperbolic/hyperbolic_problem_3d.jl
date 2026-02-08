# ============================================================
# 3D Hyperbolic Problem Definition
# ============================================================

"""
    HyperbolicProblem3D{Law, Mesh, RS, Rec, BC_L, BC_R, BC_Bo, BC_T, BC_F, BC_Ba, IC, FT}

A cell-centered finite volume problem for 3D hyperbolic conservation laws
on structured Cartesian meshes.

# Fields
- `law`: The conservation law (e.g., `EulerEquations{3}`).
- `mesh`: The 3D computational mesh (`StructuredMesh3D`).
- `riemann_solver`: The approximate Riemann solver.
- `reconstruction`: Reconstruction scheme.
- `bc_left`, `bc_right`: Boundary conditions in x-direction.
- `bc_bottom`, `bc_top`: Boundary conditions in y-direction.
- `bc_front`, `bc_back`: Boundary conditions in z-direction.
- `initial_condition`: Function `(x, y, z) -> SVector{N}` of primitive variables.
- `initial_time::FT`, `final_time::FT`: Time span.
- `cfl::FT`: CFL number for time step control.
"""
struct HyperbolicProblem3D{Law, Mesh, RS, Rec, BC_L, BC_R, BC_Bo, BC_T, BC_F, BC_Ba, IC, FT}
    law::Law
    mesh::Mesh
    riemann_solver::RS
    reconstruction::Rec
    bc_left::BC_L
    bc_right::BC_R
    bc_bottom::BC_Bo
    bc_top::BC_T
    bc_front::BC_F
    bc_back::BC_Ba
    initial_condition::IC
    initial_time::FT
    final_time::FT
    cfl::FT
end

function HyperbolicProblem3D(
        law, mesh, riemann_solver, reconstruction,
        bc_left, bc_right, bc_bottom, bc_top, bc_front, bc_back,
        initial_condition; initial_time = 0.0, final_time, cfl = 0.4
    )
    return HyperbolicProblem3D(
        law, mesh, riemann_solver, reconstruction,
        bc_left, bc_right, bc_bottom, bc_top, bc_front, bc_back,
        initial_condition,
        initial_time, final_time, cfl
    )
end

function Base.show(io::IO, ::MIME"text/plain", prob::HyperbolicProblem3D)
    nx, ny, nz = prob.mesh.nx, prob.mesh.ny, prob.mesh.nz
    t0 = prob.initial_time
    tf = prob.final_time
    law_name = nameof(typeof(prob.law))
    rs_name = nameof(typeof(prob.riemann_solver))
    return print(io, "HyperbolicProblem3D: $law_name with $rs_name on $(nx)x$(ny)x$(nz) cells, t in ($t0, $tf)")
end

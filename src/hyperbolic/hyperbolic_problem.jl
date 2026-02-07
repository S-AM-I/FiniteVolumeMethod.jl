using StaticArrays: SVector

"""
    HyperbolicProblem{Law, Mesh, RS, Rec, BCL, BCR, IC, FT}

A cell-centered finite volume problem for hyperbolic conservation laws.

# Fields
- `law`: The conservation law (e.g., `EulerEquations`).
- `mesh`: The computational mesh.
- `riemann_solver`: The approximate Riemann solver.
- `reconstruction`: Reconstruction scheme (e.g., `CellCenteredMUSCL`).
- `bc_left`: Left boundary condition.
- `bc_right`: Right boundary condition.
- `initial_condition`: Function `(x) -> SVector{N}` of primitive variables, or a vector of conserved SVectors.
- `initial_time::FT`: Start time.
- `final_time::FT`: End time.
- `cfl::FT`: CFL number for time step control.
"""
struct HyperbolicProblem{Law, Mesh, RS, Rec, BCL, BCR, IC, FT}
    law::Law
    mesh::Mesh
    riemann_solver::RS
    reconstruction::Rec
    bc_left::BCL
    bc_right::BCR
    initial_condition::IC
    initial_time::FT
    final_time::FT
    cfl::FT
end

function HyperbolicProblem(
        law, mesh, riemann_solver, reconstruction, bc_left, bc_right,
        initial_condition; initial_time = 0.0, final_time, cfl = 0.5
    )
    return HyperbolicProblem(
        law, mesh, riemann_solver, reconstruction,
        bc_left, bc_right, initial_condition,
        initial_time, final_time, cfl
    )
end

function Base.show(io::IO, ::MIME"text/plain", prob::HyperbolicProblem)
    nc = ncells(prob.mesh)
    t0 = prob.initial_time
    tf = prob.final_time
    law_name = nameof(typeof(prob.law))
    rs_name = nameof(typeof(prob.riemann_solver))
    return print(io, "HyperbolicProblem: $law_name with $rs_name on $nc cells, t ∈ ($t0, $tf)")
end

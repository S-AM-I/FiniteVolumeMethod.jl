# distributed_solve.jl — Parallel SIMPLE solver with MPI halo exchanges
#
# Test with: mpiexec -n 2 julia --project=test test/mpi_test.jl

"""
    FiniteVolumeMethod.solve_simple_distributed(
        prob::FiniteVolumeMethod.IncompressibleProblem{Dim, T},
        dmesh::DistributedFVMMesh{Dim, T};
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T} -> SolveResult{Dim, T}

Parallel SIMPLE solver with MPI halo exchanges.

Same algorithm as `solve_simple` but with:
1. Halo exchange of velocity and pressure before each iteration
2. Global residual reduction via `MPI.Allreduce`

Each rank assembles and solves on the full local mesh (initial
implementation).  Only the residual check is globally synchronized.
"""
function FiniteVolumeMethod.solve_simple_distributed(
        prob::FiniteVolumeMethod.IncompressibleProblem{Dim, T},
        dmesh::DistributedFVMMesh{Dim, T};
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    algo = prob.algorithm
    mesh = dmesh.local_mesh

    state = FiniteVolumeMethod.IncompressibleState(mesh)
    FiniteVolumeMethod.update_boundary_velocity!(state, prob.bcs, mesh)
    FiniteVolumeMethod.update_boundary_pressure!(state, prob.bcs, mesh)

    converged = false
    final_iter = 0
    residuals = Dict{Symbol, Vector{T}}(:Ux => T[], :Uy => T[], :continuity => T[])

    for iter in 1:algo.max_iterations
        final_iter = iter

        # Sync velocity and pressure across ranks before assembly
        FiniteVolumeMethod.halo_exchange!(state.U.internal, dmesh)
        FiniteVolumeMethod.halo_exchange!(state.p.internal, dmesh)

        # Assemble momentum equations
        eqs = FiniteVolumeMethod.CollocatedEquation{T}[]
        for d in 1:Dim
            eq = FiniteVolumeMethod.CollocatedEquation(mesh)
            FiniteVolumeMethod.assemble_momentum!(eq, state, prob, d)
            push!(eqs, eq)
        end

        # Extract momentum operators
        FiniteVolumeMethod.extract_momentum_operators!(state, eqs, mesh)

        # Under-relax and solve each velocity component
        for d in 1:Dim
            U_old_d = FiniteVolumeMethod._extract_component(state.U, d)
            FiniteVolumeMethod.under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            label = d == 1 ? :Ux : (d == 2 ? :Uy : :Uz)
            sol = FiniteVolumeMethod._dispatch_solve(
                FiniteVolumeMethod.to_linear_problem(eqs[d]),
                linear_solver, solver_config, label,
            )
            FiniteVolumeMethod._set_component!(state.U, d, sol.u)
        end
        FiniteVolumeMethod.update_boundary_velocity!(state, prob.bcs, mesh)

        # Pressure equation
        p_eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_pressure!(p_eq, state, prob)
        if FiniteVolumeMethod._needs_pressure_reference(prob.bcs)
            FiniteVolumeMethod.fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = FiniteVolumeMethod._dispatch_solve(
            FiniteVolumeMethod.to_linear_problem(p_eq),
            linear_solver, solver_config, :p,
        )

        # Under-relax pressure
        nc = length(mesh.cell_volumes)
        for c in 1:nc
            state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
        end
        FiniteVolumeMethod.update_boundary_pressure!(state, prob.bcs, mesh)

        # Correct velocity and fluxes
        FiniteVolumeMethod.correct_velocity!(state, mesh)
        FiniteVolumeMethod.update_boundary_velocity!(state, prob.bcs, mesh)
        FiniteVolumeMethod.correct_fluxes!(state, mesh)

        # Global residual via MPI reduction. `continuity_residual` runs
        # on the local submesh; owned-cell contributions are summed across
        # ranks. (A future Stage 2e enhancement will weight by owned-cell
        # count so the returned value is a proper L^2 norm over the global
        # domain; for now it's the domain-sum divergence residual, which
        # is what the serial path also returns.)
        local_cont = FiniteVolumeMethod.continuity_residual(state, mesh)
        global_cont = MPI.Allreduce(local_cont, MPI.SUM, dmesh.comm)
        push!(residuals[:continuity], global_cont)

        if verbose && dmesh.rank == 0
            println("SIMPLE iter $iter: continuity = $global_cont")
        end

        if global_cont < algo.tolerance
            converged = true
            break
        end
    end

    return FiniteVolumeMethod.SolveResult{Dim, T}(
        converged, final_iter, residuals, state,
    )
end

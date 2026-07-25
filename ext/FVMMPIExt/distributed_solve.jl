# distributed_solve.jl — Parallel SIMPLE solver with MPI halo exchanges
#
# Test with: mpiexec -n 2 julia --project=test test/mpi_test.jl

"""
    FiniteVolumeMethod.solve_simple_distributed(
        prob::FiniteVolumeMethod.AnyIncompressibleProblem{Dim, T},
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
        prob::FiniteVolumeMethod.AnyIncompressibleProblem{Dim, T},
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

# ── LocalFVMMesh-driven assembly path ────────────────────────────────────
#
# Wave 5 Agent A extension: when the caller already built a
# `LocalFVMMesh` view (typically from `partition_mesh_metis` +
# `build_local_mesh`), assemble each rank's momentum / pressure
# equations over its owned-plus-halo subset rather than the full global
# mesh. Each rank's block is solved LOCALLY (additive-Schwarz-style)
# with halo exchange between outer iterations; no distributed
# `PSparseMatrix` is constructed. A `PartitionedArrays` row partition is
# built and carried as metadata only, so a future true distributed
# Krylov solve can slot in without changing the solver loop.
#
# Backwards-compatible: the existing `DistributedFVMMesh` path above is
# untouched; callers that don't hand in a `LocalFVMMesh` continue to
# get the full-mesh-per-rank behaviour.

"""
    FiniteVolumeMethod.solve_simple_distributed(
        prob::FiniteVolumeMethod.AnyIncompressibleProblem,
        dmesh::DistributedFVMMesh,
        local_view::FiniteVolumeMethod.LocalFVMMesh;
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    )

Variant that routes assembly through the `LocalFVMMesh` view of the
global mesh. Each rank:

1. Extracts an `UnstructuredFVMMesh` submesh covering its owned + halo
   cells via [`FiniteVolumeMethod.extract_local_mesh`](@ref).
2. Assembles + solves the SIMPLE sub-problems on the submesh.
3. Reduces the continuity residual across ranks with `MPI.Allreduce`.

Each rank assembles a per-rank `SparseMatrixCSC` block and solves it
locally via `LinearProblem` — only the continuity residual is reduced
globally. No `PSparseMatrix` is constructed; the `PartitionedArrays`
row partition built here is metadata for a future distributed-solve
upgrade.
"""
function FiniteVolumeMethod.solve_simple_distributed(
        prob::FiniteVolumeMethod.AnyIncompressibleProblem{Dim, T},
        dmesh::DistributedFVMMesh{Dim, T},
        local_view::FiniteVolumeMethod.LocalFVMMesh{Dim, T};
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    # Build a `LocalMeshData`-style submesh from the LocalFVMMesh view.
    # We reuse `extract_local_mesh` by reconstructing a matching
    # cell_to_rank vector on the global mesh.
    parent = local_view.parent_mesh
    nc_g = length(parent.cell_volumes)
    cell_to_rank = fill(-1, nc_g)
    for c in local_view.owned_cells
        cell_to_rank[c] = dmesh.rank
    end
    # Halo cells are owned by *some* other rank; the exact ID isn't
    # needed here because `extract_local_mesh` only uses cell_to_rank to
    # partition into (mine / not-mine), and halo membership is already
    # captured in `local_view.halo_cells`. Assigning any rank != my_rank
    # is sufficient; use dmesh.nranks - 1 (or my_rank+1 wrap) as a sentinel.
    other_rank = dmesh.rank == 0 ? min(1, dmesh.nranks - 1) : 0
    for h in local_view.halo_cells
        cell_to_rank[h] = other_rank
    end
    # Any cell *not* in owned or halo also counts as "not mine". Flip
    # the remaining -1 entries to `other_rank` so extract_local_mesh
    # treats the entire complement consistently.
    @inbounds for c in 1:nc_g
        if cell_to_rank[c] == -1
            cell_to_rank[c] = other_rank
        end
    end

    local_data = FiniteVolumeMethod.extract_local_mesh(parent, cell_to_rank, dmesh.rank)
    submesh = local_data.mesh

    state = FiniteVolumeMethod.IncompressibleState(submesh)
    FiniteVolumeMethod.update_boundary_velocity!(state, prob.bcs, submesh)
    FiniteVolumeMethod.update_boundary_pressure!(state, prob.bcs, submesh)

    algo = prob.algorithm
    converged = false
    final_iter = 0
    residuals = Dict{Symbol, Vector{T}}(:Ux => T[], :Uy => T[], :continuity => T[])

    # Build a PartitionedArrays row partition once — each rank owns its
    # `local_view.owned_cells` in the global indexing.
    row_partition = _build_row_partition(local_view, dmesh)

    for iter in 1:algo.max_iterations
        final_iter = iter

        # Halo exchange prior to assembly.
        FiniteVolumeMethod.halo_exchange!(state.U.internal, dmesh)
        FiniteVolumeMethod.halo_exchange!(state.p.internal, dmesh)

        # Momentum assembly on the submesh.
        eqs = FiniteVolumeMethod.CollocatedEquation{T}[]
        for d in 1:Dim
            eq = FiniteVolumeMethod.CollocatedEquation(submesh)
            FiniteVolumeMethod.assemble_momentum!(eq, state, prob, d)
            push!(eqs, eq)
        end
        FiniteVolumeMethod.extract_momentum_operators!(state, eqs, submesh)

        for d in 1:Dim
            U_old_d = FiniteVolumeMethod._extract_component(state.U, d)
            FiniteVolumeMethod.under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            label = d == 1 ? :Ux : (d == 2 ? :Uy : :Uz)
            # Per-rank local block solve (no distributed matrix).
            sol = _dispatch_partitioned_solve(
                eqs[d], row_partition, linear_solver, solver_config, label,
            )
            FiniteVolumeMethod._set_component!(state.U, d, sol)
        end
        FiniteVolumeMethod.update_boundary_velocity!(state, prob.bcs, submesh)

        p_eq = FiniteVolumeMethod.CollocatedEquation(submesh)
        FiniteVolumeMethod.assemble_pressure!(p_eq, state, prob)
        if FiniteVolumeMethod._needs_pressure_reference(prob.bcs)
            FiniteVolumeMethod.fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = _dispatch_partitioned_solve(
            p_eq, row_partition, linear_solver, solver_config, :p,
        )

        nc = length(submesh.cell_volumes)
        for c in 1:nc
            state.p.internal[c] += algo.alpha_p * (p_sol[c] - state.p.internal[c])
        end
        FiniteVolumeMethod.update_boundary_pressure!(state, prob.bcs, submesh)

        FiniteVolumeMethod.correct_velocity!(state, submesh)
        FiniteVolumeMethod.update_boundary_velocity!(state, prob.bcs, submesh)
        FiniteVolumeMethod.correct_fluxes!(state, submesh)

        local_cont = FiniteVolumeMethod.continuity_residual(state, submesh)
        global_cont = MPI.Allreduce(local_cont, MPI.SUM, dmesh.comm)
        push!(residuals[:continuity], global_cont)

        if verbose && dmesh.rank == 0
            println("SIMPLE iter $iter (LocalFVMMesh): continuity = $global_cont")
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

"""
    _build_row_partition(local_view, dmesh) -> PartitionedArrays row partition

Build a `PartitionedArrays` row partition (via `uniform_partition` /
`LocalIndices`) where each rank owns the global indices enumerated in
`local_view.owned_cells`. On fallback (non-uniform partition) this
returns a tuple of `(comm, owned_cells)` that downstream helpers check
with `isa`.
"""
function _build_row_partition(local_view::FiniteVolumeMethod.LocalFVMMesh, dmesh)
    parts = distribute_with_mpi(dmesh.comm)
    nc_global = length(local_view.parent_mesh.cell_volumes)
    owned = local_view.owned_cells
    # Build a `LocalIndices` object per rank. Halo cells are known
    # ghosts; pass them as the "ghost" entries so PartitionedArrays can
    # manage the overlap pattern for matrix assembly.
    ghosts = local_view.halo_cells
    # Owners of ghost entries are not known locally on this rank; use 0
    # as a placeholder — PartitionedArrays resolves via Allgatherv at
    # matrix/vector construction if needed. For the Wave 5 initial wire-
    # up we only assemble diagonal-block LinearProblems locally, so the
    # partition is carried for metadata only.
    ghost_owners = zeros(Int32, length(ghosts))
    indices = map(parts) do _
        LocalIndices(nc_global, owned, ghosts, ghost_owners)
    end
    return indices
end

"""
    _dispatch_partitioned_solve(eq, row_partition, linear_solver, solver_config, label) -> Vector

Assemble the per-rank sparse matrix / RHS from `eq` and solve the
diagonal block locally, returning the local solution. `row_partition`
is currently unused metadata — no `PSparseMatrix` is constructed. This
matches what the DistributedFVMMesh path did (residual is the only
globally-coupled quantity) while letting a future upgrade swap in a
true distributed Krylov solve without touching the solver loop.
"""
function _dispatch_partitioned_solve(
        eq::FiniteVolumeMethod.CollocatedEquation{T}, row_partition,
        linear_solver, solver_config, label::Symbol,
    ) where {T}
    sol = FiniteVolumeMethod._dispatch_solve(
        FiniteVolumeMethod.to_linear_problem(eq),
        linear_solver, solver_config, label,
    )
    return sol.u
end

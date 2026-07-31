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

    # Re-root the problem on this rank's local (owned + halo) mesh: the
    # assembly kernels read `prob.mesh`, and equations/state below are
    # sized for the local mesh, not the global one the caller's problem
    # carries. Passing the global problem produced a BoundsError in
    # `add_face_coeffs_PN!` the moment a global face referenced a cell
    # beyond the local index range.
    prob = FiniteVolumeMethod.SteadyIncompressibleProblem(
        mesh, prob.bcs, algo;
        nu = prob.nu, density = prob.density, model = prob.model,
    )

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

        # Under-relax, pin halo rows as Dirichlet transmission conditions
        # at their exchanged values (a free halo unknown with a cut-stencil
        # equation acts as a phantom boundary and poisons the owned-cell
        # solution through the coupling), then solve each component.
        for d in 1:Dim
            U_old_d = FiniteVolumeMethod._extract_component(state.U, d)
            FiniteVolumeMethod.under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            _pin_halo_rows!(eqs[d], U_old_d, dmesh.n_owned, dmesh.n_local)
            label = d == 1 ? :Ux : (d == 2 ? :Uy : :Uz)
            sol = FiniteVolumeMethod._dispatch_solve(
                FiniteVolumeMethod.to_linear_problem(eqs[d]),
                linear_solver, solver_config, label,
            )
            FiniteVolumeMethod._set_component!(state.U, d, sol.u)
        end
        FiniteVolumeMethod.update_boundary_velocity!(state, prob.bcs, mesh)

        # Extract operators from the RELAXED, solved momentum equations
        # (standard SIMPLE ordering — the un-relaxed diagonal would
        # inflate D = V/A_P by 1/alpha_U and shift the converged flux
        # field, the Stage-5a defect). Halo cells' operators are then
        # overwritten with the owning rank's values: their local rows are
        # pinned identities, and the pressure equation reads A_P/H at
        # interface faces.
        FiniteVolumeMethod.extract_momentum_operators!(state, eqs, mesh)
        FiniteVolumeMethod.halo_exchange!(state.A_P, dmesh)
        FiniteVolumeMethod.halo_exchange!(state.H_U, dmesh)

        # Pressure equation. Halo rows are pinned at the exchanged
        # pressure values (Dirichlet transmission), which also makes every
        # rank's system non-singular — the global reference is pinned only
        # on the rank that owns global cell 1, matching the serial
        # convention.
        p_eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_pressure!(p_eq, state, prob)
        _pin_halo_rows!(p_eq, state.p.internal, dmesh.n_owned, dmesh.n_local)
        if FiniteVolumeMethod._needs_pressure_reference(prob.bcs)
            ref_local = get(dmesh.global_to_local, 1, 0)
            if ref_local != 0 && ref_local <= dmesh.n_owned
                FiniteVolumeMethod.fix_pressure_reference!(p_eq, ref_local, zero(T))
            end
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

        # Globally consistent flux-normalized continuity residual over
        # owned cells only — halo cells carry cut stencils, so including
        # them reports spurious divergence regardless of solution quality.
        global_cont = _distributed_continuity_residual(
            state, mesh, dmesh.n_owned, dmesh.comm,
        )
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

    # Re-root the problem on the submesh — the assembly kernels read
    # `prob.mesh`, and the equations/state below are sized for the
    # submesh (see the same re-rooting in the Additive Schwarz path).
    prob = FiniteVolumeMethod.SteadyIncompressibleProblem(
        submesh, prob.bcs, prob.algorithm;
        nu = prob.nu, density = prob.density, model = prob.model,
    )

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
        for d in 1:Dim
            U_old_d = FiniteVolumeMethod._extract_component(state.U, d)
            FiniteVolumeMethod.under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            _pin_halo_rows!(eqs[d], U_old_d, local_data.n_owned, local_data.n_local)
            label = d == 1 ? :Ux : (d == 2 ? :Uy : :Uz)
            # Per-rank local block solve (no distributed matrix).
            sol = _dispatch_partitioned_solve(
                eqs[d], row_partition, linear_solver, solver_config, label,
            )
            FiniteVolumeMethod._set_component!(state.U, d, sol)
        end
        FiniteVolumeMethod.update_boundary_velocity!(state, prob.bcs, submesh)

        # Extract operators from the relaxed, solved equations, then
        # overwrite halo operators with owner values — same ordering and
        # rationale as the DistributedFVMMesh path above.
        FiniteVolumeMethod.extract_momentum_operators!(state, eqs, submesh)
        FiniteVolumeMethod.halo_exchange!(state.A_P, dmesh)
        FiniteVolumeMethod.halo_exchange!(state.H_U, dmesh)

        p_eq = FiniteVolumeMethod.CollocatedEquation(submesh)
        FiniteVolumeMethod.assemble_pressure!(p_eq, state, prob)
        _pin_halo_rows!(p_eq, state.p.internal, local_data.n_owned, local_data.n_local)
        if FiniteVolumeMethod._needs_pressure_reference(prob.bcs)
            ref_local = get(dmesh.global_to_local, 1, 0)
            if ref_local != 0 && ref_local <= dmesh.n_owned
                FiniteVolumeMethod.fix_pressure_reference!(p_eq, ref_local, zero(T))
            end
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

        global_cont = _distributed_continuity_residual(
            state, submesh, local_data.n_owned, dmesh.comm,
        )
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

"""
    _pin_halo_rows!(eq, values, n_owned, n_local)

Pin the halo cells (local indices `n_owned+1 .. n_local`) of a local
equation as Dirichlet transmission conditions at `values` — the field
values received from the owning ranks by the preceding `halo_exchange!`.

Halo rows carry cut stencils (their off-rank faces are absent from the
local mesh), so leaving them as free unknowns lets a wrong equation
couple back into the owned cells. Symmetric elimination is used — column
entries are moved to the RHS and zeroed — so an SPD pressure matrix
stays SPD for CG/AMG.
"""
function _pin_halo_rows!(
        eq::FiniteVolumeMethod.CollocatedEquation{T}, values, n_owned::Int, n_local::Int,
    ) where {T}
    A = eq.A
    b = eq.b
    rows = SparseArrays.rowvals(A)
    vals = SparseArrays.nonzeros(A)
    @inbounds for j in 1:n_local
        j_halo = j > n_owned
        for ptr in SparseArrays.nzrange(A, j)
            r = rows[ptr]
            r == j && continue
            if j_halo
                # Column of a halo cell: move the contribution to the RHS
                # of non-halo rows, then zero the entry.
                if r <= n_owned
                    b[r] -= vals[ptr] * T(values[j])
                end
                vals[ptr] = zero(T)
            elseif r > n_owned
                # Off-diagonal entry in a halo row: zero it.
                vals[ptr] = zero(T)
            end
        end
    end
    @inbounds for h in (n_owned + 1):n_local
        vals[eq.pattern.diag_idx[h]] = one(T)
        b[h] = T(values[h])
    end
    return nothing
end

"""
    _distributed_continuity_residual(state, mesh, n_owned, comm) -> T

Globally consistent flux-normalized continuity residual for the
distributed solve. The serial `continuity_residual` sums |imbalance| over
every local cell — but halo cells carry cut stencils on the local mesh,
so their imbalance is spuriously large regardless of solution quality.
Here the numerator counts owned cells only (their face sets are complete
by construction of the halo layer), the denominator counts each global
face exactly once (on the rank that owns its P cell), and both are
`Allreduce`-summed so every rank returns the same value — matching the
serial metric at the converged state.
"""
function _distributed_continuity_residual(
        state::FiniteVolumeMethod.IncompressibleState{Dim, T},
        mesh::FiniteVolumeMethod.UnstructuredFVMMesh{Dim, T},
        n_owned::Int, comm::MPI.Comm,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    imbalance = zeros(T, nc)
    flux_scale = zero(T)
    @inbounds for f in 1:nf
        F_f = state.phi.values[f]
        P = mesh.face_cells[1, f]
        if P <= n_owned
            flux_scale += abs(F_f)
        end
        imbalance[P] += F_f
        N = mesh.face_cells[2, f]
        if N != 0
            imbalance[N] -= F_f
        end
    end
    residual = zero(T)
    @inbounds for c in 1:n_owned
        residual += abs(imbalance[c])
    end
    totals = MPI.Allreduce([residual, flux_scale], MPI.SUM, comm)
    return totals[2] > eps(T) ? totals[1] / totals[2] : totals[1]
end

# incompressible/residuals.jl — Residual computation for incompressible NS
#
# Provides momentum and continuity residual norms used to monitor
# convergence of the pressure-velocity coupling iterations.

# ── Momentum residual ──────────────────────────────────────────────

@doc """
    momentum_residual(eq, u_d) -> T

Compute the normalized scale-invariant residual of a momentum
component equation, following the OpenFOAM convention:

```
    u_avg = mean(u_d)
    normFactor = Σ_c |A_c · u_d - A_c · u_avg|  +  Σ_c |b_c - A_c · u_avg|
    residual = Σ_c |A_c · u_d - b_c|  /  (normFactor + ε)
```

where `A_c · u_avg` is the matrix-vector product at row `c` with all
entries of `u_d` replaced by the mean. This normalization is insensitive
to the absolute scale of `b` (which can be small in interior-dominated
flows such as a lid-driven cavity), avoiding the "residual plateau"
pathology of the naive `||A u - b|| / ||b||` form.

Returns `zero(T)` if `normFactor ≈ 0` (e.g. trivial zero-flow problem).

Reference: OpenFOAM Foundation `solveNoBlock` residual definition,
documented in the fvSolution section of the OpenFOAM User Guide.
"""
function momentum_residual(
        eq::CollocatedEquation{T},
        u_d::Vector{T},
    ) where {T}
    nc = length(u_d)
    nc == 0 && return zero(T)

    u_avg = sum(u_d) / nc
    A = eq.A
    b = eq.b

    # Compute A * u_d and A * u_avg·1 (row by row) via A * u_d and sum(A_row)·u_avg.
    Au = A * u_d
    # A * (u_avg · 1) = u_avg · sum(A, dims=2), row-wise.
    # Compute per-row sum of A efficiently using CSC's column iteration.
    row_sum = zeros(T, nc)
    rows = A.rowval
    vals = A.nzval
    colptr = A.colptr
    @inbounds for j in 1:nc
        for k in colptr[j]:(colptr[j + 1] - 1)
            row_sum[rows[k]] += vals[k]
        end
    end

    numerator = zero(T)
    norm_factor = zero(T)
    @inbounds for c in 1:nc
        Au_c = Au[c]
        Au_avg_c = row_sum[c] * u_avg
        numerator += abs(Au_c - b[c])
        norm_factor += abs(Au_c - Au_avg_c) + abs(b[c] - Au_avg_c)
    end

    return norm_factor > eps(T) ? numerator / norm_factor : zero(T)
end

# ── Continuity residual ─────────────────────────────────────────────

@doc """
    continuity_residual(state, mesh) -> T

Compute the L1 continuity residual: the sum of absolute cell flux
imbalances across all cells.

For each cell, the flux imbalance is the sum of face fluxes (with
appropriate sign conventions).  A divergence-free velocity field
yields zero imbalance.

# Arguments
- `state::IncompressibleState` — current solver state (uses `phi`)
- `mesh::UnstructuredFVMMesh` — mesh
"""
function continuity_residual(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Accumulate flux imbalance per cell
    imbalance = zeros(T, nc)

    for f in 1:nf
        F_f = state.phi.values[f]
        P = owner(mesh, f)
        imbalance[P] += F_f

        N = neighbour(mesh, f)
        if N != 0
            imbalance[N] -= F_f
        end
    end

    # L1 norm of cell imbalances
    residual = zero(T)
    for c in 1:nc
        residual += abs(imbalance[c])
    end

    return residual
end

# ── Courant number ─────────────────────────────────────────────────

@doc """
    compute_max_courant(state, mesh, dt) -> T

Compute the maximum face Courant number over the mesh:
```
    Co = max_f |phi_f| * dt / V_owner
```

Used by adaptive time-stepping in the transient solver.

# Arguments
- `state::IncompressibleState` — current solver state (uses `phi`)
- `mesh::UnstructuredFVMMesh` — mesh
- `dt::T` — current time step size
"""
function compute_max_courant(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    co_max = zero(T)

    for f in 1:nf
        P = owner(mesh, f)
        co_f = abs(state.phi.values[f]) * dt / mesh.cell_volumes[P]
        co_max = max(co_max, co_f)
    end

    return co_max
end

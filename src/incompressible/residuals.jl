# incompressible/residuals.jl — Residual computation for incompressible NS
#
# Provides momentum and continuity residual norms used to monitor
# convergence of the pressure-velocity coupling iterations.

# ── Momentum residual ──────────────────────────────────────────────

@doc """
    momentum_residual(eq, u_d) -> T

Compute the normalized L2 residual of a momentum component equation:
```
    r = ||A * u_d - b|| / ||b||
```

Returns `zero(T)` if `||b|| ≈ 0` to avoid division by zero.

# Arguments
- `eq::CollocatedEquation{T}` — assembled momentum equation
- `u_d::Vector{T}` — current velocity component values
"""
function momentum_residual(
        eq::CollocatedEquation{T},
        u_d::Vector{T},
    ) where {T}
    residual_vec = eq.A * u_d - eq.b
    r_norm = norm(residual_vec)
    b_norm = norm(eq.b)
    return b_norm > eps(T) ? r_norm / b_norm : zero(T)
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

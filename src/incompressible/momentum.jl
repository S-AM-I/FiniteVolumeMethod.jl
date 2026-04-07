# incompressible/momentum.jl — Momentum equation assembly for incompressible NS
#
# Assembles the component-wise momentum equations including convection,
# diffusion (Laplacian), temporal term, and pressure gradient source.
# Also provides extraction of the diagonal (A_P) and off-diagonal (H)
# operators needed by the pressure equation, and under-relaxation.

# ── Momentum assembly ──────────────────────────────────────────────

@doc """
    assemble_momentum!(eq, state, prob, component; dt, scheme, nu_eff)

Assemble the momentum equation for velocity component `component` into
the `CollocatedEquation` `eq`.

The assembled equation is:
```
    div(phi * u_d) - div(nu_eff * grad(u_d)) = -dp/dx_d * V_c  [+ ddt term]
```

# Arguments
- `eq::CollocatedEquation{T}` — equation (modified in-place)
- `state::IncompressibleState` — current solver state (flux, velocity, pressure)
- `prob::IncompressibleProblem` — problem definition (mesh, BCs, viscosity)
- `component::Int` — velocity component index (1 = x, 2 = y, ...)
- `dt` — time step (if `nothing`, no temporal term is added)
- `scheme` — convection interpolation scheme
- `nu_eff` — effective viscosity: scalar `T` or per-cell `Vector{T}` (default: `prob.nu`)
- `body_force` — per-cell body force vector (e.g. buoyancy), or `nothing`
"""
function assemble_momentum!(
        eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        component::Int;
        dt::Union{Nothing, T} = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
        nu_eff::Union{T, Vector{T}} = prob.nu,
        body_force::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Expand incompressible BCs to primitive velocity BCs
    bcs_U = expand_bcs_velocity(prob.bcs, component)

    # Convection: div(phi * u_d)
    assemble_convection!(eq, state.phi, mesh, bcs_U; scheme = scheme)

    # Diffusion: -div(nu_eff * grad(u_d))  (Laplacian assembles as
    # positive-definite operator on the LHS)
    assemble_laplacian!(eq, nu_eff, mesh, bcs_U)

    # Temporal term (if transient)
    if dt !== nothing
        phi_old = _extract_component(state.U, component)
        assemble_ddt_euler!(eq, prob.density, phi_old, mesh, dt)
    end

    # Pressure gradient source: -dp/dx_d * V_c
    # Compute gradient of pressure
    grad_p = gradient(state.p, mesh)
    for c in 1:nc
        eq.b[c] -= grad_p[c][component] * mesh.cell_volumes[c]
    end

    # Body force (buoyancy, etc.)
    if body_force !== nothing
        for c in 1:nc
            eq.b[c] += body_force[c][component] * mesh.cell_volumes[c]
        end
    end

    return nothing
end

# ── Extract momentum operators ──────────────────────────────────────

@doc """
    extract_momentum_operators!(state, eqs, mesh)

Extract diagonal coefficients `A_P` and the H-operator `H(U)` from
the assembled momentum equations.

For each cell `c`:
- `A_P[c] = eqs[1].A[c, c]` (diagonal coefficient, same for all components
  on uniform meshes)
- `H_U[c] = SVector(H_1[c], H_2[c], ...)` where
  `H_d[c] = b_d[c] - sum_{N != c} A[c, N] * u_d[N]`

These operators satisfy `A_P * U = H(U) - grad(p) * V` so that
`U = H(U) / A_P - (V / A_P) * grad(p)`.

# Arguments
- `state::IncompressibleState` — state (A_P and H_U modified in-place)
- `eqs::Vector{CollocatedEquation{T}}` — assembled momentum equations (one per component)
- `mesh::UnstructuredFVMMesh` — mesh
"""
function extract_momentum_operators!(
        state::IncompressibleState{Dim, T},
        eqs::Vector{CollocatedEquation{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    A = eqs[1].A  # Matrix structure is the same for all components

    # Store diagonal
    for c in 1:nc
        state.A_P[c] = A[c, c]
    end

    # Compute H(U) per cell using face connectivity for O(nc) performance.
    # H_d[c] = b_d[c] - sum_{N: neighbor of c} A[c, N] * u_d[N]
    # We iterate over faces of cell c to find neighbors, avoiding O(nc²).
    nf = size(mesh.face_cells, 2)

    # Pre-extract velocity components for efficiency
    u_components = Vector{Vector{T}}(undef, Dim)
    for d in 1:Dim
        u_components[d] = _extract_component(state.U, d)
    end

    # Initialize H with RHS values
    for c in 1:nc
        h = ntuple(Val(Dim)) do d
            eqs[d].b[c]
        end
        state.H_U[c] = SVector{Dim, T}(h)
    end

    # Subtract off-diagonal contributions via face loop
    for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            h_P = state.H_U[P]
            h_N = state.H_U[N]
            new_P = ntuple(Val(Dim)) do d
                a_PN = eqs[d].A[P, N]
                h_P[d] - a_PN * u_components[d][N]
            end
            new_N = ntuple(Val(Dim)) do d
                a_NP = eqs[d].A[N, P]
                h_N[d] - a_NP * u_components[d][P]
            end
            state.H_U[P] = SVector{Dim, T}(new_P)
            state.H_U[N] = SVector{Dim, T}(new_N)
        end
    end

    return nothing
end

# ── Under-relaxation ────────────────────────────────────────────────

@doc """
    under_relax_momentum!(eq, U_old_d, alpha_U)

Apply under-relaxation to the momentum equation for one velocity
component.

Modifies the diagonal and RHS so that the relaxed solution satisfies:
```
    U_new = alpha_U * U_solved + (1 - alpha_U) * U_old
```

Specifically:
- `A[c, c] → A[c, c] / alpha_U`
- `b[c] += (1 - alpha_U) / alpha_U * a_P_original * U_old_d[c]`

# Arguments
- `eq::CollocatedEquation{T}` — equation (modified in-place)
- `U_old_d::Vector{T}` — previous velocity component values
- `alpha_U::T` — under-relaxation factor (0 < alpha_U <= 1)
"""
function under_relax_momentum!(
        eq::CollocatedEquation{T},
        U_old_d::Vector{T},
        alpha_U::T,
    ) where {T}
    nc = length(eq.b)
    for c in 1:nc
        a_P = eq.A[c, c]
        eq.A[c, c] = a_P / alpha_U
        eq.b[c] += (one(T) - alpha_U) / alpha_U * a_P * U_old_d[c]
    end
    return nothing
end

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
- `dt` — time step (if `nothing`, no temporal term is added).  The ddt term
  is assembled against `state.U_old` (the old-time-level snapshot), with
  unit coefficient — the momentum equation is in *kinematic* form (ν,
  volumetric flux, p/ρ), so density must not appear in the temporal term.
- `scheme` — convection interpolation scheme
- `nu_eff` — effective viscosity: scalar `T` or per-cell `Vector{T}` (default: `prob.nu`)
- `body_force` — per-cell body force vector (e.g. buoyancy), or `nothing`.
  Must be in kinematic units (force per unit mass), consistent with the
  rest of the equation.
- `t` — current simulation time, used to evaluate time-dependent BCs
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
        t::T = zero(T),
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Expand incompressible BCs to primitive velocity BCs at time t
    bcs_U = expand_bcs_velocity(prob.bcs, component; t = t)

    # Convection: div(phi * u_d)
    assemble_convection!(eq, state.phi, mesh, bcs_U; scheme = scheme)

    # Diffusion: -div(nu_eff * grad(u_d))  (Laplacian assembles as
    # positive-definite operator on the LHS).  The explicit non-orthogonal
    # correction uses the current gradient of u_d so that the over-relaxed
    # implicit split does not over-estimate diffusion on skewed meshes.
    grad_ud = gradient(_component_scalar_field(state.U, component, mesh), mesh)
    assemble_laplacian!(
        eq, nu_eff, mesh, bcs_U;
        non_ortho_correction = true, grad_phi = grad_ud,
    )

    # Temporal term (if transient): (V/dt)(u^{n+1} - u^n) against the
    # old-time snapshot.  Kinematic form → unit density coefficient.
    if dt !== nothing
        phi_old = T[u[component] for u in state.U_old]
        assemble_ddt_euler!(eq, one(T), phi_old, mesh, dt)
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

"""
    _component_scalar_field(U, d, mesh) -> CollocatedScalarField

Build a scalar field view of velocity component `d`, including boundary
face values, for gradient reconstruction.
"""
function _component_scalar_field(
        U::CollocatedVectorField{Dim, T}, d::Int,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    internal = T[u[d] for u in U.internal]
    boundary = T[u[d] for u in U.boundary]
    return CollocatedScalarField{T}(
        Symbol(:U, d), internal, boundary, U.boundary_face_indices,
    )
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
  `H_d[c] = b_d[c] + (∇p)_d[c] * V_c - sum_{N != c} A[c, N] * u_d[N]`

`assemble_momentum!` bakes the pressure-gradient source `-∇p V` into
`b`, so it must be ADDED BACK here: H is by definition the
pressure-free part of the momentum operator.  Without this, calling
`correct_velocity!` (`U = H/A_P - (V/A_P) ∇p`) after a momentum solve
would apply the pressure gradient twice (once inside `U*` via `b`, once
in the correction), which drives an unconditional instability of the
SIMPLE/PISO loop.

These operators satisfy `A_P * U = H(U) - grad(p) * V` so that
`U = H(U) / A_P - (V / A_P) * grad(p)`.

Call this AFTER the momentum solve (and after under-relaxation), while
`state.p` still holds the pressure used during assembly.

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
    pat = eqs[1].pattern

    # Store diagonal via pre-computed nzval indices (O(1) per entry)
    for c in 1:nc
        state.A_P[c] = A.nzval[pat.diag_idx[c]]
    end

    # Compute H(U) per cell using face connectivity for O(nc) performance.
    # H_d[c] = b_d[c] + (∇p)_d[c] V_c - sum_{N: neighbor of c} A[c, N] * u_d[N]
    # We iterate over faces of cell c to find neighbors, avoiding O(nc²).
    nf = size(mesh.face_cells, 2)

    # Pre-extract velocity components for efficiency
    u_components = Vector{Vector{T}}(undef, Dim)
    for d in 1:Dim
        u_components[d] = _extract_component(state.U, d)
    end

    # Initialize H with RHS values, removing the pressure-gradient source
    # that assemble_momentum! added to b (H must be pressure-free).
    grad_p = gradient(state.p, mesh)
    for c in 1:nc
        V_c = mesh.cell_volumes[c]
        h = ntuple(Val(Dim)) do d
            eqs[d].b[c] + grad_p[c][d] * V_c
        end
        state.H_U[c] = SVector{Dim, T}(h)
    end

    # Subtract off-diagonal contributions via face loop, reading the
    # off-diagonal coefficients through the pre-computed nzval indices.
    for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            idx_PN = pat.offdiag_PN[f]
            idx_NP = pat.offdiag_NP[f]
            h_P = state.H_U[P]
            h_N = state.H_U[N]
            new_P = ntuple(Val(Dim)) do d
                a_PN = eqs[d].A.nzval[idx_PN]
                h_P[d] - a_PN * u_components[d][N]
            end
            new_N = ntuple(Val(Dim)) do d
                a_NP = eqs[d].A.nzval[idx_NP]
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
    nz = eq.A.nzval
    diag_idx = eq.pattern.diag_idx
    for c in 1:nc
        a_P = nz[diag_idx[c]]
        nz[diag_idx[c]] = a_P / alpha_U
        eq.b[c] += (one(T) - alpha_U) / alpha_U * a_P * U_old_d[c]
    end
    return nothing
end

# incompressible/pressure.jl — Pressure equation assembly for incompressible NS
#
# Assembles the pressure Poisson equation derived from the continuity
# constraint.  The equation is: div((V / A_P) * grad(p)) = div(H(U) / A_P),
# discretized as a Laplacian with per-cell diffusivity D = V / A_P and
# the H/A flux divergence as a source term.

# ── H/A flux computation ───────────────────────────────────────────

@doc """
    compute_HbyA_flux(state, mesh) -> Vector{T}

Compute the face flux from the `H(U) / A_P` velocity field.

For internal faces, the flux is the linear interpolation of `H(U) / A_P`
dotted with the face area vector.  For boundary faces, the boundary
velocity from the state is used directly.

Returns a vector of length `nfaces`.
"""
function compute_HbyA_flux(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    phi_HbyA = Vector{T}(undef, nf)
    ubmap = build_boundary_map(state.U, mesh)

    for f in 1:nf
        S_f = face_normal_area(mesh, f)

        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            # H(U)/A_P at owner and neighbour
            HbyA_P = state.H_U[P] / state.A_P[P]
            HbyA_N = state.H_U[N] / state.A_P[N]

            # Linear interpolation to face
            HbyA_f = w * HbyA_P + (one(T) - w) * HbyA_N
            phi_HbyA[f] = dot(HbyA_f, S_f)
        else
            # Boundary: use boundary velocity directly
            bi = ubmap[f]
            U_b = state.U.boundary[bi]
            phi_HbyA[f] = dot(U_b, S_f)
        end
    end

    return phi_HbyA
end

# ── Pressure equation assembly ──────────────────────────────────────

@doc """
    assemble_pressure!(eq, state, prob)

Assemble the pressure Poisson equation into `eq`.

The equation is:
```
    div(D * grad(p)) = div(H(U) / A_P)
```
where `D[c] = V_c / A_P[c]` is the per-cell pressure diffusivity.

The Laplacian is assembled using the collocated operator with expanded
pressure boundary conditions.  The RHS divergence of the H/A flux is
computed face-by-face and added to `eq.b`.

If no `FixedPressureBC` exists in the boundary conditions, a pressure
reference is fixed at cell 1 to make the system non-singular.

# Arguments
- `eq::CollocatedEquation{T}` — equation (modified in-place)
- `state::IncompressibleState` — current solver state
- `prob::IncompressibleProblem` — problem definition
"""
function assemble_pressure!(
        eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Per-cell diffusivity: D = V / A_P
    D = Vector{T}(undef, nc)
    for c in 1:nc
        D[c] = mesh.cell_volumes[c] / state.A_P[c]
    end

    # Expand BCs for pressure
    bcs_p = expand_bcs_pressure(prob.bcs)

    # Assemble Laplacian: div(D * grad(p))
    assemble_laplacian!(eq, D, mesh, bcs_p)

    # Compute H/A flux divergence and add to RHS.
    # The Laplacian assembles a positive-definite operator A*p where
    # A[P,P] > 0 for the diffusion term. This corresponds to the negative
    # Laplacian: A*p represents -div(D*grad(p)). So the equation is
    # -div(D*grad(p)) = -div(phi_HbyA), and the RHS needs the NEGATIVE
    # divergence of the HbyA flux.
    phi_HbyA = compute_HbyA_flux(state, mesh)
    for f in 1:nf
        P = owner(mesh, f)
        eq.b[P] -= phi_HbyA[f]

        N = neighbour(mesh, f)
        if N != 0
            eq.b[N] += phi_HbyA[f]
        end
    end

    # Fix pressure reference if needed (pure Neumann problem)
    if _needs_pressure_reference(prob.bcs)
        fix_pressure_reference!(eq, 1, zero(T))
    end

    return nothing
end

# ── Pressure reference ──────────────────────────────────────────────

@doc """
    fix_pressure_reference!(eq, ref_cell, ref_value)

Fix the pressure at `ref_cell` to `ref_value` by zeroing the row in
the matrix, setting the diagonal to 1, and the RHS to `ref_value`.

This removes the null-space from an all-Neumann pressure system.

# Arguments
- `eq::CollocatedEquation{T}` — equation (modified in-place)
- `ref_cell::Int` — cell index at which to fix the pressure
- `ref_value::T` — prescribed pressure value
"""
function fix_pressure_reference!(
        eq::CollocatedEquation{T},
        ref_cell::Int,
        ref_value::T,
    ) where {T}
    nc = size(eq.A, 1)
    # Zero the entire row
    for j in 1:nc
        eq.A[ref_cell, j] = zero(T)
    end
    # Set diagonal to 1, RHS to reference value
    eq.A[ref_cell, ref_cell] = one(T)
    eq.b[ref_cell] = ref_value
    eq.source[ref_cell] = zero(T)
    return nothing
end

# ── Pressure reference check ───────────────────────────────────────

@doc """
    _needs_pressure_reference(bcs) -> Bool

Return `true` if no `FixedPressureBC` exists in the boundary conditions,
meaning the pressure system is pure Neumann and needs a reference point.
"""
function _needs_pressure_reference(
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
    )
    for bc in values(bcs)
        bc isa FixedPressureBC && return false
    end
    return true
end

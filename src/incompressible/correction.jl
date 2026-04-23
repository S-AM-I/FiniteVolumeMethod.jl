# incompressible/correction.jl — Velocity and flux corrections for incompressible NS
#
# After solving the pressure equation, the velocity and face fluxes must
# be corrected to satisfy the continuity constraint.  This file provides
# the correction steps and boundary value updates.

# ── Velocity correction ─────────────────────────────────────────────

@doc """
    correct_velocity!(state, mesh)

Correct the cell-centered velocity using the pressure gradient:
```
    U[c] = H_U[c] / A_P[c] - (V_c / A_P[c]) * grad(p)[c]
```

This enforces the momentum equation balance after the pressure solve.

# Arguments
- `state::IncompressibleState` — state (U modified in-place)
- `mesh::UnstructuredFVMMesh` — mesh
"""
function correct_velocity!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    # Compute pressure gradient
    grad_p = gradient(state.p, mesh)

    for c in 1:nc
        V_c = mesh.cell_volumes[c]
        D_c = V_c / state.A_P[c]
        state.U.internal[c] = state.H_U[c] / state.A_P[c] - D_c * grad_p[c]
    end

    return nothing
end

# ── Flux correction ─────────────────────────────────────────────────

@doc """
    correct_fluxes!(state, mesh)

Correct face fluxes using the Rhie-Chow momentum interpolation to
produce a divergence-free flux field.

Delegates to [`rhie_chow_correction!`](@ref).

# Arguments
- `state::IncompressibleState` — state (phi modified in-place)
- `mesh::UnstructuredFVMMesh` — mesh
"""
function correct_fluxes!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    rhie_chow_correction!(state.phi, state.U, state.p, state.A_P, mesh)
    return nothing
end

# ── Boundary velocity update ────────────────────────────────────────

@doc """
    update_boundary_velocity!(state, bcs, mesh)

Update boundary face velocity values according to the boundary condition
types:
- `FixedVelocityBC` → prescribed value
- `NoSlipWallBC` → zero
- Others (SlipWallBC, FixedPressureBC, InletOutletBC) → copy owner cell value

# Arguments
- `state::IncompressibleState` — state (U.boundary modified in-place)
- `bcs::Dict{Symbol, <:AbstractBoundaryCondition}` — boundary conditions
- `mesh::UnstructuredFVMMesh` — mesh
"""
function update_boundary_velocity!(
        state::IncompressibleState{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    for (i, f) in enumerate(state.U.boundary_face_indices)
        tag = _face_tag(mesh, f)
        bc = get(bcs, tag, nothing)
        bc === nothing && continue

        P = owner(mesh, f)

        if bc isa FixedVelocityBC
            state.U.boundary[i] = bc.value
        elseif bc isa SpatialVelocityBC
            x_f = face_center(mesh, f)
            state.U.boundary[i] = bc.func(x_f)
        elseif bc isa NoSlipWallBC
            state.U.boundary[i] = zero(SVector{Dim, T})
        elseif bc isa InletOutletBC
            state.U.boundary[i] = bc.inlet_value
        else
            # SlipWallBC, FixedPressureBC, etc.: extrapolate from owner cell
            state.U.boundary[i] = state.U.internal[P]
        end
    end

    return nothing
end

# ── Boundary pressure update ────────────────────────────────────────

@doc """
    update_boundary_pressure!(state, bcs, mesh)

Update boundary face pressure values according to the boundary condition
types:
- `FixedPressureBC` → prescribed value
- Others → copy owner cell value (zero-gradient)

# Arguments
- `state::IncompressibleState` — state (p.boundary modified in-place)
- `bcs::Dict{Symbol, <:AbstractBoundaryCondition}` — boundary conditions
- `mesh::UnstructuredFVMMesh` — mesh
"""
function update_boundary_pressure!(
        state::IncompressibleState{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    for (i, f) in enumerate(state.p.boundary_face_indices)
        tag = _face_tag(mesh, f)
        bc = get(bcs, tag, nothing)
        bc === nothing && continue

        P = owner(mesh, f)

        if bc isa FixedPressureBC
            state.p.boundary[i] = bc.value
        else
            # Zero-gradient: copy owner cell value
            state.p.boundary[i] = state.p.internal[P]
        end
    end

    return nothing
end

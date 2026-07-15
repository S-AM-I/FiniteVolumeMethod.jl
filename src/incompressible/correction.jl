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

# Keyword Arguments
- `rho_p` — per-cell density for the compressible momentum convention
  where `state.p` holds absolute pressure: the correction becomes
  `U = H/A_P - (V_c / (ρ_c A_P)) ∇p`.  Must match the `rho_p` used in
  `assemble_momentum!` / `extract_momentum_operators!`.  Default
  `nothing` keeps the incompressible (kinematic-pressure) form.
- `porous_zones` — when non-empty porous zones are active, the pressure
  term is rebuilt by FLUX RECONSTRUCTION (OpenFOAM `fvc::reconstruct` of
  the compact face-normal correction fluxes with harmonic `D_f`) instead
  of the Green-Gauss cell gradient.  At a porous interface the pressure
  slope is discontinuous; the linearly-interpolated Green-Gauss gradient
  smears the jump into the neighbouring free cells and produces spurious
  velocity spikes that destabilize the outer loop for high resistances.
  The reconstructed correction uses the same harmonic-`D_f` face fluxes
  as the pressure equation and Rhie-Chow, so it remains consistent and
  bounded across arbitrarily large resistance jumps.
"""
function correct_velocity!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T};
        rho_p::Union{Nothing, Vector{T}} = nothing,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
    ) where {Dim, T}
    if porous_zones !== nothing && any(z -> !isempty(z.cell_indices), porous_zones)
        return _correct_velocity_reconstruct!(state, mesh; rho_p = rho_p)
    end
    nc = length(mesh.cell_volumes)

    # Compute pressure gradient
    grad_p = gradient(state.p, mesh)

    for c in 1:nc
        V_c = mesh.cell_volumes[c]
        D_c = V_c / state.A_P[c]
        if rho_p !== nothing
            D_c /= rho_p[c]
        end
        state.U.internal[c] = state.H_U[c] / state.A_P[c] - D_c * grad_p[c]
    end

    return nothing
end

"""
    _correct_velocity_reconstruct!(state, mesh; rho_p = nothing)

Flux-reconstructed velocity correction:
```
    U_c = H_c/A_P - M_c⁻¹ Σ_f (S_f/|S_f|) q_f,
    M_c = Σ_f S_f S_fᵀ/|S_f|,   q_f = D_f (p_N - p_P)/|d| |S_f|
```
with harmonic face `D_f = (V/A_P)_f` identical to
[`rhie_chow_correction!`](@ref).  Boundary faces carry no correction
flux (`q_b = 0`, matching the Rhie-Chow boundary treatment) but still
contribute to `M_c`.  This is the collocated equivalent of OpenFOAM's
`fvc::reconstruct(pEqn.flux()/...)` used for porous / large-body-force
problems.
"""
function _correct_velocity_reconstruct!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T};
        rho_p::Union{Nothing, Vector{T}} = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    M = fill(zero(SMatrix{Dim, Dim, T}), nc)
    r = fill(zero(SVector{Dim, T}), nc)

    @inbounds for f in 1:nf
        S_f = face_normal_area(mesh, f)
        mag = mesh.face_areas[f]
        mag > zero(T) || continue
        outer = (S_f * S_f') / mag
        P = owner(mesh, f)
        M[P] += outer
        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            M[N] += outer

            w = face_weight(mesh, f)
            D_P = mesh.cell_volumes[P] / state.A_P[P]
            D_N = mesh.cell_volumes[N] / state.A_P[N]
            if rho_p !== nothing
                D_P /= rho_p[P]
                D_N /= rho_p[N]
            end
            denom = w * D_N + (one(T) - w) * D_P
            D_f = denom > zero(T) ? D_P * D_N / denom : zero(T)

            _, d_mag = owner_neighbour_distance(mesh, f)
            q_f = D_f * (state.p.internal[N] - state.p.internal[P]) / d_mag * mag

            contrib = S_f * (q_f / mag)
            r[P] += contrib
            r[N] += contrib
        end
    end

    @inbounds for c in 1:nc
        state.U.internal[c] = state.H_U[c] / state.A_P[c] - M[c] \ r[c]
    end

    return nothing
end

# ── Flux correction ─────────────────────────────────────────────────

@doc """
    correct_fluxes!(state, mesh; porous_zones = nothing)

Correct face fluxes using the Rhie-Chow momentum interpolation to
produce a divergence-free flux field.

Delegates to [`rhie_chow_correction!`](@ref).  When non-empty
`porous_zones` are active, the flux is instead built directly from the
pressure-equation-consistent form
`φ_f = interp(H/A_P)·S_f - D_f snGrad(p) |S_f|`
(see [`_correct_fluxes_hbya!`](@ref)): the Rhie-Chow deferred-correction
term `D_f (snGrad - interp(∇p_GG))` relies on the cell Green-Gauss
gradient, which is polluted next to a porous interface where the
pressure slope is discontinuous, and it is inconsistent with the
flux-reconstructed cell velocity used there.

# Arguments
- `state::IncompressibleState` — state (phi modified in-place)
- `mesh::UnstructuredFVMMesh` — mesh
"""
function correct_fluxes!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T};
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
    ) where {Dim, T}
    if porous_zones !== nothing && any(z -> !isempty(z.cell_indices), porous_zones)
        return _correct_fluxes_hbya!(state, mesh)
    end
    rhie_chow_correction!(state.phi, state.U, state.p, state.A_P, mesh)
    return nothing
end

"""
    _correct_fluxes_hbya!(state, mesh)

Pressure-equation-consistent flux update (OpenFOAM
`phi = phiHbyA - pEqn.flux()` equivalent):
```
    φ_f = interp(H/A_P)·S_f - D_f (p_N - p_P)/|d| |S_f|
```
with harmonic `D_f = (V/A_P)_f` matching both the pressure Laplacian and
[`rhie_chow_correction!`](@ref).  Boundary faces use the boundary
velocity directly.  By construction `div(φ)` equals the pressure-solve
residual, independent of the smoothness of the cell pressure gradient —
robust across porous interfaces with discontinuous pressure slope.
"""
function _correct_fluxes_hbya!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    phi_HbyA = compute_HbyA_flux(state, mesh)

    @inbounds for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            D_P = mesh.cell_volumes[P] / state.A_P[P]
            D_N = mesh.cell_volumes[N] / state.A_P[N]
            denom = w * D_N + (one(T) - w) * D_P
            D_f = denom > zero(T) ? D_P * D_N / denom : zero(T)

            _, d_mag = owner_neighbour_distance(mesh, f)
            snGrad = (state.p.internal[N] - state.p.internal[P]) / d_mag
            state.phi.values[f] = phi_HbyA[f] - D_f * snGrad * mesh.face_areas[f]
        else
            state.phi.values[f] = phi_HbyA[f]
        end
    end
    return nothing
end

# ── Boundary velocity update ────────────────────────────────────────

@doc """
    update_boundary_velocity!(state, bcs, mesh; t = 0)

Update boundary face velocity values according to the boundary condition
types:
- `FixedVelocityBC` → prescribed value
- `SpatialVelocityBC` → `bc.func(x_f)`
- `TimeDependentVelocityBC`, `UniformFixedValueBC` → `bc.func(t)`
- `CodedFixedValueBC` → `bc.func(x_f, t)` (same scalar per component)
- `NoSlipWallBC`, `WallFunctionBC` → zero (wall)
- `SlipWallBC`, `SymmetryBC` → owner cell value with the wall-normal
  component projected out, so the boundary face flux `U_b · S_f` is
  exactly zero (no mass leak through slip/symmetry planes)
- Others (FixedPressureBC, outlets) → copy owner cell value

# Arguments
- `state::IncompressibleState` — state (U.boundary modified in-place)
- `bcs::Dict{Symbol, <:AbstractBoundaryCondition}` — boundary conditions
- `mesh::UnstructuredFVMMesh` — mesh
- `t` — current simulation time for time-dependent BCs (default `0`)
"""
function update_boundary_velocity!(
        state::IncompressibleState{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        mesh::UnstructuredFVMMesh{Dim, T};
        t::T = zero(T),
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
        elseif bc isa TimeDependentVelocityBC
            state.U.boundary[i] = bc.func(t)
        elseif bc isa UniformFixedValueBC
            state.U.boundary[i] = bc.func(t)
        elseif bc isa CodedFixedValueBC
            x_f = face_center(mesh, f)
            v = T(bc.func(x_f, t))
            state.U.boundary[i] = SVector{Dim, T}(ntuple(_ -> v, Val(Dim)))
        elseif bc isa NoSlipWallBC || bc isa WallFunctionBC
            state.U.boundary[i] = zero(SVector{Dim, T})
        elseif bc isa InletOutletBC
            state.U.boundary[i] = bc.inlet_value
        elseif bc isa SlipWallBC || bc isa SymmetryBC
            # Tangential extrapolation: remove the face-normal component
            # so the boundary flux is exactly zero.
            S_f = face_normal_area(mesh, f)
            n_hat = S_f / mesh.face_areas[f]
            U_P = state.U.internal[P]
            state.U.boundary[i] = U_P - dot(U_P, n_hat) * n_hat
        else
            # FixedPressureBC, outlets, etc.: extrapolate from owner cell
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

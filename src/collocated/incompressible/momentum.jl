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
- `scheme` — convection interpolation scheme (`CONV_UPWIND`,
  `CONV_LINEAR`, `CONV_BLENDED`)
- `blend` — blending factor for `CONV_BLENDED` (0 = upwind, 1 = central)
- `nu_eff` — effective viscosity: scalar `T` or per-cell `Vector{T}` (default: `prob.nu`)
- `body_force` — per-cell body force vector (e.g. buoyancy), or `nothing`.
  Must be in kinematic units (force per unit mass), consistent with the
  rest of the equation.
- `t` — current simulation time, used to evaluate time-dependent BCs
- `rho_p` — optional per-cell density used to scale the pressure-gradient
  source to `-(1/ρ_c) ∇p V_c`.  Used by the compressible pressure-based
  solvers where `state.p` holds the ABSOLUTE pressure; `nothing` (default)
  keeps the incompressible convention where `p` is already kinematic.
  Callers passing `rho_p` here must pass the same vector to
  [`extract_momentum_operators!`](@ref) and [`correct_velocity!`](@ref).
- `porous_zones` — optional `Vector{PorousZone{T}}` of Darcy-Forchheimer
  zones.  The sink `-(ν K⁻¹ + ½ F |U|) U` (kinematic form: dynamic
  coefficients divided by density, so results are density-invariant at
  fixed ν) is added with IMPLICIT diagonal treatment — the tensor
  diagonal goes into `A[c,c]`, only the off-diagonal tensor remainder is
  explicit.  The Darcy term uses the molecular viscosity `prob.nu`.
- `mrf_zones` — optional `Vector{MRFZone{T}}` of rotating reference-frame
  zones.  Uses the OpenFOAM absolute-velocity MRF formulation: the solved
  `U` is the absolute velocity, convection inside zones uses the relative
  flux (see [`mrf_make_relative!`](@ref)), and the frame term enters as
  the explicit source `-(Ω × U) V_c` (per unit mass).  This is
  algebraically equivalent to the relative-velocity form with Coriolis
  `-2Ω×u_rel` and centrifugal `-Ω×(Ω×r)` sources.  Explicit treatment is
  used because `Ω×U` is skew-symmetric across velocity components: in a
  segregated per-component solve its diagonal contribution is identically
  zero, so there is nothing to treat implicitly; stability is provided by
  the outer under-relaxation (standard OpenFOAM `MRFZone` practice).
"""
function assemble_momentum!(
        eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        component::Int;
        dt::Union{Nothing, T} = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
        nu_eff::Union{T, Vector{T}} = prob.nu,
        body_force::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        t::T = zero(T),
        rho_p::Union{Nothing, Vector{T}} = nothing,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = nothing,
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Expand incompressible BCs to primitive velocity BCs at time t
    bcs_U = expand_bcs_velocity(prob.bcs, component; t = t)

    # Convection: div(phi * u_d)
    assemble_convection!(eq, state.phi, mesh, bcs_U; scheme = scheme, blend = blend)

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

    # Pressure gradient source: -dp/dx_d * V_c  (divided by ρ_c when the
    # pressure field is absolute — compressible/VOF callers pass rho_p).
    # Porous zones use the mobility-weighted face pressures; the
    # variable-density (rho_p) path uses density-weighted face pressures
    # so hydrostatic kinks at fluid interfaces do not smear into the
    # light-fluid cells (see the *_weighted_pressure_gradient helpers).
    grad_p = _momentum_pressure_gradient(state, mesh, rho_p, porous_zones)
    if rho_p === nothing
        for c in 1:nc
            eq.b[c] -= grad_p[c][component] * mesh.cell_volumes[c]
        end
    else
        for c in 1:nc
            eq.b[c] -= grad_p[c][component] * mesh.cell_volumes[c] / rho_p[c]
        end
    end

    # Body force (buoyancy, etc.)
    if body_force !== nothing
        for c in 1:nc
            eq.b[c] += body_force[c][component] * mesh.cell_volumes[c]
        end
    end

    # Darcy-Forchheimer porous sink (implicit diagonal + explicit
    # off-diagonal tensor remainder)
    if porous_zones !== nothing
        _assemble_porous_sink!(eq, state.U, porous_zones, prob.nu, component, mesh)
    end

    # MRF frame source -(Ω × U) V_c (explicit; see docstring)
    if mrf_zones !== nothing
        _assemble_mrf_source!(eq, state.U, mrf_zones, component, mesh)
    end

    return nothing
end

# ── Porous helpers ──────────────────────────────────────────────────

"""
    _use_porous_path(porous_zones) -> Bool

True when at least one non-empty porous zone is active — gates the
porous-consistent gradient / correction / flux variants.
"""
_use_porous_path(::Nothing) = false
function _use_porous_path(zones::Vector{PorousZone{T}}) where {T}
    return any(z -> !isempty(z.cell_indices), zones)
end

@doc """
    _porous_weighted_pressure_gradient(state, mesh) -> Vector{SVector}

Green-Gauss pressure gradient with RESISTANCE-WEIGHTED internal face
pressures.  With mobility `D_c = V_c / A_P[c]` and sub-cell resistances
`R_P = δ_P / D_P`, `R_N = δ_N / D_N` (`δ` = cell-center → face-center
distance), the face pressure is

```
    p_f = (R_N p_P + R_P p_N) / (R_P + R_N)
```

which is the exact interface pressure of the piecewise-linear 1D profile
carrying a uniform flux through cells of different mobility.  For
uniform `A_P` it reduces to the standard distance-weighted linear
interpolation.  At a porous interface (mobility jump of many orders of
magnitude) the plain linear interpolation smears the in-zone pressure
slope into the free cell, producing spurious momentum sources that
destabilize the outer loop; the resistance weighting removes the
artifact at its root (Mencinger & Žun-style pressure-weighted
interpolation).  Boundary faces use the boundary pressure values.
"""
function _porous_weighted_pressure_gradient(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    p = state.p
    bmap = build_boundary_map(p, mesh)
    grad = fill(zero(SVector{Dim, T}), nc)

    @inbounds for f in 1:nf
        S_f = face_normal_area(mesh, f)
        P = owner(mesh, f)
        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            x_f = face_center(mesh, f)
            delta_P = norm(x_f - cell_center(mesh, P))
            delta_N = norm(x_f - cell_center(mesh, N))
            D_P = mesh.cell_volumes[P] / state.A_P[P]
            D_N = mesh.cell_volumes[N] / state.A_P[N]
            R_P = delta_P / max(D_P, eps(T))
            R_N = delta_N / max(D_N, eps(T))
            p_f = (R_N * p.internal[P] + R_P * p.internal[N]) / max(R_P + R_N, eps(T))
        else
            p_f = p.boundary[bmap[f]]
        end
        grad[P] += p_f * S_f
        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            grad[N] -= p_f * S_f
        end
    end

    @inbounds for c in 1:nc
        grad[c] /= mesh.cell_volumes[c]
    end
    return grad
end

@doc """
    _rho_weighted_pressure_gradient(state, mesh, rho) -> Vector{SVector}

Green-Gauss pressure gradient with DENSITY-WEIGHTED internal face
pressures:

```
    p_f = (R_N p_P + R_P p_N) / (R_P + R_N),   R_i = δ_i ρ_i
```

(`δ` = cell-center → face-center distance).  In hydrostatic equilibrium
the pressure profile is piecewise linear with slope `ρ g` — its kink
sits on the interface face between fluids of different density.  This
weighting reproduces that kink EXACTLY, so the kinematic momentum
balance `-(1/ρ_c) ∇p + g` vanishes discretely cell-by-cell.  The plain
linearly-interpolated gradient smears the heavy-fluid slope into the
light-fluid cell, producing spurious interface accelerations of order
`(ρ_heavy/ρ_light) g / 2` (≈ 400 g for air-water) that destabilize
gravity-driven VOF within a few steps.  For uniform density it reduces
to the standard distance-weighted interpolation.  Boundary faces use
the boundary pressure values.
"""
function _rho_weighted_pressure_gradient(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        rho::Vector{T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    p = state.p
    bmap = build_boundary_map(p, mesh)
    grad = fill(zero(SVector{Dim, T}), nc)

    @inbounds for f in 1:nf
        S_f = face_normal_area(mesh, f)
        P = owner(mesh, f)
        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            x_f = face_center(mesh, f)
            R_P = norm(x_f - cell_center(mesh, P)) * rho[P]
            R_N = norm(x_f - cell_center(mesh, N)) * rho[N]
            p_f = (R_N * p.internal[P] + R_P * p.internal[N]) / max(R_P + R_N, eps(T))
            grad[P] += p_f * S_f
            grad[N] -= p_f * S_f
        else
            grad[P] += p.boundary[bmap[f]] * S_f
        end
    end

    @inbounds for c in 1:nc
        grad[c] /= mesh.cell_volumes[c]
    end
    return grad
end

"""
    _momentum_pressure_gradient(state, mesh, rho_p, porous_zones)

Select the pressure gradient consistent with the momentum source
convention: porous zones → mobility-weighted; `rho_p` (variable-density
kinematic form) → density-weighted; otherwise plain Green-Gauss.  Used
identically by `assemble_momentum!` and `extract_momentum_operators!`
so H stays exactly pressure-free.
"""
function _momentum_pressure_gradient(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        rho_p::Union{Nothing, Vector{T}},
        porous_zones,
    ) where {Dim, T}
    if _use_porous_path(porous_zones)
        return _porous_weighted_pressure_gradient(state, mesh)
    elseif rho_p !== nothing
        return _rho_weighted_pressure_gradient(state, mesh, rho_p)
    else
        return gradient(state.p, mesh)
    end
end

# ── Porous zone sink ────────────────────────────────────────────────

@doc """
    _assemble_porous_sink!(eq, U, zones, nu, component, mesh)

Add the kinematic Darcy-Forchheimer momentum sink to the component-`d`
momentum equation:

```
    S_d = -[ν K⁻¹ + ½ F |U|]_{d,:} · U
```

(the dynamic OpenFOAM form `-(μ K⁻¹ + ½ ρ F |U|) U` divided by density,
matching the kinematic momentum convention).  The tensor DIAGONAL entry
`R[d,d] V_c` is added IMPLICITLY to `A[c,c]` — essential for stability in
high-resistance zones — while the off-diagonal tensor remainder
`Σ_{e≠d} R[d,e] U_e V_c` is added explicitly to the RHS (zero for
isotropic / diagonal-tensor zones).
"""
function _assemble_porous_sink!(
        eq::CollocatedEquation{T},
        U::CollocatedVectorField{Dim, T},
        zones::Vector{PorousZone{T}},
        nu::T,
        component::Int,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    d = component
    for zone in zones
        @inbounds for c in zone.cell_indices
            U3 = _lift_to_3d(U.internal[c], T)
            u_mag = norm(U3)
            R = nu * zone.K_inv + T(0.5) * zone.F * u_mag
            V_c = mesh.cell_volumes[c]
            add_diag!(eq, c, R[d, d] * V_c)
            explicit_rem = zero(T)
            for e in 1:3
                e == d && continue
                explicit_rem += R[d, e] * U3[e]
            end
            eq.b[c] -= explicit_rem * V_c
        end
    end
    return nothing
end

# ── MRF frame source ────────────────────────────────────────────────

@doc """
    _assemble_mrf_source!(eq, U, zones, component, mesh)

Add the explicit absolute-velocity MRF frame source `-(Ω × U)_d V_c` for
every cell in each `MRFZone` (kinematic — per unit mass, no density).
2D velocities are lifted to 3D with `U_z = 0`, so a planar rotation about
`Ω = (0, 0, ω_z)` produces the exact in-plane source
`-ω_z (-U_y, U_x)` rather than the legacy `_cross` 2D stub behaviour.
"""
function _assemble_mrf_source!(
        eq::CollocatedEquation{T},
        U::CollocatedVectorField{Dim, T},
        zones::Vector{MRFZone{T}},
        component::Int,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    d = component
    for zone in zones
        omega = zone.omega
        @inbounds for c in zone.cells
            U3 = _lift_to_3d(U.internal[c], T)
            src3 = -cross(omega, U3)
            eq.b[c] += src3[d] * mesh.cell_volumes[c]
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

# Keyword Arguments
- `rho_p` — per-cell density used by [`assemble_momentum!`](@ref) to scale
  the pressure-gradient source; must match what was passed there so the
  same `(1/ρ_c) ∇p V_c` term is added back (compressible solvers only,
  default `nothing`).
- `porous_zones` — must match the zones passed to
  [`assemble_momentum!`](@ref): when active, the pressure add-back uses
  the same resistance-weighted gradient
  ([`_porous_weighted_pressure_gradient`](@ref)) that assembly
  subtracted, evaluated with the PRE-UPDATE `A_P` (i.e. the one that was
  in effect during assembly), so H stays exactly pressure-free.
"""
function extract_momentum_operators!(
        state::IncompressibleState{Dim, T},
        eqs::Vector{CollocatedEquation{T}},
        mesh::UnstructuredFVMMesh{Dim, T};
        rho_p::Union{Nothing, Vector{T}} = nothing,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    A = eqs[1].A  # Matrix structure is the same for all components
    pat = eqs[1].pattern

    # Evaluate the (possibly weighted) gradient BEFORE state.A_P is
    # overwritten below, so it matches the gradient used at assembly time
    # (the porous weighting reads state.A_P).
    grad_p = _momentum_pressure_gradient(state, mesh, rho_p, porous_zones)

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
    for c in 1:nc
        V_c = mesh.cell_volumes[c]
        inv_rho = rho_p === nothing ? one(T) : one(T) / rho_p[c]
        h = ntuple(Val(Dim)) do d
            eqs[d].b[c] + grad_p[c][d] * V_c * inv_rho
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

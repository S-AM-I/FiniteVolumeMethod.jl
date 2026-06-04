---
date: 2026-04-06
---

# Post-Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide derived field computations (vorticity, Q-criterion), wall surface metrics (shear stress, y+, heat flux, Nusselt), integrated force coefficients (Cd, Cl), and field sampling for the collocated solver.

**Architecture:** Four new files in `src/postprocessing/` wired into Layer 4. All functions operate on existing `CollocatedScalarField`/`CollocatedVectorField` and `UnstructuredFVMMesh`. Velocity gradients computed via Phase 0's `gradient()`. Wall quantities use owner-cell values with linear near-wall approximation. Sampling uses nearest-cell-center interpolation.

**Tech Stack:** Julia, LinearAlgebra (dot, cross, norm), StaticArrays (SVector), Phase 0 collocated operators (gradient, mesh helpers, field types).

---

## File Map

| File | Purpose | Creates/Modifies |
|------|---------|-----------------|
| `src/postprocessing/field_operations.jl` | Vorticity, Q-criterion, enstrophy, Courant number | Create |
| `src/postprocessing/wall_quantities.jl` | Wall shear stress, y+, heat flux, Nusselt | Create |
| `src/postprocessing/forces.jl` | Pressure/viscous forces, Cd/Cl coefficients | Create |
| `src/postprocessing/sampling.jl` | Line sampling, point interpolation | Create |
| `src/layers/extensions_tooling_output.jl` | Wire postprocessing includes | Modify |
| `src/FiniteVolumeMethod.jl` | Add exports | Modify |
| `test/postprocessing.jl` | All tests | Create |
| `test/runtests.jl` | Register test | Modify |
| `validation/manifest.toml` | Register feature | Modify |

---

### Task 1: Create field_operations.jl — Derived fields from velocity

**Files:**
- Create: `src/postprocessing/field_operations.jl`

- [ ] **Step 1: Write the field operations file**

```julia
# postprocessing/field_operations.jl — Derived field computations
#
# Computes vorticity, Q-criterion, enstrophy, and Courant number from
# velocity and flux fields on UnstructuredFVMMesh.

using LinearAlgebra: norm, dot, cross

# ── Velocity gradient helper ─────────────────────────────────────────

"""
    _compute_velocity_gradients(U, mesh) -> Vector{Vector{SVector{Dim, T}}}

Compute the gradient of each velocity component. Returns `grad_U` where
`grad_U[d][c]` is the gradient of component `d` at cell `c`.
"""
function _compute_velocity_gradients(
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    grad_U = Vector{Vector{SVector{Dim, T}}}(undef, Dim)

    for d in 1:Dim
        u_d = CollocatedScalarField(Symbol(:U, d), mesh; value = zero(T))
        for c in 1:nc
            u_d.internal[c] = U.internal[c][d]
        end
        for (i, f) in enumerate(u_d.boundary_face_indices)
            bi = findfirst(==(f), U.boundary_face_indices)
            if bi !== nothing
                u_d.boundary[i] = U.boundary[bi][d]
            end
        end
        grad_U[d] = gradient(u_d, mesh)
    end

    return grad_U
end

# ── Vorticity ────────────────────────────────────────────────────────

"""
    compute_vorticity(U, mesh) -> Vector{T}  (2D)

Compute the z-component of vorticity at each cell:
`ω_z = ∂v/∂x - ∂u/∂y`
"""
function compute_vorticity(
        U::CollocatedVectorField{2, T},
        mesh::UnstructuredFVMMesh{2, T},
    ) where {T}
    grad_U = _compute_velocity_gradients(U, mesh)
    nc = length(mesh.cell_volumes)
    omega = Vector{T}(undef, nc)
    for c in 1:nc
        dvdx = grad_U[2][c][1]  # ∂v/∂x
        dudy = grad_U[1][c][2]  # ∂u/∂y
        omega[c] = dvdx - dudy
    end
    return omega
end

"""
    compute_vorticity(U, mesh) -> Vector{SVector{3, T}}  (3D)

Compute the vorticity vector at each cell:
`ω = ∇ × U = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y)`
"""
function compute_vorticity(
        U::CollocatedVectorField{3, T},
        mesh::UnstructuredFVMMesh{3, T},
    ) where {T}
    grad_U = _compute_velocity_gradients(U, mesh)
    nc = length(mesh.cell_volumes)
    omega = Vector{SVector{3, T}}(undef, nc)
    for c in 1:nc
        dudy = grad_U[1][c][2]; dudz = grad_U[1][c][3]
        dvdx = grad_U[2][c][1]; dvdz = grad_U[2][c][3]
        dwdx = grad_U[3][c][1]; dwdy = grad_U[3][c][2]
        omega[c] = SVector{3, T}(dwdy - dvdz, dudz - dwdx, dvdx - dudy)
    end
    return omega
end

# ── Q-criterion ──────────────────────────────────────────────────────

"""
    compute_q_criterion(U, mesh) -> Vector{T}

Compute the Q-criterion at each cell:
`Q = 0.5 * (|Ω|² - |S|²)`

Positive Q identifies vortex cores.
"""
function compute_q_criterion(
        U::CollocatedVectorField{2, T},
        mesh::UnstructuredFVMMesh{2, T},
    ) where {T}
    grad_U = _compute_velocity_gradients(U, mesh)
    nc = length(mesh.cell_volumes)
    Q = Vector{T}(undef, nc)
    for c in 1:nc
        dudx = grad_U[1][c][1]; dudy = grad_U[1][c][2]
        dvdx = grad_U[2][c][1]; dvdy = grad_U[2][c][2]

        S_11 = dudx; S_22 = dvdy
        S_12 = T(0.5) * (dudy + dvdx)
        Omega_12 = T(0.5) * (dvdx - dudy)

        S_sq = S_11^2 + S_22^2 + T(2) * S_12^2
        Omega_sq = T(2) * Omega_12^2

        Q[c] = T(0.5) * (Omega_sq - S_sq)
    end
    return Q
end

function compute_q_criterion(
        U::CollocatedVectorField{3, T},
        mesh::UnstructuredFVMMesh{3, T},
    ) where {T}
    grad_U = _compute_velocity_gradients(U, mesh)
    nc = length(mesh.cell_volumes)
    Q = Vector{T}(undef, nc)
    for c in 1:nc
        dudx = grad_U[1][c][1]; dudy = grad_U[1][c][2]; dudz = grad_U[1][c][3]
        dvdx = grad_U[2][c][1]; dvdy = grad_U[2][c][2]; dvdz = grad_U[2][c][3]
        dwdx = grad_U[3][c][1]; dwdy = grad_U[3][c][2]; dwdz = grad_U[3][c][3]

        S_11 = dudx; S_22 = dvdy; S_33 = dwdz
        S_12 = T(0.5) * (dudy + dvdx)
        S_13 = T(0.5) * (dudz + dwdx)
        S_23 = T(0.5) * (dvdz + dwdy)
        S_sq = S_11^2 + S_22^2 + S_33^2 + T(2) * (S_12^2 + S_13^2 + S_23^2)

        O_12 = T(0.5) * (dvdx - dudy)
        O_13 = T(0.5) * (dwdx - dudz)
        O_23 = T(0.5) * (dwdy - dvdz)
        Omega_sq = T(2) * (O_12^2 + O_13^2 + O_23^2)

        Q[c] = T(0.5) * (Omega_sq - S_sq)
    end
    return Q
end

# ── Enstrophy ────────────────────────────────────────────────────────

"""
    compute_enstrophy(U, mesh) -> Vector{T}

Compute enstrophy `|ω|²` at each cell.
"""
function compute_enstrophy(
        U::CollocatedVectorField{2, T},
        mesh::UnstructuredFVMMesh{2, T},
    ) where {T}
    omega = compute_vorticity(U, mesh)
    return [w^2 for w in omega]
end

function compute_enstrophy(
        U::CollocatedVectorField{3, T},
        mesh::UnstructuredFVMMesh{3, T},
    ) where {T}
    omega = compute_vorticity(U, mesh)
    return [dot(w, w) for w in omega]
end

# ── Courant number ───────────────────────────────────────────────────

"""
    compute_courant_number(phi, mesh, dt) -> Vector{T}

Compute the Courant number per cell:
`Co = dt * sum_f |phi_f| / (2 * V_c)`

Requires `mesh.cell_faces` to be populated.
"""
function compute_courant_number(
        phi::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    Co = zeros(T, nc)
    mesh.cell_faces === nothing && error("cell_faces required for Courant number")

    for c in 1:nc
        flux_sum = zero(T)
        for f in mesh.cell_faces[c]
            flux_sum += abs(phi.values[f])
        end
        Co[c] = dt * flux_sum / (T(2) * mesh.cell_volumes[c])
    end

    return Co
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/postprocessing/field_operations.jl"); println("OK")'
```

---

### Task 2: Create wall_quantities.jl — Wall surface metrics

**Files:**
- Create: `src/postprocessing/wall_quantities.jl`

- [ ] **Step 1: Write wall quantities**

```julia
# postprocessing/wall_quantities.jl — Wall surface metrics
#
# Computes wall shear stress, y+, heat flux, and Nusselt number at
# named boundary patches on UnstructuredFVMMesh.

using LinearAlgebra: norm, dot

# ── Patch face helper ────────────────────────────────────────────────

"""
    _patch_faces(mesh, patch::Symbol) -> Vector{Int}

Return face indices belonging to boundary patch `patch`.
"""
function _patch_faces(mesh::UnstructuredFVMMesh{Dim, T}, patch::Symbol) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    faces = Int[]
    for f in 1:nf
        if !is_internal_face(mesh, f)
            tag = _face_tag(mesh, f)
            tag == patch && push!(faces, f)
        end
    end
    return faces
end

# ── Wall shear stress ────────────────────────────────────────────────

"""
    compute_wall_shear_stress(U, nu, mesh, patch) -> Vector{SVector{Dim, T}}

Compute wall shear stress at each face of boundary `patch`.

Uses the linear near-wall approximation:
`τ_w = ν * U_tangential / d`

where `d` is the distance from the cell center to the face center and
`U_tangential` is the velocity component parallel to the wall.
"""
function compute_wall_shear_stress(
        U::CollocatedVectorField{Dim, T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    tau = Vector{SVector{Dim, T}}(undef, length(faces))

    for (i, f) in enumerate(faces)
        P = owner(mesh, f)
        U_P = U.internal[P]
        x_P = cell_center(mesh, P)
        x_f = face_center(mesh, f)
        d_vec = x_f - x_P
        d = norm(d_vec)

        if d > zero(T)
            n_hat = d_vec / d
            U_normal = dot(U_P, n_hat) * n_hat
            U_tan = U_P - U_normal
            tau[i] = nu * U_tan / d
        else
            tau[i] = zero(SVector{Dim, T})
        end
    end

    return tau
end

# ── y+ ───────────────────────────────────────────────────────────────

"""
    compute_y_plus(U, nu, mesh, patch) -> Vector{T}

Compute y+ at each face of boundary `patch`.

`y+ = y * u_τ / ν` where `u_τ = sqrt(|τ_w|)` (ρ = 1 for incompressible).
"""
function compute_y_plus(
        U::CollocatedVectorField{Dim, T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    tau = compute_wall_shear_stress(U, nu, mesh, patch)
    yp = Vector{T}(undef, length(faces))

    for (i, f) in enumerate(faces)
        P = owner(mesh, f)
        x_P = cell_center(mesh, P)
        x_f = face_center(mesh, f)
        y = norm(x_f - x_P)

        tau_mag = norm(tau[i])
        u_tau = sqrt(tau_mag)
        yp[i] = nu > zero(T) ? y * u_tau / nu : zero(T)
    end

    return yp
end

# ── Wall heat flux ───────────────────────────────────────────────────

"""
    compute_wall_heat_flux(T_field, k, mesh, patch) -> Vector{T}

Compute wall heat flux at each face of boundary `patch`:
`q_w = -k * (T_wall - T_cell) / d`

Positive q means heat flows out of the domain.
"""
function compute_wall_heat_flux(
        T_field::CollocatedScalarField{T},
        k::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    pbmap = build_boundary_map(T_field)
    q = Vector{T}(undef, length(faces))

    for (i, f) in enumerate(faces)
        P = owner(mesh, f)
        T_cell = T_field.internal[P]
        T_wall = T_field.boundary[pbmap[f]]
        x_P = cell_center(mesh, P)
        x_f = face_center(mesh, f)
        d = norm(x_f - x_P)

        q[i] = d > zero(T) ? -k * (T_wall - T_cell) / d : zero(T)
    end

    return q
end

# ── Nusselt number ───────────────────────────────────────────────────

"""
    compute_nusselt_number(T_field, k, mesh, patch; T_ref, L_ref) -> Vector{T}

Compute Nusselt number at each face of boundary `patch`:
`Nu = q_w * L_ref / (k * (T_wall - T_ref))`
"""
function compute_nusselt_number(
        T_field::CollocatedScalarField{T},
        k::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol;
        T_ref::T,
        L_ref::T,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    q_w = compute_wall_heat_flux(T_field, k, mesh, patch)
    pbmap = build_boundary_map(T_field)
    Nu = Vector{T}(undef, length(faces))

    for (i, f) in enumerate(faces)
        T_wall = T_field.boundary[pbmap[f]]
        dT = T_wall - T_ref
        if abs(dT) > eps(T) && k > zero(T)
            Nu[i] = abs(q_w[i]) * L_ref / (k * abs(dT))
        else
            Nu[i] = zero(T)
        end
    end

    return Nu
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/postprocessing/field_operations.jl"); include("src/postprocessing/wall_quantities.jl"); println("OK")'
```

---

### Task 3: Create forces.jl — Integrated force coefficients

**Files:**
- Create: `src/postprocessing/forces.jl`

- [ ] **Step 1: Write forces computation**

```julia
# postprocessing/forces.jl — Integrated forces and coefficients
#
# Computes pressure and viscous forces on boundary patches, and
# aerodynamic coefficients (Cd, Cl) from the integrated forces.

using LinearAlgebra: norm, dot

"""
    compute_forces(p, U, nu, mesh, patch)

Compute pressure and viscous forces on boundary `patch`.

Pressure force: `F_p = -sum_f p_f * S_f` (outward-pointing)
Viscous force: `F_v = sum_f τ_w_f * A_f`

Returns `(pressure = SVector, viscous = SVector)`.
"""
function compute_forces(
        p::CollocatedScalarField{T},
        U::CollocatedVectorField{Dim, T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    faces = _patch_faces(mesh, patch)
    pbmap = build_boundary_map(p)
    tau_w = compute_wall_shear_stress(U, nu, mesh, patch)

    F_pressure = zero(SVector{Dim, T})
    F_viscous = zero(SVector{Dim, T})

    for (i, f) in enumerate(faces)
        S_f = face_normal_area(mesh, f)
        p_f = p.boundary[pbmap[f]]
        A_f = mesh.face_areas[f]

        F_pressure = F_pressure - p_f * S_f
        F_viscous = F_viscous + tau_w[i] * A_f
    end

    return (pressure = F_pressure, viscous = F_viscous)
end

"""
    force_coefficients(pressure_force, viscous_force; rho_ref, U_ref, A_ref,
        drag_direction, lift_direction)

Compute aerodynamic force coefficients from integrated forces.

- `Cd = (F_total · drag_dir) / (q * A_ref)` where `q = 0.5 * ρ * U²`
- `Cl = (F_total · lift_dir) / (q * A_ref)`
- `Cd_pressure`, `Cd_viscous` for separate contributions

Returns `(Cd, Cl, Cd_pressure, Cd_viscous)` named tuple.
"""
function force_coefficients(
        pressure_force::SVector{Dim, T},
        viscous_force::SVector{Dim, T};
        rho_ref::T,
        U_ref::T,
        A_ref::T,
        drag_direction::SVector{Dim, T} = _default_drag_dir(Val(Dim), T),
        lift_direction::SVector{Dim, T} = _default_lift_dir(Val(Dim), T),
    ) where {Dim, T}
    q = T(0.5) * rho_ref * U_ref^2
    qA = q * A_ref

    F_total = pressure_force + viscous_force

    Cd = qA > zero(T) ? dot(F_total, drag_direction) / qA : zero(T)
    Cl = qA > zero(T) ? dot(F_total, lift_direction) / qA : zero(T)
    Cd_p = qA > zero(T) ? dot(pressure_force, drag_direction) / qA : zero(T)
    Cd_v = qA > zero(T) ? dot(viscous_force, drag_direction) / qA : zero(T)

    return (Cd = Cd, Cl = Cl, Cd_pressure = Cd_p, Cd_viscous = Cd_v)
end

_default_drag_dir(::Val{2}, ::Type{T}) where {T} = SVector{2, T}(one(T), zero(T))
_default_drag_dir(::Val{3}, ::Type{T}) where {T} = SVector{3, T}(one(T), zero(T), zero(T))
_default_lift_dir(::Val{2}, ::Type{T}) where {T} = SVector{2, T}(zero(T), one(T))
_default_lift_dir(::Val{3}, ::Type{T}) where {T} = SVector{3, T}(zero(T), one(T), zero(T))
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["field_operations", "wall_quantities", "forces"]; include("src/postprocessing/$f.jl"); end; println("OK")'
```

---

### Task 4: Create sampling.jl — Field interpolation along lines

**Files:**
- Create: `src/postprocessing/sampling.jl`

- [ ] **Step 1: Write sampling functions**

```julia
# postprocessing/sampling.jl — Field sampling along lines and at points
#
# Nearest-cell-center interpolation (0th order) for extracting field
# values along lines or at arbitrary points.

using LinearAlgebra: norm

# ── Nearest cell lookup ──────────────────────────────────────────────

"""
    _find_nearest_cell(mesh, point) -> Int

Find the cell whose center is nearest to `point` (brute force).
"""
function _find_nearest_cell(
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    best_cell = 1
    best_dist = T(Inf)

    for c in 1:nc
        x_c = cell_center(mesh, c)
        d = norm(point - x_c)
        if d < best_dist
            best_dist = d
            best_cell = c
        end
    end

    return best_cell
end

# ── Point sampling ───────────────────────────────────────────────────

"""
    sample_field_at_point(field, mesh, point) -> T

Sample a scalar field at `point` using nearest-cell-center interpolation.
"""
function sample_field_at_point(
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T},
    ) where {Dim, T}
    c = _find_nearest_cell(mesh, point)
    return field.internal[c]
end

"""
    sample_field_at_point(field, mesh, point) -> SVector{Dim, T}

Sample a vector field at `point` using nearest-cell-center interpolation.
"""
function sample_field_at_point(
        field::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T},
    ) where {Dim, T}
    c = _find_nearest_cell(mesh, point)
    return field.internal[c]
end

# ── Line sampling ────────────────────────────────────────────────────

"""
    sample_line(field, mesh, p1, p2, n_points)

Sample a scalar field at `n_points` evenly spaced along the line from
`p1` to `p2`.

Returns `(positions, distances, values)` where:
- `positions::Vector{SVector{Dim, T}}` — sample point coordinates
- `distances::Vector{T}` — distance along the line from `p1`
- `values::Vector{T}` — field values at each sample point
"""
function sample_line(
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        p1::SVector{Dim, T},
        p2::SVector{Dim, T},
        n_points::Int,
    ) where {Dim, T}
    positions = Vector{SVector{Dim, T}}(undef, n_points)
    distances = Vector{T}(undef, n_points)
    values = Vector{T}(undef, n_points)

    L = norm(p2 - p1)
    dir = L > zero(T) ? (p2 - p1) / L : zero(SVector{Dim, T})

    for i in 1:n_points
        t = (i - 1) / max(n_points - 1, 1)
        pt = p1 + t * (p2 - p1)
        positions[i] = pt
        distances[i] = t * L
        values[i] = sample_field_at_point(field, mesh, pt)
    end

    return (positions = positions, distances = distances, values = values)
end

"""
    sample_line(field, mesh, p1, p2, n_points)

Sample a vector field along a line. Returns `values::Vector{SVector{Dim, T}}`.
"""
function sample_line(
        field::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        p1::SVector{Dim, T},
        p2::SVector{Dim, T},
        n_points::Int,
    ) where {Dim, T}
    positions = Vector{SVector{Dim, T}}(undef, n_points)
    distances = Vector{T}(undef, n_points)
    values = Vector{SVector{Dim, T}}(undef, n_points)

    L = norm(p2 - p1)

    for i in 1:n_points
        t = (i - 1) / max(n_points - 1, 1)
        pt = p1 + t * (p2 - p1)
        positions[i] = pt
        distances[i] = t * L
        values[i] = sample_field_at_point(field, mesh, pt)
    end

    return (positions = positions, distances = distances, values = values)
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["field_operations", "wall_quantities", "forces", "sampling"]; include("src/postprocessing/$f.jl"); end; println("OK")'
```

---

### Task 5: Wire into module — Layer 4 includes + exports

**Files:**
- Modify: `src/layers/extensions_tooling_output.jl`
- Modify: `src/FiniteVolumeMethod.jl`

- [ ] **Step 1: Add includes to Layer 4**

Append to `src/layers/extensions_tooling_output.jl` after `include("../capabilities.jl")`:

```julia
# Post-Processing (Phase 12)
include("../postprocessing/field_operations.jl")
include("../postprocessing/wall_quantities.jl")
include("../postprocessing/forces.jl")
include("../postprocessing/sampling.jl")
```

- [ ] **Step 2: Add exports**

Add a new export block in `src/FiniteVolumeMethod.jl` after the Phase 3 thermal exports and before `export FVMGeometry`:

```julia
# --- Post-Processing (Phase 12) ---
export
    # Field operations
    compute_vorticity,
    compute_q_criterion,
    compute_enstrophy,
    compute_courant_number,
    # Wall quantities
    compute_wall_shear_stress,
    compute_y_plus,
    compute_wall_heat_flux,
    compute_nusselt_number,
    # Forces
    compute_forces,
    force_coefficients,
    # Sampling
    sample_line,
    sample_field_at_point
```

- [ ] **Step 3: Verify module loads**

```bash
julia --project -e 'using FiniteVolumeMethod; println("Phase 12: ", compute_vorticity)'
```

- [ ] **Step 4: Commit**

```bash
git add src/postprocessing/ src/layers/extensions_tooling_output.jl src/FiniteVolumeMethod.jl
git commit -m "feat: add post-processing (vorticity, wall quantities, forces, sampling) — Phase 12"
```

---

### Task 6: Write tests

**Files:**
- Create: `test/postprocessing.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the test file**

Create `test/postprocessing.jl` with the `build_cartesian_unstructured_mesh` helper copied from `test/incompressible.jl`. Include these tests:

1. **Vorticity of rigid body rotation** — Set U = (-ωy, ωx) with ω=1 on a 4x4 mesh. Verify ω_z ≈ 2.0 at interior cells (boundary cells may differ due to gradient BCs).
2. **Q-criterion zero for uniform flow** — U = (1, 0) everywhere → Q = 0.
3. **Enstrophy positive** — Enstrophy of a non-uniform field should be > 0.
4. **Courant number** — Uniform flux on uniform mesh with known dt → verify Co is constant and matches analytical value.
5. **Wall shear stress direction** — Channel flow U = (U_x, 0), walls at top/bottom → τ_w should be in x-direction.
6. **y+ finite** — Verify y+ is non-negative and finite for a simple flow.
7. **Wall heat flux sign** — Hot wall (T_wall > T_cell) → q_w should be negative (heat into domain).
8. **Force computation** — Set pressure field to constant, compute forces on a patch → pressure force = -p * total_face_area_vector.
9. **Force coefficients arithmetic** — Known force, known dynamic pressure → verify Cd = F_drag / (q*A).
10. **sample_line** — Sample a linearly varying scalar field along a line → values should be linear.
11. **sample_field_at_point** — Sample at cell center → should return exact cell value.

- [ ] **Step 2: Register test**

Add `safe_include("postprocessing.jl")` to `test/runtests.jl` after the mesh_io test.

- [ ] **Step 3: Run tests**

```bash
julia --project=test test/postprocessing.jl
```

- [ ] **Step 4: Run Runic**

```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "src/postprocessing/"])'
julia --project -e 'using Runic; Runic.main(["--inplace", "test/postprocessing.jl"])'
```

- [ ] **Step 5: Commit**

```bash
git add test/postprocessing.jl test/runtests.jl
git commit -m "test: add post-processing test suite"
```

---

### Task 7: Register in validation manifest + final verification

**Files:**
- Modify: `validation/manifest.toml`

- [ ] **Step 1: Add postprocessing feature**

Append to `validation/manifest.toml`:

```toml
# ── Phase 12: Post-Processing ──────────────────────────────────────

[[features]]
feature = "postprocessing"
maturity = "experimental"
validation = "smoke_tested"
role = "research_support_tooling"
solver_family = "collocated"
precision_policy = "float64_cpu_reference"
random_seed_policy = "deterministic"
backend_policy = "cpu_reference"
required_ladder_stages = ["verification"]
summary = "Derived field operations (vorticity, Q-criterion), wall quantities (shear stress, y+, heat flux, Nusselt), force coefficients, and line sampling."
limitations = [
  "Experimental — 0th-order nearest-cell interpolation for sampling; higher-order deferred.",
  "Wall quantities use linear near-wall approximation; wall functions not integrated.",
  "Force coefficients assume ρ = 1 for viscous contribution.",
]
```

- [ ] **Step 2: Verify all exports**

```bash
julia --project -e '
using FiniteVolumeMethod
for sym in [:compute_vorticity, :compute_q_criterion, :compute_enstrophy,
            :compute_courant_number, :compute_wall_shear_stress, :compute_y_plus,
            :compute_wall_heat_flux, :compute_nusselt_number, :compute_forces,
            :force_coefficients, :sample_line, :sample_field_at_point]
    @assert isdefined(FiniteVolumeMethod, sym) "Missing export: $sym"
end
println("All Phase 12 exports verified")
'
```

- [ ] **Step 3: Run tests + regression**

```bash
julia --project=test test/postprocessing.jl
julia --project=test test/incompressible.jl
```

- [ ] **Step 4: Runic check**

```bash
julia --project -e 'using Runic; Runic.main(["--check", "src/postprocessing/"])'
```

- [ ] **Step 5: Commit**

```bash
git add validation/manifest.toml
git commit -m "feat: register postprocessing in validation manifest"
```

# VOF Multiphase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two-phase immiscible flow simulation using Volume of Fluid: alpha transport with interface compression, boundedness limiter, mixture property blending, CSF surface tension, and a transient VOF solver wrapper.

**Architecture:** Six new files in `src/multiphase/` wired into Layer 2 after thermal. Alpha is solved as an explicit scalar transport with compression. Mixture density/viscosity replace constants in momentum. Gravity uses full variable density (not Boussinesq). Pressure equation uses density-weighted diffusivity assembled inline. Surface tension via CSF adds a body force to momentum.

**Tech Stack:** Julia, LinearAlgebra (dot, norm), StaticArrays (SVector), Phase 0 collocated operators (assemble_convection!, assemble_laplacian!, assemble_ddt_euler!, gradient), Phase 1 incompressible solvers (assemble_momentum!, pressure assembly, velocity correction).

---

## File Map

| File | Purpose | Creates/Modifies |
|------|---------|-----------------|
| `src/multiphase/types.jl` | TwoPhaseProperties, VOFState | Create |
| `src/multiphase/alpha_transport.jl` | assemble_alpha!, compute_compression_flux | Create |
| `src/multiphase/boundedness.jl` | clip_alpha! | Create |
| `src/multiphase/mixture.jl` | update_mixture_properties! | Create |
| `src/multiphase/surface_tension.jl` | compute_curvature, compute_surface_tension_force | Create |
| `src/multiphase/solvers.jl` | solve_vof with PISO/PIMPLE time stepping | Create |
| `src/layers/discretization_assembly_kernels.jl` | Wire multiphase includes | Modify |
| `src/FiniteVolumeMethod.jl` | Add exports | Modify |
| `test/multiphase_vof.jl` | All tests | Create |
| `test/runtests.jl` | Register test | Modify |
| `validation/manifest.toml` | Register feature | Modify |

---

### Task 1: Create types.jl — Two-phase properties and VOF state

**Files:**
- Create: `src/multiphase/types.jl`

- [ ] **Step 1: Create directory and write types**

```bash
mkdir -p src/multiphase
```

Write `src/multiphase/types.jl`:

```julia
# multiphase/types.jl — Core types for Volume of Fluid multiphase
#
# Defines fluid property pairs for two immiscible phases and the
# mutable VOF state (volume fraction + mixture properties).

"""
    TwoPhaseProperties{T}

Physical properties for a two-phase immiscible flow system.

# Fields
- `rho1::T` — density of fluid 1 (α = 1), e.g., water = 1000 kg/m³
- `rho2::T` — density of fluid 2 (α = 0), e.g., air = 1.225 kg/m³
- `mu1::T` — dynamic viscosity of fluid 1, e.g., water = 1e-3 Pa·s
- `mu2::T` — dynamic viscosity of fluid 2, e.g., air = 1.8e-5 Pa·s
- `sigma::T` — surface tension coefficient [N/m] (0 = disabled)
"""
struct TwoPhaseProperties{T}
    rho1::T
    rho2::T
    mu1::T
    mu2::T
    sigma::T
end

"""
    TwoPhaseProperties(; rho1, rho2, mu1, mu2, sigma)

Construct two-phase properties with keyword defaults for water/air at 20°C.
"""
function TwoPhaseProperties(;
        rho1::Real = 1000.0,
        rho2::Real = 1.225,
        mu1::Real = 1.0e-3,
        mu2::Real = 1.8e-5,
        sigma::Real = 0.072,
    )
    T = promote_type(typeof(rho1), typeof(rho2), typeof(mu1), typeof(mu2), typeof(sigma))
    return TwoPhaseProperties{T}(T(rho1), T(rho2), T(mu1), T(mu2), T(sigma))
end

"""Check if surface tension is active."""
has_surface_tension(props::TwoPhaseProperties) = props.sigma > 0

"""
    VOFState{T}

Mutable state for VOF multiphase simulation.

# Fields
- `alpha::CollocatedScalarField{T}` — volume fraction [0, 1]
- `rho::Vector{T}` — mixture density per cell
- `mu::Vector{T}` — mixture dynamic viscosity per cell
"""
mutable struct VOFState{T}
    alpha::CollocatedScalarField{T}
    rho::Vector{T}
    mu::Vector{T}
end

"""
    VOFState(mesh; alpha_init = 0.0)

Construct a VOF state with uniform initial volume fraction.
"""
function VOFState(
        mesh::UnstructuredFVMMesh{Dim, T};
        alpha_init::Real = 0.0,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    alpha = CollocatedScalarField(:alpha, mesh; value = T(alpha_init))
    rho = fill(T(1.0), nc)   # placeholder, updated by update_mixture_properties!
    mu = fill(T(1.0e-3), nc)  # placeholder
    return VOFState{T}(alpha, rho, mu)
end

"""
    VOFState(mesh, alpha_func::Function, props::TwoPhaseProperties)

Construct a VOF state with spatially varying initial alpha defined by
`alpha_func(x::SVector) -> T`. Also initializes mixture properties.
"""
function VOFState(
        mesh::UnstructuredFVMMesh{Dim, T},
        alpha_func::Function,
        props::TwoPhaseProperties{T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        x_c = cell_center(mesh, c)
        alpha.internal[c] = clamp(alpha_func(x_c), zero(T), one(T))
    end
    # Set boundary values
    for (i, f) in enumerate(alpha.boundary_face_indices)
        x_f = face_center(mesh, f)
        alpha.boundary[i] = clamp(alpha_func(x_f), zero(T), one(T))
    end
    rho = Vector{T}(undef, nc)
    mu = Vector{T}(undef, nc)
    state = VOFState{T}(alpha, rho, mu)
    update_mixture_properties!(state, props)
    return state
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/multiphase/types.jl"); println("OK")'
```

---

### Task 2: Create mixture.jl + boundedness.jl — Property blending and limiter

**Files:**
- Create: `src/multiphase/mixture.jl`
- Create: `src/multiphase/boundedness.jl`

- [ ] **Step 1: Write mixture property update**

Write `src/multiphase/mixture.jl`:

```julia
# multiphase/mixture.jl — Mixture property computation from volume fraction
#
# Blends density and viscosity linearly by alpha for two-phase VOF.

"""
    update_mixture_properties!(vof_state, props)

Update mixture density and viscosity from current volume fraction:
- `ρ[c] = α[c]·ρ₁ + (1 - α[c])·ρ₂`
- `μ[c] = α[c]·μ₁ + (1 - α[c])·μ₂`
"""
function update_mixture_properties!(
        vof_state::VOFState{T},
        props::TwoPhaseProperties{T},
    ) where {T}
    nc = length(vof_state.rho)
    for c in 1:nc
        a = vof_state.alpha.internal[c]
        vof_state.rho[c] = a * props.rho1 + (one(T) - a) * props.rho2
        vof_state.mu[c] = a * props.mu1 + (one(T) - a) * props.mu2
    end
    return nothing
end
```

- [ ] **Step 2: Write boundedness limiter**

Write `src/multiphase/boundedness.jl`:

```julia
# multiphase/boundedness.jl — Boundedness limiter for volume fraction
#
# Clips alpha to [0, 1] and redistributes the error to maintain
# global conservation of the volume fraction field.

"""
    clip_alpha!(alpha, mesh)

Clip volume fraction to [0, 1] bounds with conservative redistribution.

After clipping, any global excess or deficit is distributed proportionally
among cells that are not at the bounds, preserving total α·V.
"""
function clip_alpha!(
        alpha::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    # Total alpha*volume before clipping
    total_before = zero(T)
    for c in 1:nc
        total_before += alpha.internal[c] * mesh.cell_volumes[c]
    end

    # Clip
    for c in 1:nc
        alpha.internal[c] = clamp(alpha.internal[c], zero(T), one(T))
    end

    # Total after clipping
    total_after = zero(T)
    for c in 1:nc
        total_after += alpha.internal[c] * mesh.cell_volumes[c]
    end

    # Redistribute difference proportionally to maintain conservation
    diff = total_before - total_after
    if abs(diff) > eps(T) * abs(total_before)
        # Find cells that can absorb the correction
        total_correctable_volume = zero(T)
        for c in 1:nc
            a = alpha.internal[c]
            if diff > 0 && a < one(T)
                total_correctable_volume += mesh.cell_volumes[c]
            elseif diff < 0 && a > zero(T)
                total_correctable_volume += mesh.cell_volumes[c]
            end
        end

        if total_correctable_volume > eps(T)
            correction = diff / total_correctable_volume
            for c in 1:nc
                a = alpha.internal[c]
                if diff > 0 && a < one(T)
                    alpha.internal[c] = min(a + correction, one(T))
                elseif diff < 0 && a > zero(T)
                    alpha.internal[c] = max(a + correction, zero(T))
                end
            end
        end
    end

    return nothing
end
```

- [ ] **Step 3: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/multiphase/types.jl"); include("src/multiphase/mixture.jl"); include("src/multiphase/boundedness.jl"); println("OK")'
```

---

### Task 3: Create alpha_transport.jl — Alpha equation + compression

**Files:**
- Create: `src/multiphase/alpha_transport.jl`

- [ ] **Step 1: Write alpha transport and compression**

Write `src/multiphase/alpha_transport.jl`:

```julia
# multiphase/alpha_transport.jl — Volume fraction transport equation
#
# Assembles the alpha advection equation with optional interface
# compression term for maintaining interface sharpness.

"""
    compute_compression_flux(
        alpha, phi, mesh; C_alpha = 1.0,
    ) -> Vector{T}

Compute the interface compression flux per face.

`phi_c_f = C_alpha · |phi_f| · (n_interface · S_f) / |S_f|`

where `n_interface = ∇α/|∇α|` is the interface normal direction.
"""
function compute_compression_flux(
        alpha::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        C_alpha::T = one(T),
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    phi_c = zeros(T, nf)

    # Compute interface normal (gradient of alpha)
    grad_alpha = gradient(alpha, mesh)

    for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            # Interpolate interface normal to face
            n_P = grad_alpha[P]
            n_N = grad_alpha[N]
            n_f = w * n_P + (one(T) - w) * n_N
            n_mag = norm(n_f)

            if n_mag > eps(T)
                n_hat = n_f / n_mag
                S_f = face_normal_area(mesh, f)
                S_mag = mesh.face_areas[f]

                # Compression flux: aligned with interface normal
                phi_c[f] = C_alpha * abs(phi.values[f]) * dot(n_hat, S_f) / max(S_mag, eps(T))
            end
        end
    end

    return phi_c
end

"""
    assemble_alpha!(
        eq, alpha, phi, mesh, bcs_alpha;
        dt, C_alpha = 1.0,
    )

Assemble the volume fraction transport equation with interface compression.

The equation is:
`∂α/∂t + div(phi · α) + div(phi_c · α(1-α)) = 0`

The standard convection `div(phi · α)` is assembled implicitly.
The compression term `div(phi_c · α(1-α))` is added explicitly to the RHS.
"""
function assemble_alpha!(
        eq::CollocatedEquation{T},
        alpha::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_alpha::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::T,
        C_alpha::T = one(T),
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)

    # Standard convection: div(phi · alpha)
    assemble_convection!(eq, phi, mesh, bcs_alpha)

    # Temporal: ddt(alpha)
    assemble_ddt_euler!(eq, one(T), alpha.internal, mesh, dt)

    # Interface compression (explicit source)
    if C_alpha > zero(T)
        phi_c = compute_compression_flux(alpha, phi, mesh; C_alpha = C_alpha)
        pbmap = build_boundary_map(alpha)

        for f in 1:nf
            if is_internal_face(mesh, f)
                P = owner(mesh, f)
                N = neighbour(mesh, f)
                w = face_weight(mesh, f)

                # Interpolate alpha to face
                alpha_f = w * alpha.internal[P] + (one(T) - w) * alpha.internal[N]
                compression = phi_c[f] * alpha_f * (one(T) - alpha_f)

                eq.b[P] -= compression
                eq.b[N] += compression
            end
        end
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["types", "mixture", "boundedness", "alpha_transport"]; include("src/multiphase/$f.jl"); end; println("OK")'
```

---

### Task 4: Create surface_tension.jl — CSF model

**Files:**
- Create: `src/multiphase/surface_tension.jl`

- [ ] **Step 1: Write curvature and surface tension force**

Write `src/multiphase/surface_tension.jl`:

```julia
# multiphase/surface_tension.jl — Continuum Surface Force (CSF) model
#
# Computes interface curvature from the volume fraction gradient and
# produces a body force F_st = σ · κ · ∇α for the momentum equation.

using LinearAlgebra: norm, dot

"""
    compute_curvature(alpha, mesh) -> Vector{T}

Compute the interface curvature `κ = -div(∇α / |∇α|)` per cell.

Steps:
1. Compute `∇α` via Green-Gauss gradient
2. Normalize to get interface normal `n̂ = ∇α / |∇α|`
3. Compute `div(n̂)` via face summation
4. `κ = -div(n̂)`
"""
function compute_curvature(
        alpha::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Step 1: gradient of alpha
    grad_alpha = gradient(alpha, mesh)

    # Step 2: normalize to get interface normal per cell
    n_hat = Vector{SVector{Dim, T}}(undef, nc)
    for c in 1:nc
        g_mag = norm(grad_alpha[c])
        if g_mag > T(1.0e-12)
            n_hat[c] = grad_alpha[c] / g_mag
        else
            n_hat[c] = zero(SVector{Dim, T})
        end
    end

    # Step 3: div(n_hat) via face summation
    div_n = zeros(T, nc)
    for f in 1:nf
        P = owner(mesh, f)
        S_f = face_normal_area(mesh, f)

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            # Interpolate n_hat to face
            n_f = w * n_hat[P] + (one(T) - w) * n_hat[N]
            flux = dot(n_f, S_f)

            div_n[P] += flux
            div_n[N] -= flux
        else
            # Boundary: use owner value
            flux = dot(n_hat[P], S_f)
            div_n[P] += flux
        end
    end

    # Normalize by cell volume
    kappa = Vector{T}(undef, nc)
    for c in 1:nc
        div_n[c] /= mesh.cell_volumes[c]
        kappa[c] = -div_n[c]
    end

    return kappa
end

"""
    compute_surface_tension_force(alpha, props, mesh) -> Union{Nothing, Vector{SVector{Dim, T}}}

Compute the CSF surface tension body force: `F_st = σ · κ · ∇α`.

Returns `nothing` when `sigma == 0` (surface tension disabled).
"""
function compute_surface_tension_force(
        alpha::CollocatedScalarField{T},
        props::TwoPhaseProperties{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    if !has_surface_tension(props)
        return nothing
    end

    nc = length(mesh.cell_volumes)
    grad_alpha = gradient(alpha, mesh)
    kappa = compute_curvature(alpha, mesh)

    force = Vector{SVector{Dim, T}}(undef, nc)
    for c in 1:nc
        force[c] = props.sigma * kappa[c] * grad_alpha[c]
    end

    return force
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["types", "mixture", "boundedness", "alpha_transport", "surface_tension"]; include("src/multiphase/$f.jl"); end; println("OK")'
```

---

### Task 5: Create solvers.jl — VOF transient solver

**Files:**
- Create: `src/multiphase/solvers.jl`

- [ ] **Step 1: Write the VOF solver**

Write `src/multiphase/solvers.jl`:

```julia
# multiphase/solvers.jl — VOF transient solver wrapper
#
# Time-stepping loop: alpha transport → boundedness → mixture update →
# body forces → PISO/PIMPLE momentum+pressure with variable density.

using Printf: @sprintf

"""
    solve_vof(mesh, props, bcs_U, bcs_p, bcs_alpha, tspan, dt; kwargs...)

Solve a transient two-phase VOF flow problem.

Each time step:
1. Solve alpha transport with interface compression
2. Apply boundedness limiter
3. Update mixture properties (ρ, μ)
4. Compute body forces (gravity + surface tension)
5. PISO/PIMPLE step with variable density

# Arguments
- `mesh` — `UnstructuredFVMMesh`
- `props` — `TwoPhaseProperties`
- `bcs_U` — velocity boundary conditions
- `bcs_p` — pressure boundary conditions
- `bcs_alpha` — volume fraction boundary conditions
- `tspan` — `(t_start, t_end)`
- `dt` — time step size

# Keyword Arguments
- `alpha_init` — initial alpha: constant `T` or function `f(x) -> T`
- `g` — gravity vector (default: zero)
- `C_alpha` — compression coefficient (default: 1.0)
- `algorithm` — `PISO()` or `PIMPLE()` (default: `PISO()`)
- `linear_solver` — LinearSolve.jl algorithm
- `save_every` — save interval
- `verbose` — print progress

# Returns
`(SolveResult, VOFState)` tuple.
"""
function solve_vof(
        mesh::UnstructuredFVMMesh{Dim, T},
        props::TwoPhaseProperties{T},
        bcs_U::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_p::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_alpha::Dict{Symbol, <:AbstractBoundaryCondition},
        tspan::Tuple{T, T},
        dt::T;
        alpha_init::Union{T, Function} = zero(T),
        g::SVector{Dim, T} = zero(SVector{Dim, T}),
        C_alpha::T = one(T),
        algorithm::AbstractPVCoupling = PISO(),
        linear_solver = nothing,
        save_every::Int = 1,
        verbose::Bool = false,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    # Create incompressible problem (density=1 placeholder, actual rho handled via body force)
    prob = IncompressibleProblem(mesh, bcs_U, algorithm; nu = T(1.0e-3), density = one(T))

    # Initialize flow state
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, bcs_U, mesh)
    update_boundary_pressure!(state, bcs_p, mesh)

    # Initialize VOF state
    if alpha_init isa Function
        vof_state = VOFState(mesh, alpha_init, props)
    else
        vof_state = VOFState(mesh; alpha_init = alpha_init)
        update_mixture_properties!(vof_state, props)
    end

    # Residual tracking
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    t_start, t_end = tspan
    t = t_start
    n_steps = 0

    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)

        # ── 1. Alpha transport ──────────────────────────────────
        alpha_eq = CollocatedEquation(mesh)
        assemble_alpha!(
            alpha_eq, vof_state.alpha, state.phi, mesh, bcs_alpha;
            dt = dt_actual, C_alpha = C_alpha,
        )
        alpha_sol = _solve_linear(to_linear_problem(alpha_eq), linear_solver)
        for c in 1:nc
            vof_state.alpha.internal[c] = alpha_sol.u[c]
        end

        # ── 2. Boundedness limiter ──────────────────────────────
        clip_alpha!(vof_state.alpha, mesh)

        # ── 3. Update mixture properties ────────────────────────
        update_mixture_properties!(vof_state, props)

        # ── 4. Body forces (gravity + surface tension) ──────────
        body_force = Vector{SVector{Dim, T}}(undef, nc)
        for c in 1:nc
            body_force[c] = vof_state.rho[c] * g
        end

        # Surface tension
        F_st = compute_surface_tension_force(vof_state.alpha, props, mesh)
        if F_st !== nothing
            for c in 1:nc
                body_force[c] = body_force[c] + F_st[c]
            end
        end

        # ── 5. Kinematic viscosity per cell ─────────────────────
        nu_eff = Vector{T}(undef, nc)
        for c in 1:nc
            nu_eff[c] = vof_state.mu[c] / vof_state.rho[c]
        end

        # ── 6. PISO/PIMPLE step with variable density ──────────
        if algorithm isa PISO
            _vof_piso_step!(
                state, prob, dt_actual, algorithm.n_correctors,
                nu_eff, body_force, vof_state.rho;
                linear_solver = linear_solver,
            )
        elseif algorithm isa PIMPLE
            _vof_pimple_step!(
                state, prob, dt_actual,
                nu_eff, body_force, vof_state.rho;
                linear_solver = linear_solver,
            )
        end

        t += dt_actual
        n_steps += 1

        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)

        if verbose && n_steps % max(1, round(Int, (t_end - t_start) / dt / 20)) == 0
            alpha_min = minimum(vof_state.alpha.internal)
            alpha_max = maximum(vof_state.alpha.internal)
            println(
                "Step ", lpad(n_steps, 6),
                "  t=", @sprintf("%.4e", t),
                "  cont=", @sprintf("%.3e", r_cont),
                "  α=[", @sprintf("%.4f", alpha_min), ",", @sprintf("%.4f", alpha_max), "]",
            )
        end
    end

    result = SolveResult{Dim, T}(true, n_steps, residuals, state)
    return (result, vof_state)
end

# ── VOF PISO step (variable density) ────────────────────────────────

function _vof_piso_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T, n_correctors::Int,
        nu_eff::Vector{T},
        body_force::Vector{SVector{Dim, T}},
        rho::Vector{T};
        linear_solver = nothing,
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Momentum predictor
    eqs = CollocatedEquation{T}[]
    for d in 1:Dim
        eq = CollocatedEquation(mesh)
        assemble_momentum!(eq, state, prob, d;
            dt = dt, nu_eff = nu_eff, body_force = body_force)
        push!(eqs, eq)
    end

    extract_momentum_operators!(state, eqs, mesh)

    for d in 1:Dim
        sol = _solve_linear(to_linear_problem(eqs[d]), linear_solver)
        _set_component!(state.U, d, sol.u)
    end
    update_boundary_velocity!(state, prob.bcs, mesh)

    # Pressure corrector with density-weighted diffusivity
    for k in 1:n_correctors
        p_eq = CollocatedEquation(mesh)

        # Density-weighted pressure diffusivity: D = V / (rho * A_P)
        D = Vector{T}(undef, nc)
        for c in 1:nc
            D[c] = mesh.cell_volumes[c] / (rho[c] * state.A_P[c])
        end

        bcs_p = expand_bcs_pressure(prob.bcs)
        assemble_laplacian!(p_eq, D, mesh, bcs_p)

        # RHS: divergence of H(U)/A_P flux (same as standard but density-weighted)
        phi_HbyA = compute_HbyA_flux(state, mesh)
        nf = size(mesh.face_cells, 2)
        for f in 1:nf
            P = owner(mesh, f)
            p_eq.b[P] -= phi_HbyA[f]
            if is_internal_face(mesh, f)
                N = neighbour(mesh, f)
                p_eq.b[N] += phi_HbyA[f]
            end
        end

        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end

        p_sol = _solve_linear(to_linear_problem(p_eq), linear_solver)
        for c in 1:nc
            state.p.internal[c] = p_sol.u[c]
        end

        update_boundary_pressure!(state, prob.bcs, mesh)
        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh)

        if k < n_correctors
            eqs_k = CollocatedEquation{T}[]
            for d in 1:Dim
                eq = CollocatedEquation(mesh)
                assemble_momentum!(eq, state, prob, d;
                    dt = dt, nu_eff = nu_eff, body_force = body_force)
                push!(eqs_k, eq)
            end
            extract_momentum_operators!(state, eqs_k, mesh)
        end
    end

    return nothing
end

# ── VOF PIMPLE step (variable density) ──────────────────────────────

function _vof_pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T,
        nu_eff::Vector{T},
        body_force::Vector{SVector{Dim, T}},
        rho::Vector{T};
        linear_solver = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::PIMPLE{T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    for outer in 1:algo.n_outer
        is_final = (outer == algo.n_outer)

        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(eq, state, prob, d;
                dt = dt, nu_eff = nu_eff, body_force = body_force)
            push!(eqs, eq)
        end
        extract_momentum_operators!(state, eqs, mesh)

        for d in 1:Dim
            if !is_final
                U_old_d = _extract_component(state.U, d)
                under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            end
            sol = _solve_linear(to_linear_problem(eqs[d]), linear_solver)
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)

        for k in 1:algo.n_correctors
            p_eq = CollocatedEquation(mesh)

            D = Vector{T}(undef, nc)
            for c in 1:nc
                D[c] = mesh.cell_volumes[c] / (rho[c] * state.A_P[c])
            end

            bcs_p = expand_bcs_pressure(prob.bcs)
            assemble_laplacian!(p_eq, D, mesh, bcs_p)

            phi_HbyA = compute_HbyA_flux(state, mesh)
            nf = size(mesh.face_cells, 2)
            for f in 1:nf
                P = owner(mesh, f)
                p_eq.b[P] -= phi_HbyA[f]
                if is_internal_face(mesh, f)
                    N = neighbour(mesh, f)
                    p_eq.b[N] += phi_HbyA[f]
                end
            end

            if _needs_pressure_reference(prob.bcs)
                fix_pressure_reference!(p_eq, 1, zero(T))
            end

            p_sol = _solve_linear(to_linear_problem(p_eq), linear_solver)

            if !is_final
                for c in 1:nc
                    state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
                end
            else
                for c in 1:nc
                    state.p.internal[c] = p_sol.u[c]
                end
            end

            update_boundary_pressure!(state, prob.bcs, mesh)
            correct_velocity!(state, mesh)
            update_boundary_velocity!(state, prob.bcs, mesh)
            correct_fluxes!(state, mesh)
        end
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["types", "mixture", "boundedness", "alpha_transport", "surface_tension", "solvers"]; include("src/multiphase/$f.jl"); end; println("OK")'
```

---

### Task 6: Wire into module — Layer 2 includes + exports

**Files:**
- Modify: `src/layers/discretization_assembly_kernels.jl`
- Modify: `src/FiniteVolumeMethod.jl`

- [ ] **Step 1: Add includes to Layer 2**

Append to `src/layers/discretization_assembly_kernels.jl` after the thermal includes (after `include("../thermal/solvers.jl")`):

```julia
# Multiphase VOF (Phase 7)
# Depends on Phase 0 operators + Phase 1 incompressible.
include("../multiphase/types.jl")
include("../multiphase/mixture.jl")
include("../multiphase/boundedness.jl")
include("../multiphase/alpha_transport.jl")
include("../multiphase/surface_tension.jl")
include("../multiphase/solvers.jl")
```

- [ ] **Step 2: Add exports**

In `src/FiniteVolumeMethod.jl`, add after the Phase 12 post-processing exports and before `export FVMGeometry`:

```julia
# --- Multiphase VOF (Phase 7) ---
export
    TwoPhaseProperties,
    VOFState,
    has_surface_tension,
    assemble_alpha!,
    compute_compression_flux,
    clip_alpha!,
    update_mixture_properties!,
    compute_curvature,
    compute_surface_tension_force,
    solve_vof
```

- [ ] **Step 3: Verify module loads**

```bash
julia --project -e 'using FiniteVolumeMethod; println("Phase 7: ", TwoPhaseProperties)'
```

- [ ] **Step 4: Commit**

```bash
git add src/multiphase/ src/layers/discretization_assembly_kernels.jl src/FiniteVolumeMethod.jl
git commit -m "feat: add VOF multiphase (alpha transport, CSF surface tension, mixture properties) — Phase 7"
```

---

### Task 7: Write tests

**Files:**
- Create: `test/multiphase_vof.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the test file**

Create `test/multiphase_vof.jl`. Copy `build_cartesian_unstructured_mesh` from `test/incompressible.jl`. Tests:

1. **TwoPhaseProperties defaults** — rho1=1000, rho2=1.225, sigma=0.072
2. **VOFState construction** — uniform alpha, correct array sizes
3. **VOFState with function init** — alpha_func sets left half to 1, right half to 0
4. **update_mixture_properties!** — alpha=1 gives rho1/mu1, alpha=0 gives rho2/mu2, alpha=0.5 gives averages
5. **clip_alpha! within bounds** — field already in [0,1] → unchanged
6. **clip_alpha! clips and conserves** — set some cells to 1.5 and -0.3, verify clipped to [0,1] and total alpha*V conserved
7. **assemble_alpha! smoke** — 4x4 mesh, verify nonzero A matrix
8. **compute_compression_flux** — uniform alpha → zero compression (no gradient). Non-uniform → nonzero compression at interface
9. **compute_curvature** — uniform alpha → zero curvature. Step function → nonzero curvature at interface
10. **compute_surface_tension_force** — sigma=0 returns nothing. sigma>0 returns force vector at interface
11. **solve_vof smoke** — 4x4 mesh, all walls, alpha=1 in left half, 2 time steps. Verify returns (SolveResult, VOFState) with bounded alpha.
12. **has_surface_tension** — true when sigma>0, false when sigma=0

- [ ] **Step 2: Register test**

Add `safe_include("multiphase_vof.jl")` to `test/runtests.jl` after the LES turbulence test.

- [ ] **Step 3: Run tests**

```bash
julia --project=test test/multiphase_vof.jl
```

- [ ] **Step 4: Run Runic**

```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "src/multiphase/"])'
julia --project -e 'using Runic; Runic.main(["--inplace", "test/multiphase_vof.jl"])'
```

- [ ] **Step 5: Commit**

```bash
git add test/multiphase_vof.jl test/runtests.jl
git commit -m "test: add VOF multiphase test suite"
```

---

### Task 8: Register in validation manifest + final verification

**Files:**
- Modify: `validation/manifest.toml`

- [ ] **Step 1: Add multiphase_vof feature**

Append to `validation/manifest.toml`:

```toml
# ── Phase 7: Multiphase VOF ───────────────────────────────────────

[[features]]
feature = "multiphase_vof"
maturity = "experimental"
validation = "smoke_tested"
role = "research_tooling"
solver_family = "collocated"
precision_policy = "float64_cpu_reference"
random_seed_policy = "deterministic"
backend_policy = "cpu_reference"
required_ladder_stages = ["verification", "benchmark"]
summary = "Volume of Fluid two-phase flow with alpha transport, interface compression, CSF surface tension, and mixture property blending."
limitations = [
  "Experimental — validated via smoke tests only; dam break benchmark pending.",
  "Boundedness limiter is clip+redistribute, not face-based MULES.",
  "No contact angle model for wall-interface interaction.",
]
```

- [ ] **Step 2: Verify all exports**

```bash
julia --project -e '
using FiniteVolumeMethod
for sym in [:TwoPhaseProperties, :VOFState, :solve_vof, :clip_alpha!,
            :update_mixture_properties!, :compute_curvature, :compute_surface_tension_force,
            :assemble_alpha!, :compute_compression_flux, :has_surface_tension]
    @assert isdefined(FiniteVolumeMethod, sym) "Missing: $sym"
end
println("All Phase 7 exports verified")
'
```

- [ ] **Step 3: Run tests + regression**

```bash
julia --project=test test/multiphase_vof.jl
julia --project=test test/incompressible.jl
```

- [ ] **Step 4: Runic check**

```bash
julia --project -e 'using Runic; Runic.main(["--check", "src/multiphase/"])'
```

- [ ] **Step 5: Commit**

```bash
git add validation/manifest.toml
git commit -m "feat: register multiphase_vof in validation manifest"
```

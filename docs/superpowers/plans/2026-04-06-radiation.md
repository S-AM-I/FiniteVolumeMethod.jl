# Radiation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add P1 thermal radiation model — a single diffusion equation for incident radiation G, radiation source for the energy equation, and a combined thermal+radiation solver wrapper.

**Architecture:** Three new files in `src/radiation/` wired into Layer 2. P1 equation assembled via Phase 0 Laplacian + absorption diagonal + emission source. Radiation source couples to energy equation explicitly (lagged one iteration). Combined solver wraps Phase 3 thermal solver with P1 step.

**Tech Stack:** Julia, Phase 0 collocated operators (assemble_laplacian!, CollocatedEquation), Phase 3 thermal solver (solve_simple_thermal pattern), SciMLBase (LinearProblem).

---

## File Map

| File | Purpose | Creates/Modifies |
|------|---------|-----------------|
| `src/radiation/types.jl` | AbstractRadiationModel, P1Model, RadiationState, STEFAN_BOLTZMANN, BC helpers | Create |
| `src/radiation/p1.jl` | assemble_p1!, solve_p1_radiation, compute_radiation_source | Create |
| `src/radiation/solvers.jl` | solve_simple_thermal_radiation wrapper | Create |
| `src/layers/discretization_assembly_kernels.jl` | Wire radiation includes | Modify |
| `src/FiniteVolumeMethod.jl` | Add exports | Modify |
| `test/radiation.jl` | All tests | Create |
| `test/runtests.jl` | Register test | Modify |
| `validation/manifest.toml` | Register feature | Modify |

---

### Task 1: Create all 3 radiation source files

**Files:**
- Create: `src/radiation/types.jl`
- Create: `src/radiation/p1.jl`
- Create: `src/radiation/solvers.jl`

- [ ] **Step 1: Create directory and types.jl**

```bash
mkdir -p src/radiation
```

Write `src/radiation/types.jl`:

```julia
# radiation/types.jl — Core types for radiation modeling
#
# Defines the radiation model hierarchy, the P1 model, radiation state,
# and boundary condition convenience constructors.

"""Stefan-Boltzmann constant σ [W/(m²·K⁴)]."""
const STEFAN_BOLTZMANN = 5.670374419e-8

"""
    AbstractRadiationModel

Supertype for radiation models.
"""
abstract type AbstractRadiationModel end

"""
    P1Model{T} <: AbstractRadiationModel

P1 radiation approximation. Solves a single diffusion equation for the
incident radiation field G:

    -div(Γ·grad(G)) + a·G = 4·a·σ·T⁴

where `Γ = 1/(3a)` and `a` is the absorption coefficient.

# Fields
- `a::T` — absorption coefficient [1/m]
"""
struct P1Model{T} <: AbstractRadiationModel
    a::T
end

"""
    P1Model(; a = 0.1)

Construct a P1 radiation model with constant absorption coefficient.
"""
P1Model(; a::Real = 0.1) = P1Model{typeof(Float64(a))}(Float64(a))

"""
    RadiationState{T}

Mutable state for radiation models. Holds the incident radiation field.

# Fields
- `G::CollocatedScalarField{T}` — incident radiation [W/m²]
"""
mutable struct RadiationState{T}
    G::CollocatedScalarField{T}
end

"""
    RadiationState(mesh; G_init = 0.0)

Construct a zero-initialized radiation state.
"""
function RadiationState(
        mesh::UnstructuredFVMMesh{Dim, T};
        G_init::Real = 0.0,
    ) where {Dim, T}
    G = CollocatedScalarField(:G, mesh; value = T(G_init))
    return RadiationState{T}(G)
end

# ── BC convenience constructors ──────────────────────────────────────

"""
    marshak_wall_bc(rad_model::P1Model, T_wall)

Marshak boundary condition for an opaque wall at temperature `T_wall`:
`G + (2/(3a))·∂G/∂n = 4σT_wall⁴`

Implemented as `ParabolicRobin(1, 2/(3a), 4σT⁴)`.
"""
function marshak_wall_bc(rad_model::P1Model{T}, T_wall::Real) where {T}
    b_coeff = T(2) / (T(3) * rad_model.a)
    c_val = T(4) * T(STEFAN_BOLTZMANN) * T(T_wall)^4
    return ParabolicRobin(one(T), b_coeff, c_val)
end

"""
    radiation_inlet_bc(T_inlet)

Fixed incident radiation BC from a known temperature:
`G = 4σT⁴`
"""
function radiation_inlet_bc(T_inlet::Real)
    G_val = 4.0 * STEFAN_BOLTZMANN * Float64(T_inlet)^4
    return ParabolicDirichlet(G_val)
end
```

- [ ] **Step 2: Write p1.jl**

Write `src/radiation/p1.jl`:

```julia
# radiation/p1.jl — P1 radiation equation assembly and solver
#
# Assembles: -div(Γ·grad(G)) + a·G = 4·a·σ·T⁴
# where Γ = 1/(3a). Uses Phase 0 Laplacian + source terms.

"""
    assemble_p1!(
        eq, rad_model, T_field, mesh, bcs_G,
    )

Assemble the P1 radiation equation into `eq`.

The equation `-div(Γ·grad(G)) + a·G = 4aσT⁴` becomes:
- Laplacian with diffusivity `Γ = 1/(3a)` → contributes to A (positive diagonal)
- Absorption `a·V` → added to diagonal
- Emission `4aσT⁴·V` → added to RHS
"""
function assemble_p1!(
        eq::CollocatedEquation{T},
        rad_model::P1Model{T},
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    a = rad_model.a

    # Radiation diffusivity: Γ = 1/(3a)
    gamma = one(T) / (T(3) * a)

    # Laplacian: -div(Γ·grad(G)) assembled as positive-definite operator
    assemble_laplacian!(eq, gamma, mesh, bcs_G)

    # Absorption (implicit): a·V on diagonal
    for c in 1:nc
        eq.A[c, c] += a * mesh.cell_volumes[c]
    end

    # Emission (explicit RHS): 4·a·σ·T⁴·V
    sigma = T(STEFAN_BOLTZMANN)
    for c in 1:nc
        T_c = max(T_field.internal[c], zero(T))
        eq.b[c] += T(4) * a * sigma * T_c^4 * mesh.cell_volumes[c]
    end

    return nothing
end

"""
    solve_p1_radiation(
        rad_model, T_field, mesh, bcs_G; linear_solver = nothing,
    ) -> CollocatedScalarField{T}

Assemble and solve the P1 radiation equation, returning the G field.
"""
function solve_p1_radiation(
        rad_model::P1Model{T},
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition};
        linear_solver = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    assemble_p1!(eq, rad_model, T_field, mesh, bcs_G)

    lp = to_linear_problem(eq)
    sol = _solve_linear(lp, linear_solver)

    G = CollocatedScalarField(:G, mesh)
    for c in 1:nc
        G.internal[c] = max(sol.u[c], zero(T))
    end

    return G
end

"""
    compute_radiation_source(
        rad_model, G, T_field,
    ) -> Vector{T}

Compute the volumetric radiation source term for the energy equation:
`S_rad[c] = a · G[c] - 4 · a · σ · T[c]⁴`

Positive = net absorption (fluid heats up).
Negative = net emission (fluid cools).

To add to the energy equation (which is scaled by 1/(ρ·Cp)):
`eq.b[c] += S_rad[c] * V_c / (rho * Cp)`
"""
function compute_radiation_source(
        rad_model::P1Model{T},
        G::CollocatedScalarField{T},
        T_field::CollocatedScalarField{T},
    ) where {T}
    nc = length(G.internal)
    a = rad_model.a
    sigma = T(STEFAN_BOLTZMANN)
    S_rad = Vector{T}(undef, nc)

    for c in 1:nc
        T_c = max(T_field.internal[c], zero(T))
        S_rad[c] = a * G.internal[c] - T(4) * a * sigma * T_c^4
    end

    return S_rad
end
```

- [ ] **Step 3: Write solvers.jl**

Write `src/radiation/solvers.jl`:

```julia
# radiation/solvers.jl — Combined thermal + radiation solver wrapper
#
# Extends the Phase 3 thermal solver with a P1 radiation step after
# the energy equation. Radiation source is lagged one iteration.

using Printf: @sprintf

"""
    solve_simple_thermal_radiation(
        prob, thermal_props, rad_model;
        bcs_T, bcs_G,
        turb_model, turb_bcs, T_init,
        linear_solver, verbose,
    ) -> Tuple{SolveResult, ThermalState, RadiationState}

Solve steady incompressible flow with energy equation and P1 radiation.

Each SIMPLE iteration:
1. Update effective properties (k_eff, nu_eff, buoyancy)
2. Momentum + pressure + correction
3. Turbulence (optional)
4. Solve energy equation with radiation source in RHS
5. Solve P1 radiation equation for G
6. Update radiation source
7. Check convergence
"""
function solve_simple_thermal_radiation(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T},
        rad_model::P1Model{T};
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition},
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        T_init::Real = thermal_props.T_ref,
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Initialize states
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    thermal_state = ThermalState(mesh; T_init = T(T_init), k_init = thermal_props.k)
    rad_state = RadiationState(mesh; G_init = T(4) * T(STEFAN_BOLTZMANN) * T(T_init)^4)

    # Turbulence (optional)
    turb_state = nothing
    if turb_model !== nothing
        turb_state = RANSTurbulenceState(turb_model, mesh)
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
    end

    # Radiation source (initialized to zero, updated after first G solve)
    S_rad = zeros(T, nc)

    # Residuals
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    converged = false
    final_iter = 0

    for iter in 1:algo.max_iterations
        final_iter = iter

        # ── Effective properties ────────────────────────────────
        nu_t_vec = turb_state === nothing ? nothing : turb_state.nu_t
        update_k_eff!(thermal_state, thermal_props, nu_t_vec, prob.density)
        nu_eff = turb_state === nothing ? prob.nu : compute_nu_eff(prob.nu, turb_state.nu_t)
        alpha_eff = compute_alpha_eff(thermal_state.k_eff, prob.density, thermal_props.Cp)

        # Buoyancy
        body_force = compute_buoyancy_source(thermal_state.T_field, thermal_props, prob.density)

        # ── Momentum ────────────────────────────────────────────
        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(eq, state, prob, d;
                nu_eff = nu_eff, body_force = body_force)
            push!(eqs, eq)
        end

        extract_momentum_operators!(state, eqs, mesh)

        for d in 1:Dim
            U_old_d = _extract_component(state.U, d)
            under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            sol = _solve_linear(to_linear_problem(eqs[d]), linear_solver)
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)

        # ── Pressure ────────────────────────────────────────────
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = _solve_linear(to_linear_problem(p_eq), linear_solver)

        for c in 1:nc
            state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
        end
        update_boundary_pressure!(state, prob.bcs, mesh)

        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh)

        # ── Turbulence (optional) ───────────────────────────────
        if turb_model !== nothing
            _update_turbulence!(
                turb_state, turb_model, state, prob, mesh, turb_bcs;
                linear_solver = linear_solver,
            )
        end

        # ── Energy equation + radiation source ──────────────────
        T_eq = CollocatedEquation(mesh)
        assemble_energy!(T_eq, thermal_state.T_field, state.phi, alpha_eff, mesh, bcs_T)

        # Add radiation source to energy RHS (scaled by 1/(rho*Cp))
        rho_Cp = prob.density * thermal_props.Cp
        for c in 1:nc
            T_eq.b[c] += S_rad[c] * mesh.cell_volumes[c] / rho_Cp
        end

        T_sol = _solve_linear(to_linear_problem(T_eq), linear_solver)
        for c in 1:nc
            thermal_state.T_field.internal[c] = T_sol.u[c]
        end

        # ── P1 radiation ────────────────────────────────────────
        rad_state.G = solve_p1_radiation(
            rad_model, thermal_state.T_field, mesh, bcs_G;
            linear_solver = linear_solver,
        )
        S_rad = compute_radiation_source(rad_model, rad_state.G, thermal_state.T_field)

        # ── Convergence ─────────────────────────────────────────
        max_res = zero(T)
        for d in 1:Dim
            u_d = _extract_component(state.U, d)
            r = momentum_residual(eqs[d], u_d)
            push!(residuals[component_labels[d]], r)
            max_res = max(max_res, r)
        end
        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)
        max_res = max(max_res, r_cont)

        if verbose
            _print_simple_residuals(iter, residuals, component_labels)
        end

        if max_res < algo.tolerance
            converged = true
            break
        end
    end

    result = SolveResult{Dim, T}(converged, final_iter, residuals, state)
    return (result, thermal_state, rad_state)
end
```

- [ ] **Step 4: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["types", "p1", "solvers"]; include("src/radiation/$f.jl"); end; println("OK")'
```

---

### Task 2: Wire into module + exports

**Files:**
- Modify: `src/layers/discretization_assembly_kernels.jl`
- Modify: `src/FiniteVolumeMethod.jl`

- [ ] **Step 1: Add includes to Layer 2**

Append after multiphase includes:

```julia
# Radiation (Phase 9)
# Depends on Phase 0 operators + Phase 3 thermal.
include("../radiation/types.jl")
include("../radiation/p1.jl")
include("../radiation/solvers.jl")
```

- [ ] **Step 2: Add exports**

After the Phase 7 multiphase exports:

```julia
# --- Radiation (Phase 9) ---
export
    AbstractRadiationModel,
    P1Model,
    RadiationState,
    STEFAN_BOLTZMANN,
    assemble_p1!,
    solve_p1_radiation,
    compute_radiation_source,
    marshak_wall_bc,
    radiation_inlet_bc,
    solve_simple_thermal_radiation
```

- [ ] **Step 3: Verify module loads**

```bash
julia --project -e 'using FiniteVolumeMethod; println("Phase 9: ", P1Model)'
```

---

### Task 3: Write tests

**Files:**
- Create: `test/radiation.jl`
- Modify: `test/runtests.jl`

Tests:

1. **P1Model construction** — default a=0.1, custom a
2. **RadiationState construction** — G initialized, correct length
3. **STEFAN_BOLTZMANN value** — verify ≈ 5.67e-8
4. **marshak_wall_bc** — verify produces ParabolicRobin with correct coefficients
5. **radiation_inlet_bc** — verify produces ParabolicDirichlet(4σT⁴)
6. **assemble_p1! smoke** — 4x4 mesh, nonzero A after assembly
7. **solve_p1_radiation** — uniform T=300K field, solve P1. G should be positive and ≈ 4σT⁴ in optically thick limit
8. **compute_radiation_source** — known G and T, verify S_rad = aG - 4aσT⁴. At equilibrium (G = 4σT⁴), S_rad ≈ 0
9. **solve_simple_thermal_radiation smoke** — 8x4 mesh, heated channel with radiation, 5 iterations. Verify returns (SolveResult, ThermalState, RadiationState) with finite fields

---

### Task 4: Register in validation manifest

Append feature `radiation` with maturity `experimental`.

---

Commit and push all at once.

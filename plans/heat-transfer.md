---
date: 2026-04-06
---

# Conjugate Heat Transfer & Buoyancy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add thermal modeling: fluid energy equation with turbulent heat flux, Boussinesq buoyancy for natural convection, solid-region conduction, and Dirichlet-Neumann conjugate heat transfer coupling between fluid and solid.

**Architecture:** New `src/thermal/` directory with types, energy equation assembly, buoyancy source, solid conduction, conjugate coupling iteration, and thermal solver wrappers. Energy equation is a scalar transport (convection + Laplacian) solved segregated after turbulence in SIMPLE/PISO/PIMPLE loops. Buoyancy adds a body force to momentum via a new keyword. Conjugate coupling iterates Dirichlet-Neumann between fluid and solid solvers.

**Tech Stack:** Julia, Phase 0 collocated operators (assemble_convection!, assemble_laplacian!, gradient), Phase 1 incompressible solvers, Phase 2a turbulence (optional nu_t for k_eff).

---

## File Map

| File | Purpose | Creates/Modifies |
|------|---------|-----------------|
| `src/thermal/types.jl` | FluidThermalProperties, SolidThermalProperties, ThermalState, ConjugateHeatTransferProblem | Create |
| `src/thermal/energy_equation.jl` | assemble_energy!, solve_energy!, update_k_eff! | Create |
| `src/thermal/buoyancy.jl` | compute_buoyancy_source | Create |
| `src/thermal/solid_conduction.jl` | assemble_solid_conduction!, solve_solid_conduction | Create |
| `src/thermal/conjugate.jl` | solve_conjugate_ht, compute_interface_heat_flux | Create |
| `src/thermal/solvers.jl` | solve_simple_thermal, solve_incompressible_thermal | Create |
| `src/incompressible/momentum.jl` | Add body_force keyword | Modify |
| `src/layers/discretization_assembly_kernels.jl` | Wire thermal includes | Modify |
| `src/FiniteVolumeMethod.jl` | Add thermal exports | Modify |
| `test/thermal.jl` | All tests | Create |
| `test/runtests.jl` | Register test | Modify |
| `validation/manifest.toml` | Register features | Modify |

---

### Task 1: Add body_force keyword to assemble_momentum!

**Files:**
- Modify: `src/incompressible/momentum.jl`

- [ ] **Step 1: Add body_force keyword argument**

In `src/incompressible/momentum.jl`, modify `assemble_momentum!` to accept a `body_force` keyword. Change the signature from:

```julia
function assemble_momentum!(
        eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        component::Int;
        dt::Union{Nothing, T} = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
        nu_eff::Union{T, Vector{T}} = prob.nu,
    ) where {Dim, T}
```

to:

```julia
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
```

And add before `return nothing` at the end of the function:

```julia
    # Body force (buoyancy, etc.)
    if body_force !== nothing
        for c in 1:nc
            eq.b[c] += body_force[c][component] * mesh.cell_volumes[c]
        end
    end
```

Also update the docstring to include `body_force` in the arguments list.

- [ ] **Step 2: Verify backward compatibility**

```bash
julia --project=test test/incompressible.jl
```
Expected: All 94 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/incompressible/momentum.jl
git commit -m "feat: add body_force keyword to assemble_momentum! for buoyancy support"
```

---

### Task 2: Create types.jl — Thermal property types and state

**Files:**
- Create: `src/thermal/types.jl`

- [ ] **Step 1: Create directory and types file**

```bash
mkdir -p src/thermal
```

Write `src/thermal/types.jl`:

```julia
# thermal/types.jl — Core types for heat transfer modeling
#
# Defines fluid and solid thermal property structs, the mutable thermal
# state (temperature + effective conductivity), and the conjugate heat
# transfer problem container.

# ── Fluid thermal properties ─────────────────────────────────────────

"""
    FluidThermalProperties{Dim, T}

Thermophysical properties for the fluid region in thermal simulations.

# Fields
- `Cp::T` — specific heat capacity [J/(kg·K)]
- `k::T` — laminar thermal conductivity [W/(m·K)]
- `Pr_t::T` — turbulent Prandtl number (default 0.85)
- `beta::T` — thermal expansion coefficient [1/K] (0 = no buoyancy)
- `T_ref::T` — reference temperature for Boussinesq approximation [K]
- `g::SVector{Dim, T}` — gravity vector [m/s²]
"""
struct FluidThermalProperties{Dim, T}
    Cp::T
    k::T
    Pr_t::T
    beta::T
    T_ref::T
    g::SVector{Dim, T}
end

"""
    FluidThermalProperties{Dim}(; Cp, k, Pr_t, beta, T_ref, g)

Construct fluid thermal properties with keyword arguments.
When `beta == 0` (default), buoyancy is disabled.
"""
function FluidThermalProperties{Dim}(;
        Cp::Real = 1005.0,
        k::Real = 0.026,
        Pr_t::Real = 0.85,
        beta::Real = 0.0,
        T_ref::Real = 300.0,
        g = nothing,
    ) where {Dim}
    T = promote_type(typeof(Cp), typeof(k), typeof(Pr_t), typeof(beta), typeof(T_ref))
    if g === nothing
        g_vec = Dim == 2 ? SVector{2, T}(zero(T), T(-9.81)) : SVector{3, T}(zero(T), zero(T), T(-9.81))
    else
        g_vec = SVector{Dim, T}(g)
    end
    return FluidThermalProperties{Dim, T}(T(Cp), T(k), T(Pr_t), T(beta), T(T_ref), g_vec)
end

"""Check if buoyancy is active."""
has_buoyancy(props::FluidThermalProperties) = props.beta != 0

# ── Solid thermal properties ─────────────────────────────────────────

"""
    SolidThermalProperties{T}

Thermophysical properties for a solid conduction region.

# Fields
- `rho::T` — density [kg/m³]
- `Cp::T` — specific heat capacity [J/(kg·K)]
- `k::T` — thermal conductivity [W/(m·K)]
- `Q_gen::T` — volumetric heat generation [W/m³] (default 0)
"""
struct SolidThermalProperties{T}
    rho::T
    Cp::T
    k::T
    Q_gen::T
end

function SolidThermalProperties(;
        rho::Real = 7800.0, Cp::Real = 500.0,
        k::Real = 50.0, Q_gen::Real = 0.0,
    )
    T = promote_type(typeof(rho), typeof(Cp), typeof(k), typeof(Q_gen))
    return SolidThermalProperties{T}(T(rho), T(Cp), T(k), T(Q_gen))
end

# ── Thermal state ────────────────────────────────────────────────────

"""
    ThermalState{T}

Mutable state for the temperature field and effective thermal conductivity.

# Fields
- `T_field::CollocatedScalarField{T}` — temperature [K]
- `k_eff::Vector{T}` — effective conductivity per cell [W/(m·K)]
"""
mutable struct ThermalState{T}
    T_field::CollocatedScalarField{T}
    k_eff::Vector{T}
end

"""
    ThermalState(mesh; T_init = 300.0, k_init = 0.026)

Construct a thermal state on `mesh` with uniform initial temperature.
"""
function ThermalState(
        mesh::UnstructuredFVMMesh{Dim, T};
        T_init::Real = 300.0,
        k_init::Real = 0.026,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    T_field = CollocatedScalarField(:T, mesh; value = T(T_init))
    k_eff = fill(T(k_init), nc)
    return ThermalState{T}(T_field, k_eff)
end

# ── Conjugate heat transfer problem ──────────────────────────────────

"""
    ConjugateHeatTransferProblem{Dim, T, FM, SM}

Multi-region conjugate heat transfer problem coupling a fluid domain
(incompressible NS + energy equation) with a solid conduction domain
via Dirichlet-Neumann iteration at their shared interface.

# Fields
- `fluid_prob` — incompressible flow problem for the fluid domain
- `fluid_thermal` — fluid thermal properties
- `fluid_bcs_T` — temperature BCs for the fluid domain
- `solid_mesh` — mesh for the solid conduction domain
- `solid_thermal` — solid thermal properties
- `solid_bcs_T` — temperature BCs for the solid domain
- `interface_fluid_patch` — patch name on the fluid mesh at the interface
- `interface_solid_patch` — patch name on the solid mesh at the interface
- `max_coupling_iterations` — coupling loop iteration limit
- `coupling_tolerance` — convergence threshold for interface temperature
"""
struct ConjugateHeatTransferProblem{Dim, T, FM, SM}
    fluid_prob::IncompressibleProblem{Dim, T, FM}
    fluid_thermal::FluidThermalProperties{Dim, T}
    fluid_bcs_T::Dict{Symbol, AbstractBoundaryCondition}
    solid_mesh::SM
    solid_thermal::SolidThermalProperties{T}
    solid_bcs_T::Dict{Symbol, AbstractBoundaryCondition}
    interface_fluid_patch::Symbol
    interface_solid_patch::Symbol
    max_coupling_iterations::Int
    coupling_tolerance::T
end

function ConjugateHeatTransferProblem(
        fluid_prob::IncompressibleProblem{Dim, T},
        fluid_thermal::FluidThermalProperties{Dim, T},
        fluid_bcs_T,
        solid_mesh::UnstructuredFVMMesh{Dim, T},
        solid_thermal::SolidThermalProperties{T},
        solid_bcs_T;
        interface_fluid_patch::Symbol,
        interface_solid_patch::Symbol,
        max_coupling_iterations::Int = 50,
        coupling_tolerance::T = T(1e-4),
    ) where {Dim, T}
    return ConjugateHeatTransferProblem{Dim, T, typeof(fluid_prob.mesh), typeof(solid_mesh)}(
        fluid_prob, fluid_thermal, fluid_bcs_T,
        solid_mesh, solid_thermal, solid_bcs_T,
        interface_fluid_patch, interface_solid_patch,
        max_coupling_iterations, coupling_tolerance,
    )
end

# ── BC convenience constructors ──────────────────────────────────────

"""Fixed temperature BC (wall, inlet)."""
thermal_inlet_bc(T_val::Real) = ParabolicDirichlet(Float64(T_val))

"""Insulated (zero heat flux) BC."""
thermal_insulated_bc() = ParabolicNeumann(0.0)

"""Fixed heat flux BC (heated/cooled wall)."""
thermal_heated_wall_bc(q::Real) = ParabolicNeumann(Float64(q))

"""Convective BC: h·(T - T_inf)."""
thermal_convective_bc(h::Real, T_inf::Real) = ParabolicRobin(Float64(h), 1.0, Float64(h * T_inf))
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/thermal/types.jl"); println("OK")'
```

---

### Task 3: Create energy_equation.jl — Fluid energy transport

**Files:**
- Create: `src/thermal/energy_equation.jl`

- [ ] **Step 1: Write energy equation assembly**

Write `src/thermal/energy_equation.jl`:

```julia
# thermal/energy_equation.jl — Energy equation assembly for incompressible flow
#
# Assembles the temperature transport equation:
#   ∂T/∂t + div(phi·T) = div(alpha_eff · grad(T))
# where alpha_eff = k_eff / (rho·Cp) is the effective thermal diffusivity.
# The equation is divided by rho·Cp so convection uses phi directly.

"""
    assemble_energy!(
        eq::CollocatedEquation{T},
        T_field::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        alpha_eff::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
    )

Assemble the energy (temperature) transport equation into `eq`.

The equation is scaled by `1/(ρ·Cp)` so that:
- Convection uses the volumetric face flux `phi` directly
- Diffusion uses thermal diffusivity `alpha_eff = k_eff/(ρ·Cp)`
- Temporal term has unit density coefficient

# Arguments
- `eq` — equation (modified in-place)
- `T_field` — current temperature field (for temporal term)
- `phi` — face volumetric flux from the flow solver
- `alpha_eff` — effective thermal diffusivity: scalar or per-cell vector
- `mesh` — unstructured FVM mesh
- `bcs_T` — temperature boundary conditions
- `dt` — time step (nothing for steady state)
"""
function assemble_energy!(
        eq::CollocatedEquation{T},
        T_field::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        alpha_eff::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
    ) where {Dim, T}
    # Convection: div(phi · T)
    assemble_convection!(eq, phi, mesh, bcs_T)

    # Diffusion: div(alpha_eff · grad(T))
    assemble_laplacian!(eq, alpha_eff, mesh, bcs_T)

    # Temporal term (if transient)
    if dt !== nothing
        assemble_ddt_euler!(eq, one(T), T_field.internal, mesh, dt)
    end

    return nothing
end

"""
    update_k_eff!(
        thermal_state::ThermalState{T},
        thermal_props::FluidThermalProperties{Dim, T},
        nu_t::Union{Nothing, Vector{T}},
        density::T,
    )

Update effective thermal conductivity from turbulent viscosity:
  `k_eff[c] = k_laminar + ρ · Cp · ν_t[c] / Pr_t`

When `nu_t` is `nothing`, uses laminar conductivity only.
"""
function update_k_eff!(
        thermal_state::ThermalState{T},
        thermal_props::FluidThermalProperties{Dim, T},
        nu_t::Union{Nothing, Vector{T}},
        density::T,
    ) where {Dim, T}
    k_lam = thermal_props.k
    for c in eachindex(thermal_state.k_eff)
        if nu_t === nothing
            thermal_state.k_eff[c] = k_lam
        else
            k_t = density * thermal_props.Cp * nu_t[c] / thermal_props.Pr_t
            thermal_state.k_eff[c] = k_lam + k_t
        end
    end
    return nothing
end

"""
    compute_alpha_eff(k_eff::Vector{T}, rho::T, Cp::T) -> Vector{T}

Compute thermal diffusivity `alpha_eff = k_eff / (rho * Cp)`.
"""
function compute_alpha_eff(k_eff::Vector{T}, rho::T, Cp::T) where {T}
    rho_Cp = rho * Cp
    alpha = Vector{T}(undef, length(k_eff))
    for c in eachindex(k_eff)
        alpha[c] = k_eff[c] / rho_Cp
    end
    return alpha
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/thermal/types.jl"); include("src/thermal/energy_equation.jl"); println("OK")'
```

---

### Task 4: Create buoyancy.jl — Boussinesq body force

**Files:**
- Create: `src/thermal/buoyancy.jl`

- [ ] **Step 1: Write buoyancy source computation**

Write `src/thermal/buoyancy.jl`:

```julia
# thermal/buoyancy.jl — Boussinesq buoyancy source term
#
# Computes the body force F_b = -ρ · β · (T - T_ref) · g for natural
# convection using the Boussinesq approximation.

"""
    compute_buoyancy_source(
        T_field::CollocatedScalarField{T},
        props::FluidThermalProperties{Dim, T},
        density::T,
    ) -> Vector{SVector{Dim, T}}

Compute the Boussinesq buoyancy body force per cell:

    F_b[c] = -ρ · β · (T[c] - T_ref) · g

Returns a vector of `SVector{Dim, T}` with one entry per cell.
Returns `nothing` when `β == 0` (no buoyancy).
"""
function compute_buoyancy_source(
        T_field::CollocatedScalarField{T},
        props::FluidThermalProperties{Dim, T},
        density::T,
    ) where {Dim, T}
    if !has_buoyancy(props)
        return nothing
    end

    nc = length(T_field.internal)
    force = Vector{SVector{Dim, T}}(undef, nc)
    for c in 1:nc
        dT = T_field.internal[c] - props.T_ref
        force[c] = -density * props.beta * dT * props.g
    end
    return force
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/thermal/types.jl"); include("src/thermal/buoyancy.jl"); println("OK")'
```

---

### Task 5: Create solid_conduction.jl — Solid-region thermal solver

**Files:**
- Create: `src/thermal/solid_conduction.jl`

- [ ] **Step 1: Write solid conduction assembly and solver**

Write `src/thermal/solid_conduction.jl`:

```julia
# thermal/solid_conduction.jl — Solid conduction equation
#
# Solves: ρ·Cp · ∂T/∂t = div(k · grad(T)) + Q_gen
# For steady state: div(k · grad(T)) = -Q_gen

"""
    assemble_solid_conduction!(
        eq::CollocatedEquation{T},
        solid::SolidThermalProperties{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
        T_old::Union{Nothing, Vector{T}} = nothing,
    )

Assemble the solid conduction equation into `eq`.

For steady state, assembles `div(k · grad(T)) + Q_gen = 0`.
For transient, adds `ρ·Cp · (T - T_old) / dt`.

# Arguments
- `eq` — equation (modified in-place)
- `solid` — solid thermal properties
- `mesh` — solid mesh
- `bcs_T` — temperature boundary conditions for the solid
- `dt` — time step (nothing for steady)
- `T_old` — temperature at previous time step (required if transient)
"""
function assemble_solid_conduction!(
        eq::CollocatedEquation{T},
        solid::SolidThermalProperties{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
        T_old::Union{Nothing, Vector{T}} = nothing,
    ) where {Dim, T}
    # Diffusion: div(k · grad(T))
    assemble_laplacian!(eq, solid.k, mesh, bcs_T)

    # Volumetric heat generation source
    nc = length(mesh.cell_volumes)
    if solid.Q_gen != zero(T)
        for c in 1:nc
            eq.b[c] += solid.Q_gen * mesh.cell_volumes[c]
        end
    end

    # Temporal term (transient only)
    if dt !== nothing && T_old !== nothing
        rho_Cp = solid.rho * solid.Cp
        assemble_ddt_euler!(eq, rho_Cp, T_old, mesh, dt)
    end

    return nothing
end

"""
    solve_solid_conduction(
        mesh::UnstructuredFVMMesh{Dim, T},
        solid::SolidThermalProperties{T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt = nothing,
        T_old = nothing,
        linear_solver = nothing,
    ) -> CollocatedScalarField{T}

Solve the solid conduction equation and return the temperature field.

For steady state (`dt = nothing`), performs a single linear solve.
For transient, requires `T_old` (previous temperature values).
"""
function solve_solid_conduction(
        mesh::UnstructuredFVMMesh{Dim, T},
        solid::SolidThermalProperties{T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
        T_old::Union{Nothing, Vector{T}} = nothing,
        linear_solver = nothing,
    ) where {Dim, T}
    eq = CollocatedEquation(mesh)
    assemble_solid_conduction!(eq, solid, mesh, bcs_T; dt = dt, T_old = T_old)

    lp = to_linear_problem(eq)
    sol = _solve_linear(lp, linear_solver)

    nc = length(mesh.cell_volumes)
    T_field = CollocatedScalarField(:T_solid, mesh)
    for c in 1:nc
        T_field.internal[c] = sol.u[c]
    end

    return T_field
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/thermal/types.jl"); include("src/thermal/solid_conduction.jl"); println("OK")'
```

---

### Task 6: Create conjugate.jl — Dirichlet-Neumann coupling

**Files:**
- Create: `src/thermal/conjugate.jl`

- [ ] **Step 1: Write conjugate coupling iteration**

Write `src/thermal/conjugate.jl`:

```julia
# thermal/conjugate.jl — Conjugate heat transfer coupling
#
# Dirichlet-Neumann iteration between a fluid domain (incompressible NS
# + energy equation) and a solid conduction domain. The fluid sees a
# Dirichlet (fixed temperature) BC at the interface; the solid sees a
# Neumann (fixed heat flux) BC computed from the fluid solution.

"""
    compute_interface_heat_flux(
        T_field::CollocatedScalarField{T},
        k::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        interface_patch::Symbol,
    ) -> Dict{Int, T}

Compute the heat flux at each face of `interface_patch`:

    q_f = -k · (T_boundary - T_cell) / d_cell_to_face

Returns a dictionary mapping face index to heat flux value.
Positive flux means heat flows out of the fluid domain.
"""
function compute_interface_heat_flux(
        T_field::CollocatedScalarField{T},
        k::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        interface_patch::Symbol,
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    flux = Dict{Int, T}()

    pbmap = build_boundary_map(T_field)

    for f in 1:nf
        if !is_internal_face(mesh, f)
            tag = _face_tag(mesh, f)
            if tag == interface_patch
                P = owner(mesh, f)
                T_cell = T_field.internal[P]
                T_bnd = T_field.boundary[pbmap[f]]

                x_c = cell_center(mesh, P)
                x_f = face_center(mesh, f)
                d = norm(x_f - x_c)
                d = max(d, T(1e-15))

                flux[f] = -k * (T_bnd - T_cell) / d
            end
        end
    end

    return flux
end

"""
    _extract_interface_temperatures(
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) -> Dict{Int, T}

Extract boundary face temperatures at the given patch.
"""
function _extract_interface_temperatures(
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    temps = Dict{Int, T}()
    pbmap = build_boundary_map(T_field)
    nf = size(mesh.face_cells, 2)

    for f in 1:nf
        if !is_internal_face(mesh, f)
            tag = _face_tag(mesh, f)
            if tag == patch
                temps[f] = T_field.boundary[pbmap[f]]
            end
        end
    end

    return temps
end

"""
    solve_conjugate_ht(
        cht_prob::ConjugateHeatTransferProblem{Dim, T};
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        verbose = false,
    ) -> Tuple{SolveResult{Dim, T}, ThermalState{T}, CollocatedScalarField{T}}

Solve a conjugate heat transfer problem using Dirichlet-Neumann iteration.

The algorithm:
1. Initialize interface temperature to `T_ref`
2. Solve fluid (SIMPLE + energy) with interface temperature as Dirichlet BC
3. Compute heat flux at interface from fluid temperature gradient
4. Solve solid conduction with heat flux as Neumann BC
5. Extract new interface temperature from solid solution
6. Under-relax and check convergence
7. Repeat until converged or max iterations reached

Returns: (fluid_result, fluid_thermal_state, solid_temperature_field)
"""
function solve_conjugate_ht(
        cht_prob::ConjugateHeatTransferProblem{Dim, T};
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    fluid_prob = cht_prob.fluid_prob
    fluid_mesh = fluid_prob.mesh
    solid_mesh = cht_prob.solid_mesh
    alpha_coupling = T(0.5)  # under-relaxation for interface temperature

    # Initialize interface temperature
    T_interface = cht_prob.fluid_thermal.T_ref

    fluid_result = nothing
    thermal_state = nothing
    solid_T = nothing

    for coupling_iter in 1:cht_prob.max_coupling_iterations
        T_interface_old = T_interface

        # ── 1. Fluid solve with Dirichlet at interface ──────────────
        fluid_bcs_T = copy(cht_prob.fluid_bcs_T)
        fluid_bcs_T[cht_prob.interface_fluid_patch] = ParabolicDirichlet(T_interface)

        fluid_result, thermal_state = solve_simple_thermal(
            fluid_prob, cht_prob.fluid_thermal;
            bcs_T = fluid_bcs_T,
            turb_model = turb_model,
            turb_bcs = turb_bcs,
            T_init = T_interface,
            linear_solver = linear_solver,
        )

        # ── 2. Compute interface heat flux from fluid ───────────────
        q_interface = compute_interface_heat_flux(
            thermal_state.T_field, cht_prob.fluid_thermal.k,
            fluid_mesh, cht_prob.interface_fluid_patch,
        )

        # Average heat flux for the solid Neumann BC
        if !isempty(q_interface)
            q_avg = sum(values(q_interface)) / length(q_interface)
        else
            q_avg = zero(T)
        end

        # ── 3. Solid solve with Neumann at interface ────────────────
        solid_bcs_T = copy(cht_prob.solid_bcs_T)
        solid_bcs_T[cht_prob.interface_solid_patch] = ParabolicNeumann(q_avg)

        solid_T = solve_solid_conduction(
            solid_mesh, cht_prob.solid_thermal, solid_bcs_T;
            linear_solver = linear_solver,
        )

        # ── 4. Extract interface temperature from solid ─────────────
        solid_interface_temps = _extract_interface_temperatures(
            solid_T, solid_mesh, cht_prob.interface_solid_patch,
        )

        if !isempty(solid_interface_temps)
            T_interface_new = sum(values(solid_interface_temps)) / length(solid_interface_temps)
        else
            T_interface_new = T_interface
        end

        # ── 5. Under-relax ──────────────────────────────────────────
        T_interface = (one(T) - alpha_coupling) * T_interface_old +
            alpha_coupling * T_interface_new

        # ── 6. Check convergence ────────────────────────────────────
        delta_T = abs(T_interface - T_interface_old)

        if verbose
            println("CHT iter ", lpad(coupling_iter, 3),
                ": T_interface = ", round(T_interface; digits = 4),
                "  delta_T = ", round(delta_T; sigdigits = 3))
        end

        if delta_T < cht_prob.coupling_tolerance
            break
        end
    end

    return (fluid_result, thermal_state, solid_T)
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["types", "energy_equation", "buoyancy", "solid_conduction", "conjugate"]; include("src/thermal/$f.jl"); end; println("OK")'
```

---

### Task 7: Create solvers.jl — Thermal SIMPLE/PISO/PIMPLE wrappers

**Files:**
- Create: `src/thermal/solvers.jl`

- [ ] **Step 1: Write thermal solver wrappers**

Write `src/thermal/solvers.jl`:

```julia
# thermal/solvers.jl — Thermal SIMPLE/PISO/PIMPLE solver wrappers
#
# Extends the incompressible solvers with energy equation and optional
# buoyancy coupling. Follows the Phase 2a turbulence wrapper pattern.

using Printf: @sprintf

"""
    solve_simple_thermal(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T};
        bcs_T,
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        T_init = thermal_props.T_ref,
        linear_solver = nothing,
        verbose = false,
    ) -> Tuple{SolveResult{Dim, T}, ThermalState{T}}

Solve steady incompressible flow with energy equation using SIMPLE.

Each iteration:
1. Update effective conductivity and thermal diffusivity
2. Compute buoyancy force (if β > 0)
3. Assemble + solve momentum with `nu_eff` and `body_force`
4. Pressure solve + correction
5. Solve turbulence (if turbulence model provided)
6. Assemble + solve energy equation
7. Check convergence
"""
function solve_simple_thermal(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T};
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        T_init::Real = thermal_props.T_ref,
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Initialize flow state
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    # Initialize thermal state
    thermal_state = ThermalState(mesh; T_init = T(T_init), k_init = thermal_props.k)

    # Initialize turbulence (optional)
    turb_state = nothing
    if turb_model !== nothing
        turb_state = RANSTurbulenceState(turb_model, mesh)
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
    end

    # Residual tracking
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

        # ── Buoyancy ────────────────────────────────────────────
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
            solve_turbulence!(
                turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs;
                linear_solver = linear_solver,
            )
            turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
        end

        # ── Energy equation ─────────────────────────────────────
        T_eq = CollocatedEquation(mesh)
        assemble_energy!(T_eq, thermal_state.T_field, state.phi, alpha_eff, mesh, bcs_T)
        T_sol = _solve_linear(to_linear_problem(T_eq), linear_solver)
        for c in 1:nc
            thermal_state.T_field.internal[c] = T_sol.u[c]
        end

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
    return (result, thermal_state)
end

"""
    solve_incompressible_thermal(
        prob, thermal_props, tspan, dt;
        bcs_T, turb_model, turb_bcs, T_init, linear_solver, verbose,
    ) -> Tuple{SolveResult, ThermalState}

Solve transient incompressible flow with energy equation using PISO or PIMPLE.
"""
function solve_incompressible_thermal(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T},
        tspan::Tuple{T, T},
        dt::T;
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        T_init::Real = thermal_props.T_ref,
        save_every::Int = 1,
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    thermal_state = ThermalState(mesh; T_init = T(T_init), k_init = thermal_props.k)

    turb_state = nothing
    if turb_model !== nothing
        turb_state = RANSTurbulenceState(turb_model, mesh)
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
    end

    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    t_start, t_end = tspan
    t = t_start
    n_steps = 0

    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)

        # Effective properties
        nu_t_vec = turb_state === nothing ? nothing : turb_state.nu_t
        update_k_eff!(thermal_state, thermal_props, nu_t_vec, prob.density)
        nu_eff = turb_state === nothing ? prob.nu : compute_nu_eff(prob.nu, turb_state.nu_t)
        alpha_eff = compute_alpha_eff(thermal_state.k_eff, prob.density, thermal_props.Cp)
        body_force = compute_buoyancy_source(thermal_state.T_field, thermal_props, prob.density)

        # Flow step with thermal coupling
        if prob.algorithm isa PISO
            _thermal_piso_step!(state, prob, dt_actual, prob.algorithm.n_correctors,
                nu_eff, body_force; linear_solver = linear_solver)
        elseif prob.algorithm isa PIMPLE
            _thermal_pimple_step!(state, prob, dt_actual, nu_eff, body_force;
                linear_solver = linear_solver)
        end

        # Turbulence update
        if turb_model !== nothing
            solve_turbulence!(turb_state, turb_model, state.U, state.phi,
                prob.nu, mesh, turb_bcs; dt = dt_actual, linear_solver = linear_solver)
            turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
        end

        # Energy equation
        T_eq = CollocatedEquation(mesh)
        assemble_energy!(T_eq, thermal_state.T_field, state.phi, alpha_eff, mesh, bcs_T;
            dt = dt_actual)
        T_sol = _solve_linear(to_linear_problem(T_eq), linear_solver)
        for c in 1:nc
            thermal_state.T_field.internal[c] = T_sol.u[c]
        end

        t += dt_actual
        n_steps += 1

        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)

        if verbose && n_steps % max(1, round(Int, (t_end - t_start) / dt / 20)) == 0
            println("Step ", lpad(n_steps, 6), "  t=", @sprintf("%.4e", t),
                "  cont=", @sprintf("%.3e", r_cont))
        end
    end

    result = SolveResult{Dim, T}(true, n_steps, residuals, state)
    return (result, thermal_state)
end

# ── Thermal PISO step ────────────────────────────────────────────────

function _thermal_piso_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T, n_correctors::Int,
        nu_eff::Union{T, Vector{T}},
        body_force::Union{Nothing, Vector{SVector{Dim, T}}};
        linear_solver = nothing,
    ) where {Dim, T}
    mesh = prob.mesh

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

    for k in 1:n_correctors
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = _solve_linear(to_linear_problem(p_eq), linear_solver)

        nc = length(mesh.cell_volumes)
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

# ── Thermal PIMPLE step ──────────────────────────────────────────────

function _thermal_pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T,
        nu_eff::Union{T, Vector{T}},
        body_force::Union{Nothing, Vector{SVector{Dim, T}}};
        linear_solver = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::PIMPLE{T}
    mesh = prob.mesh

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

        nc = length(mesh.cell_volumes)
        for k in 1:algo.n_correctors
            p_eq = CollocatedEquation(mesh)
            assemble_pressure!(p_eq, state, prob)
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
julia --project -e 'using FiniteVolumeMethod; for f in ["types", "energy_equation", "buoyancy", "solid_conduction", "conjugate", "solvers"]; include("src/thermal/$f.jl"); end; println("OK")'
```

---

### Task 8: Wire into module — Layer 2 includes + exports

**Files:**
- Modify: `src/layers/discretization_assembly_kernels.jl`
- Modify: `src/FiniteVolumeMethod.jl`

- [ ] **Step 1: Add includes to Layer 2**

Append to `src/layers/discretization_assembly_kernels.jl` after the turbulence includes (after line 157 `include("../turbulence/solvers.jl")`):

```julia
# Conjugate Heat Transfer & Buoyancy (Phase 3)
# Depends on Phase 0 operators + Phase 1 incompressible + Phase 2a turbulence.
include("../thermal/types.jl")
include("../thermal/energy_equation.jl")
include("../thermal/buoyancy.jl")
include("../thermal/solid_conduction.jl")
include("../thermal/conjugate.jl")
include("../thermal/solvers.jl")
```

- [ ] **Step 2: Add exports to FiniteVolumeMethod.jl**

Add a new export block after the Phase 2a RANS exports (after `turbulence_wall_bc` on line 349) and before `export FVMGeometry`:

```julia
# --- Conjugate Heat Transfer & Buoyancy (Phase 3) ---
export
    # Types
    FluidThermalProperties,
    SolidThermalProperties,
    ThermalState,
    ConjugateHeatTransferProblem,
    # Energy equation
    assemble_energy!,
    update_k_eff!,
    compute_alpha_eff,
    # Buoyancy
    compute_buoyancy_source,
    has_buoyancy,
    # Solid conduction
    assemble_solid_conduction!,
    solve_solid_conduction,
    # Conjugate
    solve_conjugate_ht,
    compute_interface_heat_flux,
    # Solver wrappers
    solve_simple_thermal,
    solve_incompressible_thermal,
    # BC convenience
    thermal_inlet_bc,
    thermal_insulated_bc,
    thermal_heated_wall_bc,
    thermal_convective_bc
```

- [ ] **Step 3: Verify module loads**

```bash
julia --project -e 'using FiniteVolumeMethod; println("Phase 3 loaded: ", FluidThermalProperties)'
```
Expected: Prints the type name.

- [ ] **Step 4: Commit**

```bash
git add src/thermal/ src/incompressible/momentum.jl src/layers/discretization_assembly_kernels.jl src/FiniteVolumeMethod.jl
git commit -m "feat: add conjugate heat transfer, buoyancy, and energy equation (Phase 3)"
```

---

### Task 9: Write tests

**Files:**
- Create: `test/thermal.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the test file**

Create `test/thermal.jl` with the mesh builder copied from `test/incompressible.jl` (since `safe_include` runs in isolated modules). Include tests for:

1. **Type construction** — FluidThermalProperties, SolidThermalProperties, ThermalState defaults
2. **BC convenience** — thermal_inlet_bc, thermal_insulated_bc, thermal_heated_wall_bc, thermal_convective_bc
3. **compute_alpha_eff** — verify k_eff/(rho*Cp) arithmetic
4. **update_k_eff!** — with and without nu_t
5. **Buoyancy source** — verify F = -rho*beta*(T-Tref)*g, and nothing when beta=0
6. **Energy equation smoke** — assemble on 4x4 mesh, verify A nonzero
7. **Solid conduction** — solve steady 1D-like conduction with T_left=400, T_right=300, verify linear profile
8. **solve_simple_thermal smoke** — 8x4 channel with heated inlet, run 5 iterations, verify T field is finite and bounded
9. **PISO thermal smoke** — 2 time steps, verify returns correct types
10. **compute_interface_heat_flux** — set up known T field, verify flux sign and magnitude

- [ ] **Step 2: Register test in runtests.jl**

Add `safe_include("thermal.jl")` after the `turbulence_rans.jl` test.

- [ ] **Step 3: Run tests**

```bash
julia --project=test test/thermal.jl
```

- [ ] **Step 4: Run Runic**

```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "src/thermal/"])'
julia --project -e 'using Runic; Runic.main(["--inplace", "test/thermal.jl"])'
```

- [ ] **Step 5: Commit**

```bash
git add test/thermal.jl test/runtests.jl
git commit -m "test: add thermal/heat transfer test suite"
```

---

### Task 10: Register in validation manifest + final verification

**Files:**
- Modify: `validation/manifest.toml`

- [ ] **Step 1: Add conjugate_heat_transfer feature**

Append to `validation/manifest.toml`:

```toml
# ── Phase 3: Conjugate Heat Transfer & Buoyancy ────────────────────

[[features]]
feature = "conjugate_heat_transfer"
maturity = "experimental"
validation = "smoke_tested"
role = "research_tooling"
solver_family = "collocated"
precision_policy = "float64_cpu_reference"
random_seed_policy = "deterministic"
backend_policy = "cpu_reference"
required_ladder_stages = ["verification", "benchmark"]
summary = "Fluid energy equation, Boussinesq buoyancy, solid conduction, and Dirichlet-Neumann conjugate heat transfer coupling."
limitations = [
  "Experimental — validated via smoke tests only; heated cavity and conjugate benchmarks pending.",
  "Constant fluid properties (rho, Cp) only; temperature-dependent properties deferred.",
  "Conjugate coupling uses scalar interface temperature (face-averaged); per-face mapping deferred.",
]
```

- [ ] **Step 2: Verify all exports**

```bash
julia --project -e '
using FiniteVolumeMethod
@assert isdefined(FiniteVolumeMethod, :FluidThermalProperties)
@assert isdefined(FiniteVolumeMethod, :SolidThermalProperties)
@assert isdefined(FiniteVolumeMethod, :ThermalState)
@assert isdefined(FiniteVolumeMethod, :ConjugateHeatTransferProblem)
@assert isdefined(FiniteVolumeMethod, :solve_simple_thermal)
@assert isdefined(FiniteVolumeMethod, :compute_buoyancy_source)
@assert isdefined(FiniteVolumeMethod, :solve_conjugate_ht)
@assert isdefined(FiniteVolumeMethod, :solve_solid_conduction)
println("All Phase 3 exports verified")
'
```

- [ ] **Step 3: Run thermal tests**

```bash
julia --project=test test/thermal.jl
```

- [ ] **Step 4: Regression check**

```bash
julia --project=test test/incompressible.jl && julia --project=test test/turbulence_rans.jl
```
Expected: 94 + 127 tests pass.

- [ ] **Step 5: Run Runic check**

```bash
julia --project -e 'using Runic; Runic.main(["--check", "src/thermal/"])'
```

- [ ] **Step 6: Commit**

```bash
git add validation/manifest.toml
git commit -m "feat: register conjugate_heat_transfer in validation manifest"
```

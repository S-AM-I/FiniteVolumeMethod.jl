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
- `T_ref::T` — reference temperature [K] (also enthalpy datum: `h(T_ref) = 0`
  for constant-Cp formulation)
- `g::SVector{Dim, T}` — gravity vector [m/s²]
- `use_enthalpy::Bool` — if `true` the solver wrappers transport enthalpy
  `h` instead of temperature `T`. For constant `Cp` the two forms are
  equivalent up to a constant shift; the enthalpy form is preferable for
  high-Mach compressible and for future variable-Cp extensions (Stage v3
  fast-path Wave 1). Defaults to `false` (T-form).
"""
struct FluidThermalProperties{Dim, T}
    Cp::T
    k::T
    Pr_t::T
    beta::T
    T_ref::T
    g::SVector{Dim, T}
    use_enthalpy::Bool
end

"""
    FluidThermalProperties{Dim}(; Cp, k, Pr_t, beta, T_ref, g, use_enthalpy)

Construct fluid thermal properties with keyword arguments.
When `beta == 0` (default), buoyancy is disabled.
When `use_enthalpy == true`, the enthalpy form of the energy equation
is assembled by the thermal solver wrappers.
"""
function FluidThermalProperties{Dim}(;
        Cp::Real = 1005.0,
        k::Real = 0.026,
        Pr_t::Real = 0.85,
        beta::Real = 0.0,
        T_ref::Real = 300.0,
        g = nothing,
        use_enthalpy::Bool = false,
    ) where {Dim}
    T = promote_type(typeof(Cp), typeof(k), typeof(Pr_t), typeof(beta), typeof(T_ref))
    if g === nothing
        g_vec = Dim == 2 ? SVector{2, T}(zero(T), T(-9.81)) : SVector{3, T}(zero(T), zero(T), T(-9.81))
    else
        g_vec = SVector{Dim, T}(g)
    end
    return FluidThermalProperties{Dim, T}(
        T(Cp), T(k), T(Pr_t), T(beta), T(T_ref), g_vec, use_enthalpy,
    )
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
        coupling_tolerance::T = T(1.0e-4),
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

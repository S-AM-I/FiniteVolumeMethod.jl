# incompressible/boundary_conditions.jl — Boundary condition types for incompressible NS
#
# Defines high-level BC types (FixedVelocityBC, NoSlipWallBC, etc.) and
# expansion functions that convert them into the primitive ParabolicDirichlet /
# ParabolicNeumann conditions consumed by the collocated operators.

# ── Boundary condition types ────────────────────────────────────────

@doc """
    FixedVelocityBC{Dim, T} <: AbstractBoundaryCondition

Prescribe a fixed velocity vector on a boundary patch.  Pressure gets
a zero-gradient (Neumann) condition.

# Fields
- `value::SVector{Dim, T}` — prescribed velocity vector
"""
struct FixedVelocityBC{Dim, T} <: AbstractBoundaryCondition
    value::SVector{Dim, T}
end

@doc """
    FixedVelocityBC(value::NTuple{Dim, T})

Construct a [`FixedVelocityBC`](@ref) from a tuple.
"""
function FixedVelocityBC(value::NTuple{Dim, T}) where {Dim, T}
    return FixedVelocityBC{Dim, T}(SVector{Dim, T}(value))
end

@doc """
    FixedPressureBC{T} <: AbstractBoundaryCondition

Prescribe a fixed pressure value on a boundary patch.  Velocity gets
a zero-gradient (Neumann) condition.

# Fields
- `value::T` — prescribed pressure
"""
struct FixedPressureBC{T} <: AbstractBoundaryCondition
    value::T
end

@doc """
    NoSlipWallBC <: AbstractBoundaryCondition

No-slip wall: velocity = 0 (Dirichlet), pressure zero-gradient (Neumann).
"""
struct NoSlipWallBC <: AbstractBoundaryCondition end

@doc """
    SlipWallBC <: AbstractBoundaryCondition

Slip wall: velocity zero-gradient (Neumann), pressure zero-gradient (Neumann).
Tangential velocity is unconstrained; normal flux is zero via the face flux.
"""
struct SlipWallBC <: AbstractBoundaryCondition end

@doc """
    InletOutletBC{Dim, T} <: AbstractBoundaryCondition

Inlet-outlet boundary: acts as a fixed velocity inlet (Dirichlet on U)
with zero-gradient pressure.

# Fields
- `inlet_value::SVector{Dim, T}` — velocity applied when flow is into the domain
"""
struct InletOutletBC{Dim, T} <: AbstractBoundaryCondition
    inlet_value::SVector{Dim, T}
end

# ── Velocity BC expansion ──────────────────────────────────────────

@doc """
    expand_velocity_bc(bc, component::Int) -> AbstractBoundaryCondition

Convert an incompressible BC into the primitive `ParabolicDirichlet` or
`ParabolicNeumann` for the given velocity component equation.
"""
function expand_velocity_bc end

function expand_velocity_bc(bc::FixedVelocityBC, component::Int)
    return ParabolicDirichlet(bc.value[component])
end

function expand_velocity_bc(::NoSlipWallBC, ::Int)
    return ParabolicDirichlet(0.0)
end

function expand_velocity_bc(::SlipWallBC, ::Int)
    return ParabolicNeumann(0.0)
end

function expand_velocity_bc(bc::FixedPressureBC, ::Int)
    return ParabolicNeumann(0.0)
end

function expand_velocity_bc(bc::InletOutletBC, component::Int)
    return ParabolicDirichlet(bc.inlet_value[component])
end

# ── Pressure BC expansion ──────────────────────────────────────────

@doc """
    expand_pressure_bc(bc) -> AbstractBoundaryCondition

Convert an incompressible BC into the primitive `ParabolicDirichlet` or
`ParabolicNeumann` for the pressure equation.
"""
function expand_pressure_bc end

function expand_pressure_bc(bc::FixedPressureBC)
    return ParabolicDirichlet(bc.value)
end

function expand_pressure_bc(::FixedVelocityBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(::NoSlipWallBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(::SlipWallBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(::InletOutletBC)
    return ParabolicNeumann(0.0)
end

# ── Batch expansion helpers ─────────────────────────────────────────

@doc """
    expand_bcs_velocity(bcs::Dict{Symbol, <:AbstractBoundaryCondition}, component::Int)

Expand all incompressible BCs to primitive velocity BCs for the given
component.  Returns a `Dict{Symbol, AbstractBoundaryCondition}`.
"""
function expand_bcs_velocity(
        bcs::Dict{Symbol, <:AbstractBoundaryCondition}, component::Int,
    )
    return Dict{Symbol, AbstractBoundaryCondition}(
        name => expand_velocity_bc(bc, component)
            for (name, bc) in bcs
    )
end

@doc """
    expand_bcs_pressure(bcs::Dict{Symbol, <:AbstractBoundaryCondition})

Expand all incompressible BCs to primitive pressure BCs.
Returns a `Dict{Symbol, AbstractBoundaryCondition}`.
"""
function expand_bcs_pressure(
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
    )
    return Dict{Symbol, AbstractBoundaryCondition}(
        name => expand_pressure_bc(bc)
            for (name, bc) in bcs
    )
end

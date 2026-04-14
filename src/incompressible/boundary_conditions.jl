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

@doc """
    ZeroGradientBC <: AbstractBoundaryCondition

Zero-gradient (Neumann(0)) for both velocity and pressure.
Equivalent to OpenFOAM's `zeroGradient` condition.
"""
struct ZeroGradientBC <: AbstractBoundaryCondition end

@doc """
    TotalPressureBC{T} <: AbstractBoundaryCondition

Total pressure inlet: ``p_0 = p + \\tfrac{1}{2}|\\mathbf{U}|^2``.
Expands to Dirichlet for pressure using the specified total pressure value.
The full dynamic-pressure correction requires the velocity field and is
applied during the solve loop (future enhancement); the initial expansion
uses ``p_0`` directly.

# Fields
- `p0::T` — total pressure value
"""
struct TotalPressureBC{T} <: AbstractBoundaryCondition
    p0::T
end

@doc """
    SymmetryBC <: AbstractBoundaryCondition

Symmetry plane.  Zero normal velocity, zero-gradient for tangential
velocity and pressure.  Expands as zero-gradient (Neumann) for both
velocity and pressure; the normal-velocity constraint is enforced by the
face-flux treatment in the SIMPLE/PISO loop.
"""
struct SymmetryBC <: AbstractBoundaryCondition end

@doc """
    FlowRateInletBC{Dim, T} <: AbstractBoundaryCondition

Fixed volume-flow-rate inlet.  Stores a bulk velocity vector that
corresponds to the desired flow rate divided by the patch area:
``\\mathbf{U} = (Q / A)\\,\\hat{\\mathbf{n}}``.

# Fields
- `velocity::SVector{Dim, T}` — bulk velocity vector
"""
struct FlowRateInletBC{Dim, T} <: AbstractBoundaryCondition
    velocity::SVector{Dim, T}
end

@doc """
    FlowRateInletBC(velocity::NTuple{Dim, T})

Construct a [`FlowRateInletBC`](@ref) from a tuple.
"""
function FlowRateInletBC(velocity::NTuple{Dim, T}) where {Dim, T}
    return FlowRateInletBC{Dim, T}(SVector{Dim, T}(velocity))
end

@doc """
    TimeDependentVelocityBC{Dim, T, F} <: AbstractBoundaryCondition

Time-dependent velocity inlet.  Wraps a callable `func(t)` that returns
an `SVector{Dim, T}`.  The initial expansion evaluates `func(t_ref)`;
time-dependent updates during the solve loop are a future enhancement.

# Fields
- `func::F` — callable `t -> SVector{Dim, T}`
- `t_ref::T` — reference time used for the initial expansion (default `0.0`)
"""
struct TimeDependentVelocityBC{Dim, T, F} <: AbstractBoundaryCondition
    func::F
    t_ref::T
end

@doc """
    TimeDependentVelocityBC{Dim, T}(func) where {Dim, T}

Construct a [`TimeDependentVelocityBC`](@ref) with `t_ref = 0`.
"""
function TimeDependentVelocityBC{Dim, T}(func) where {Dim, T}
    return TimeDependentVelocityBC{Dim, T, typeof(func)}(func, zero(T))
end

@doc """
    WallFunctionBC{T} <: AbstractBoundaryCondition

Wall function BC for turbulent flows.  Velocity uses a Robin BC
approximation (expanded as Neumann(0) — the wall shear stress is
computed by the turbulence model during the solve loop).  Pressure
gets zero-gradient (Neumann).

# Fields
- `roughness::T` — surface roughness height [m] (`0` = hydraulically smooth)
"""
struct WallFunctionBC{T} <: AbstractBoundaryCondition
    roughness::T
end

@doc """
    WallFunctionBC(; roughness = 0.0)

Construct a [`WallFunctionBC`](@ref).  Default is a smooth wall.
"""
WallFunctionBC(; roughness = 0.0) = WallFunctionBC(roughness)

@doc """
    ConvectiveOutletBC <: AbstractBoundaryCondition

Convective (advective) outlet boundary condition.  Prevents wave
reflections at outflow boundaries by advecting interior values out of
the domain.  Expands as zero-gradient (Neumann) for both velocity and
pressure; the convective correction is applied in the solve loop.
Equivalent to OpenFOAM's `advective` or `convective` outlet.
"""
struct ConvectiveOutletBC <: AbstractBoundaryCondition end

@doc """
    PressureInletVelocityBC{T} <: AbstractBoundaryCondition

Pressure-driven inlet: pressure is fixed (Dirichlet) and velocity is
derived from the pressure gradient (Neumann).  Use when the inlet
pressure is known but the resulting velocity profile should develop
naturally.

# Fields
- `p_value::T` — prescribed inlet pressure
"""
struct PressureInletVelocityBC{T} <: AbstractBoundaryCondition
    p_value::T
end

@doc """
    CyclicBC <: AbstractBoundaryCondition

Cyclic (periodic) boundary condition.  Paired patches map to each
other so that the flow leaving one patch enters the partner patch.

The SIMPLE/PISO/PIMPLE solver loops automatically detect `CyclicBC`
entries, match face pairs via [`match_cyclic_faces`](@ref), and apply
cross-boundary cell coupling via [`apply_cyclic_bc!`](@ref) to all
assembled equations (momentum, pressure, and optionally turbulence,
thermal, species, alpha).

The individual BC expansion functions (`expand_velocity_bc`,
`expand_pressure_bc`) return `Neumann(0)` since the actual coupling
is applied at the equation level after assembly.

# Fields
- `partner_patch::Symbol` — name of the partner boundary patch
"""
struct CyclicBC <: AbstractBoundaryCondition
    partner_patch::Symbol
end

@doc """
    CustomBC{T} <: AbstractBoundaryCondition

User-defined boundary condition with explicit Dirichlet/Neumann
specification for velocity and pressure.  The most flexible BC type —
equivalent to OpenFOAM's `codedFixedValue` / `codedMixed`.

# Fields
- `velocity_type::Symbol`  — `:dirichlet` or `:neumann`
- `velocity_value::T`      — value per component (or gradient)
- `pressure_type::Symbol`  — `:dirichlet` or `:neumann`
- `pressure_value::T`      — pressure value (or gradient)
"""
struct CustomBC{T} <: AbstractBoundaryCondition
    velocity_type::Symbol
    velocity_value::T
    pressure_type::Symbol
    pressure_value::T
end

@doc """
    CustomBC(; velocity_type = :dirichlet, velocity_value = 0.0,
               pressure_type = :neumann, pressure_value = 0.0)

Construct a [`CustomBC`](@ref) with keyword arguments.
"""
function CustomBC(;
        velocity_type = :dirichlet, velocity_value = 0.0,
        pressure_type = :neumann, pressure_value = 0.0,
    )
    return CustomBC(velocity_type, velocity_value, pressure_type, pressure_value)
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

function expand_velocity_bc(::ZeroGradientBC, ::Int)
    return ParabolicNeumann(0.0)
end

function expand_velocity_bc(::TotalPressureBC, ::Int)
    return ParabolicNeumann(0.0)
end

function expand_velocity_bc(::SymmetryBC, ::Int)
    return ParabolicNeumann(0.0)
end

function expand_velocity_bc(bc::FlowRateInletBC, component::Int)
    return ParabolicDirichlet(bc.velocity[component])
end

function expand_velocity_bc(bc::TimeDependentVelocityBC, component::Int)
    return ParabolicDirichlet(bc.func(bc.t_ref)[component])
end

function expand_velocity_bc(::WallFunctionBC, ::Int)
    return ParabolicNeumann(0.0)
end

function expand_velocity_bc(::ConvectiveOutletBC, ::Int)
    return ParabolicNeumann(0.0)
end

function expand_velocity_bc(::PressureInletVelocityBC, ::Int)
    return ParabolicNeumann(0.0)
end

function expand_velocity_bc(::CyclicBC, ::Int)
    return ParabolicNeumann(0.0)
end

function expand_velocity_bc(bc::CustomBC, ::Int)
    if bc.velocity_type === :dirichlet
        return ParabolicDirichlet(bc.velocity_value)
    else
        return ParabolicNeumann(bc.velocity_value)
    end
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

function expand_pressure_bc(::ZeroGradientBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(bc::TotalPressureBC)
    return ParabolicDirichlet(bc.p0)
end

function expand_pressure_bc(::SymmetryBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(::FlowRateInletBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(::TimeDependentVelocityBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(::WallFunctionBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(::ConvectiveOutletBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(bc::PressureInletVelocityBC)
    return ParabolicDirichlet(bc.p_value)
end

function expand_pressure_bc(::CyclicBC)
    return ParabolicNeumann(0.0)
end

function expand_pressure_bc(bc::CustomBC)
    if bc.pressure_type === :dirichlet
        return ParabolicDirichlet(bc.pressure_value)
    else
        return ParabolicNeumann(bc.pressure_value)
    end
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

# ── Cyclic BC helpers ────────���─────────────────────────────────────

"""
    collect_cyclic_pairs(bcs, mesh) -> Vector{Vector{Tuple{Int, Int}}}

Scan `bcs` for `CyclicBC` entries, match face pairs using
[`match_cyclic_faces`](@ref), and return a vector of matched-pair lists
(one per cyclic patch pair).  Returns an empty vector when no cyclic
BCs are present.

Each cyclic pair is matched only once: the patch that appears first
alphabetically is treated as `patch1`.
"""
function collect_cyclic_pairs(
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    all_pairs = Vector{Vector{Tuple{Int, Int}}}()
    seen = Set{Symbol}()

    for (name, bc) in bcs
        bc isa CyclicBC || continue
        name in seen && continue
        partner = bc.partner_patch
        partner in seen && continue

        # Mark both patches as handled
        push!(seen, name)
        push!(seen, partner)

        pairs = match_cyclic_faces(mesh, name, partner)
        push!(all_pairs, pairs)
    end

    return all_pairs
end

"""
    apply_cyclic_to_equation!(eq, field, mesh, cyclic_pairs)

Apply cyclic coupling to an assembled equation for all cyclic patch pairs.
No-op when `cyclic_pairs` is empty.
"""
function apply_cyclic_to_equation!(
        eq::CollocatedEquation{T},
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}},
    ) where {Dim, T}
    for pairs in cyclic_pairs
        apply_cyclic_bc!(eq, field, mesh, pairs)
    end
    return nothing
end

"""
    _make_scalar_field(values, state) -> CollocatedScalarField

Create a temporary `CollocatedScalarField` from a vector of cell values,
using the boundary face indices from `state.p`.  Used internally to pass
velocity component data to `apply_cyclic_bc!`.
"""
function _make_scalar_field(
        values::Vector{T},
        state::IncompressibleState{Dim, T},
    ) where {Dim, T}
    bfi = state.p.boundary_face_indices
    return CollocatedScalarField{T}(
        :_tmp, values, zeros(T, length(bfi)), bfi,
    )
end

"""
    update_boundary_cyclic!(state, mesh, cyclic_pairs)

Update boundary face values for cyclic patches by copying from the
partner cell.  For velocity, the partner cell's internal velocity is
copied to the boundary face.  For pressure, similarly.
"""
function update_boundary_cyclic!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}},
    ) where {Dim, T}
    isempty(cyclic_pairs) && return nothing

    # Build reverse lookup: boundary face index → position in boundary array
    ubmap_U = Dict(f => i for (i, f) in enumerate(state.U.boundary_face_indices))
    ubmap_p = Dict(f => i for (i, f) in enumerate(state.p.boundary_face_indices))

    for pairs in cyclic_pairs
        for (f1, f2) in pairs
            c1 = owner(mesh, f1)
            c2 = owner(mesh, f2)

            # f1 boundary gets c2's value, f2 boundary gets c1's value
            if haskey(ubmap_U, f1)
                state.U.boundary[ubmap_U[f1]] = state.U.internal[c2]
            end
            if haskey(ubmap_U, f2)
                state.U.boundary[ubmap_U[f2]] = state.U.internal[c1]
            end
            if haskey(ubmap_p, f1)
                state.p.boundary[ubmap_p[f1]] = state.p.internal[c2]
            end
            if haskey(ubmap_p, f2)
                state.p.boundary[ubmap_p[f2]] = state.p.internal[c1]
            end
        end
    end

    return nothing
end

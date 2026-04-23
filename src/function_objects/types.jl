# function_objects/types.jl — Runtime function objects (Stage 7d)
#
# Analog to OpenFOAM's `functionObjects` subsystem: user-level runtime
# hooks for monitoring (forces, probes, field averages) and for
# expression-driven boundary conditions. All function objects share a
# `run(fo, state, t, iter)` interface and are invoked from the solver's
# post-iteration callback site.
#
# The expression-BC support uses a simple closure-based eval rather
# than a string-parsed DSL — Julia lets users pass `(x, t, u) -> …`
# directly, which is both more efficient and safer than an `eval`-based
# DSL. A full string-expression parser is a Stage 7d follow-up for
# users who need to drive BCs from config files.

using StaticArrays: SVector
using LinearAlgebra: norm, dot

"""
    AbstractFunctionObject

Umbrella for runtime function objects. Concrete subtypes implement
`run!(fo, state, t, iter)` which is invoked after each solver iteration
by the caller. Typical subtypes: force probes, line samplers, field
statistics, expression-BC drivers.
"""
abstract type AbstractFunctionObject end

# ── Probes ───────────────────────────────────────────────────────────

"""
    PointProbe{Dim, T, Fn}

Sample a scalar field at a Cartesian probe point every time `run!` is
called. `cell_index` is resolved once at construction; values are
pushed into `history`.

# Fields
- `position::SVector{Dim, T}`
- `cell_index::Int`
- `extract::Fn` — `state -> value` accessor (closure over which field to probe).
- `history::Vector{Tuple{T, T}}` — (time, value) pairs.
- `name::Symbol`
"""
mutable struct PointProbe{Dim, T, Fn} <: AbstractFunctionObject
    position::SVector{Dim, T}
    cell_index::Int
    extract::Fn
    history::Vector{Tuple{T, T}}
    name::Symbol
end

function PointProbe(
        name::Symbol, position::SVector{Dim, T}, cell_index::Int, extract::Fn,
    ) where {Dim, T, Fn}
    return PointProbe{Dim, T, Fn}(position, cell_index, extract, Tuple{T, T}[], name)
end

function run!(fo::PointProbe{Dim, T, Fn}, state, t::T, iter::Int) where {Dim, T, Fn}
    push!(fo.history, (t, fo.extract(state, fo.cell_index)))
    return nothing
end

# ── Force / moment probes ────────────────────────────────────────────

"""
    ForceProbe{Dim, T, Fn}

Sum of pressure + viscous forces on a list of boundary faces — a
drag/lift/side-force monitor. `compute_force` is a user closure
`(state, face_indices) -> SVector{Dim, T}` that does the actual
integration (standard OpenFOAM forceCoeffs computes the same thing).
"""
mutable struct ForceProbe{Dim, T, Fn} <: AbstractFunctionObject
    face_indices::Vector{Int}
    compute_force::Fn
    history::Vector{Tuple{T, SVector{Dim, T}}}
    name::Symbol
end

function ForceProbe(
        name::Symbol, face_indices::Vector{Int}, compute_force::Fn, ::Val{Dim}, ::Type{T},
    ) where {Dim, T, Fn}
    return ForceProbe{Dim, T, Fn}(
        face_indices, compute_force, Tuple{T, SVector{Dim, T}}[], name,
    )
end

function run!(fo::ForceProbe{Dim, T, Fn}, state, t::T, iter::Int) where {Dim, T, Fn}
    F = fo.compute_force(state, fo.face_indices)
    push!(fo.history, (t, F))
    return nothing
end

# ── Expression-BC support ────────────────────────────────────────────

"""
    ExpressionBC{Dim, T, Fn}

A boundary condition whose prescribed value is computed at runtime
from a closure `(x, t) -> value`, where `x` is a face-center position
and `t` is the current time. Enables time- or space-dependent BCs
(e.g. pulsating inlets, Womersley flow, travelling-wave perturbations)
without recompiling the solver.

The attached closure must be thread-safe and allocation-free — the
solver may call it inside the inner iteration loop.

Used as a drop-in replacement for a plain Dirichlet BC by evaluating
the closure at each face's center every iteration.
"""
struct ExpressionBC{Dim, T, Fn} <: AbstractFVMBoundaryCondition
    fn::Fn
end
ExpressionBC(fn::Fn, ::Val{Dim}, ::Type{T}) where {Dim, T, Fn} =
    ExpressionBC{Dim, T, Fn}(fn)

"""
    evaluate_expression_bc(bc::ExpressionBC, x, t) -> value

Evaluate the BC expression at face center `x` and time `t`.
"""
@inline evaluate_expression_bc(bc::ExpressionBC, x, t) = bc.fn(x, t)

# ── Field statistics ─────────────────────────────────────────────────

"""
    FieldStatistics{T}

Compute per-cell running time-averages of a scalar field. After `N`
iterations, `mean[c] = (1/N) Σ_iter field[c]`.
Used for turbulence post-processing (〈u〉, 〈p〉, 〈u'u'〉).
"""
mutable struct FieldStatistics{T} <: AbstractFunctionObject
    name::Symbol
    mean::Vector{T}
    n_samples::Int
end
FieldStatistics(name::Symbol, n_cells::Int, ::Type{T} = Float64) where {T} =
    FieldStatistics{T}(name, zeros(T, n_cells), 0)

function update!(stats::FieldStatistics{T}, field::AbstractVector{T}) where {T}
    length(field) == length(stats.mean) || error("field size mismatch")
    n = stats.n_samples + 1
    @inbounds for c in eachindex(field)
        stats.mean[c] = stats.mean[c] + (field[c] - stats.mean[c]) / n
    end
    stats.n_samples = n
    return nothing
end

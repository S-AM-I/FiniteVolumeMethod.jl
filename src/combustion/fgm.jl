# combustion/fgm.jl — Flamelet-Generated Manifold (FGM) tabulated chemistry
#
# FGM tabulates precomputed flamelet solutions on a 2D grid of
# (progress variable C, mixture fraction Z) ∈ [0, 1]². Each table entry
# stores the per-species mass fractions Y_i (and, in richer variants,
# temperature / heat-release / production rates). At runtime the
# solver interpolates the table instead of integrating detailed
# kinetics — a ~10³× speed-up for turbulent-flame calculations.
#
# Real FGM tables are built via Cantera flamelet solutions; that step
# lives in `ext/FVMCanteraExt.jl`. This file owns the table type, the
# interpolation kernel, and a callback-driven builder so the V&V suite
# can exercise the runtime path without Cantera installed.

"""
    FGMTable{NS, T}

Bilinear lookup table indexed by progress variable `C ∈ [0, 1]` and
mixture fraction `Z ∈ [0, 1]`, returning `NS` species mass fractions.

# Fields
- `C_grid::Vector{T}` — `NC` equispaced progress-variable nodes on `[0, 1]`
- `Z_grid::Vector{T}` — `NZ` equispaced mixture-fraction nodes on `[0, 1]`
- `Y::Array{T, 3}` — size `(NC, NZ, NS)` species mass-fraction table
"""
struct FGMTable{NS, T}
    C_grid::Vector{T}
    Z_grid::Vector{T}
    Y::Array{T, 3}

    function FGMTable{NS, T}(
            C_grid::AbstractVector, Z_grid::AbstractVector, Y::AbstractArray{<:Real, 3},
        ) where {NS, T}
        size(Y, 1) == length(C_grid) ||
            error("FGMTable: Y first dim ($(size(Y, 1))) must match length(C_grid) ($(length(C_grid)))")
        size(Y, 2) == length(Z_grid) ||
            error("FGMTable: Y second dim ($(size(Y, 2))) must match length(Z_grid) ($(length(Z_grid)))")
        size(Y, 3) == NS ||
            error("FGMTable: Y third dim ($(size(Y, 3))) must equal NS=$NS")
        return new{NS, T}(Vector{T}(C_grid), Vector{T}(Z_grid), Array{T, 3}(Y))
    end
end

"""
    build_fgm_table_from_callback(f, NC, NZ; Ttype = Float64) -> FGMTable{NS, Ttype}

Build an `FGMTable` by sampling the caller-supplied function
`f(C, Z) -> NTuple{NS, Real}` on an `NC × NZ` uniform grid covering
`[0, 1]²`.

Convenient for tests and for mock Cantera substitutes; the full
Cantera-driven builder lives in `FVMCanteraExt`.
"""
function build_fgm_table_from_callback(
        f, NC::Int, NZ::Int; Ttype::Type = Float64,
    )
    NC >= 2 || error("build_fgm_table_from_callback: NC must be ≥ 2, got $NC")
    NZ >= 2 || error("build_fgm_table_from_callback: NZ must be ≥ 2, got $NZ")
    C_grid = collect(range(Ttype(0), Ttype(1); length = NC))
    Z_grid = collect(range(Ttype(0), Ttype(1); length = NZ))

    # Probe once to determine NS.
    first_sample = f(C_grid[1], Z_grid[1])
    first_sample isa Tuple ||
        error("build_fgm_table_from_callback: f must return a Tuple, got $(typeof(first_sample))")
    NS = length(first_sample)

    Y = Array{Ttype, 3}(undef, NC, NZ, NS)
    @inbounds for iZ in 1:NZ, iC in 1:NC
        sample = f(C_grid[iC], Z_grid[iZ])
        length(sample) == NS ||
            error("build_fgm_table_from_callback: f returned $(length(sample)) species at ($(iC),$(iZ)), expected $NS")
        for s in 1:NS
            Y[iC, iZ, s] = Ttype(sample[s])
        end
    end

    return FGMTable{NS, Ttype}(C_grid, Z_grid, Y)
end

# ── Interpolation ─────────────────────────────────────────────────

@inline function _fgm_bracket(grid::Vector{T}, x::T) where {T}
    n = length(grid)
    x_clamped = clamp(x, grid[1], grid[end])
    # Binary search for i such that grid[i] <= x_clamped <= grid[i+1].
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if grid[mid] <= x_clamped
            lo = mid
        else
            hi = mid
        end
    end
    i = lo
    dx = grid[i + 1] - grid[i]
    t = dx > zero(T) ? (x_clamped - grid[i]) / dx : zero(T)
    return i, t
end

"""
    lookup_fgm(table, C, Z) -> NTuple{NS, T}

Bilinearly interpolate the `FGMTable` at progress variable `C` and
mixture fraction `Z`. Out-of-range inputs are clamped to `[0, 1]` — no
extrapolation.
"""
function lookup_fgm(table::FGMTable{NS, T}, C::Real, Z::Real) where {NS, T}
    iC, tC = _fgm_bracket(table.C_grid, T(C))
    iZ, tZ = _fgm_bracket(table.Z_grid, T(Z))
    return ntuple(Val(NS)) do s
        y00 = table.Y[iC, iZ, s]
        y10 = table.Y[iC + 1, iZ, s]
        y01 = table.Y[iC, iZ + 1, s]
        y11 = table.Y[iC + 1, iZ + 1, s]
        return (one(T) - tC) * (one(T) - tZ) * y00 +
            tC * (one(T) - tZ) * y10 +
            (one(T) - tC) * tZ * y01 +
            tC * tZ * y11
    end
end

"""
    lookup_fgm!(Y_out, table, C, Z)

In-place bilinear-interpolation variant of [`lookup_fgm`](@ref) that
writes into `Y_out`, an `AbstractVector` of length `NS`.
"""
function lookup_fgm!(
        Y_out::AbstractVector{T}, table::FGMTable{NS, T}, C::Real, Z::Real,
    ) where {NS, T}
    length(Y_out) == NS ||
        error("lookup_fgm!: Y_out length $(length(Y_out)) ≠ NS=$NS")
    sample = lookup_fgm(table, C, Z)
    @inbounds for s in 1:NS
        Y_out[s] = sample[s]
    end
    return Y_out
end

# ── Cantera stub ──────────────────────────────────────────────────

"""
    compute_fgm_table_from_cantera(mechanism, NC, NZ, fuel, oxidizer; kwargs...) -> FGMTable

Solve a 1D counterflow flamelet using Cantera to fill an FGM table.
Requires the weak dependency `Cantera.jl`; otherwise this stub errors.
The real implementation lives in `ext/FVMCanteraExt.jl`.
"""
function compute_fgm_table_from_cantera(
        mechanism, NC::Int, NZ::Int,
        fuel::AbstractString, oxidizer::AbstractString; kwargs...,
    )
    return error(
        "compute_fgm_table_from_cantera requires Cantera.jl — add `using Cantera` to enable the FVMCanteraExt extension.",
    )
end

# porous/types.jl — Porous-media momentum source (Stage 6c)
#
# Adds a Darcy / Darcy-Forchheimer momentum-sink to the incompressible
# pressure-based solver so cells inside a porous region experience the
# equivalent bulk drag:
#
#   S_u = -ρ · ( D · ν + F · |u|/2 ) u
#
# where `D` is the viscous (Darcy) resistance tensor [1/m²], `F` is the
# inertial (Forchheimer) resistance tensor [1/m], and the formula is
# written per unit volume.
#
# For isotropic media D and F reduce to scalars. Orthotropic media use
# a diagonal tensor aligned with the principal axes of the porous
# structure.
#
# Reference: Whitaker (1996), "The Forchheimer equation: a theoretical
# development"; standard treatment in every industrial CFD code.

using StaticArrays: SVector, SMatrix

"""
    AbstractPorousModel{Dim, T}

Umbrella for porous-zone resistance models. Concrete subtypes evaluate
per-cell momentum sink `momentum_source(model, c, u_cell, ρ, μ)` where
`c` is the global cell index.
"""
abstract type AbstractPorousModel{Dim, T} end

"""
    DarcyPorous{Dim, T}(; cell_indices, D = 0.0) <: AbstractPorousModel

Isotropic Darcy resistance `S = -ρ D μ u` per unit volume. `D` is the
inverse permeability scalar (1/m²); `μ` is the local dynamic viscosity.

For the (vanishingly common) anisotropic case use `OrthotropicPorous`.
"""
struct DarcyPorous{Dim, T} <: AbstractPorousModel{Dim, T}
    cell_indices::Vector{Int}
    D::T
end
DarcyPorous{Dim}(; cell_indices::Vector{Int}, D::Real = 0.0) where {Dim} =
    DarcyPorous{Dim, typeof(float(D))}(cell_indices, float(D))

function porous_momentum_source(
        model::DarcyPorous{Dim, T}, c::Int,
        u_cell::SVector{Dim, T}, rho::T, mu::T,
    ) where {Dim, T}
    in(c, model.cell_indices) || return zero(SVector{Dim, T})
    return -rho * mu * model.D * u_cell
end

"""
    DarcyForchheimerPorous{Dim, T}(; cell_indices, D = 0.0, F = 0.0) <: AbstractPorousModel

Isotropic Darcy-Forchheimer: `S = -ρ (D μ + F |u|/2) u`.
- `D::T` — viscous (Darcy) resistance (1/m²).
- `F::T` — inertial (Forchheimer) resistance (1/m).

At low Reynolds number the quadratic Forchheimer term vanishes and the
model reduces to Darcy. At high Re it dominates.
"""
struct DarcyForchheimerPorous{Dim, T} <: AbstractPorousModel{Dim, T}
    cell_indices::Vector{Int}
    D::T
    F::T
end
function DarcyForchheimerPorous{Dim}(;
        cell_indices::Vector{Int},
        D::Real = 0.0, F::Real = 0.0,
    ) where {Dim}
    T = promote_type(typeof(float(D)), typeof(float(F)))
    return DarcyForchheimerPorous{Dim, T}(cell_indices, T(D), T(F))
end

function porous_momentum_source(
        model::DarcyForchheimerPorous{Dim, T}, c::Int,
        u_cell::SVector{Dim, T}, rho::T, mu::T,
    ) where {Dim, T}
    in(c, model.cell_indices) || return zero(SVector{Dim, T})
    u_mag = sqrt(sum(ui -> ui^2, u_cell))
    coeff = model.D * mu + T(0.5) * model.F * u_mag
    return -rho * coeff * u_cell
end

"""
    OrthotropicPorous{Dim, T}(; cell_indices, D_diag, F_diag) <: AbstractPorousModel

Orthotropic (diagonal tensor) Darcy-Forchheimer. Principal axes are
assumed aligned with the mesh axes (x, y, z); a rotated porous
structure requires pre-rotating the cell-level velocity — future Stage 6
follow-up.

# Fields
- `D_diag::SVector{Dim, T}` — diagonal entries of the Darcy tensor.
- `F_diag::SVector{Dim, T}` — diagonal entries of the Forchheimer tensor.
"""
struct OrthotropicPorous{Dim, T} <: AbstractPorousModel{Dim, T}
    cell_indices::Vector{Int}
    D_diag::SVector{Dim, T}
    F_diag::SVector{Dim, T}
end

function porous_momentum_source(
        model::OrthotropicPorous{Dim, T}, c::Int,
        u_cell::SVector{Dim, T}, rho::T, mu::T,
    ) where {Dim, T}
    in(c, model.cell_indices) || return zero(SVector{Dim, T})
    u_mag = sqrt(sum(ui -> ui^2, u_cell))
    source = zero(SVector{Dim, T})
    @inbounds for d in 1:Dim
        coeff = model.D_diag[d] * mu + T(0.5) * model.F_diag[d] * u_mag
        source = Base.setindex(source, -rho * coeff * u_cell[d], d)
    end
    return source
end

# ---------------------------------------------------------------------------
# v3.0 fast-path API: `PorousZone` holds the full permeability tensor
# `K` [m²] and Forchheimer coefficient tensor `F` [1/m], matching the
# OpenFOAM `explicitPorositySource` convention. The momentum sink is
#
#   S_U = − ( μ · K⁻¹  +  0.5 · ρ · F · |U| ) · U
#
# where the first (Darcy) term is linear in `U` and the second
# (Forchheimer) term is quadratic in `|U|`. Both `K` and `F` are
# general 3×3 SMatrices so anisotropic / off-diagonal porous media are
# supported with no extra code.
# ---------------------------------------------------------------------------

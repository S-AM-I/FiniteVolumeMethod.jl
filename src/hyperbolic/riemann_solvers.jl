using StaticArrays: SVector

"""
    AbstractRiemannSolver

Abstract supertype for approximate Riemann solvers used at cell interfaces.

All subtypes must support calling via `solve_riemann(solver, law, wL, wR, dir)`.
"""
abstract type AbstractRiemannSolver end

"""
    solve_riemann(solver::AbstractRiemannSolver, law, wL, wR, dir) -> SVector

Compute the numerical flux at a cell interface given left and right primitive states
`wL`, `wR`, and direction `dir` (1=x, 2=y, 3=z).

Returns the numerical flux vector.
"""
function solve_riemann end

# ============================================================
# Lax-Friedrichs (Rusanov) Solver
# ============================================================

"""
    LaxFriedrichsSolver <: AbstractRiemannSolver

The Lax-Friedrichs (Rusanov) approximate Riemann solver.

The numerical flux is:
  `F* = ½(F(UL) + F(UR)) - ½ λ_max (UR - UL)`
where `λ_max = max(|vL| + cL, |vR| + cR)`.

This is the most diffusive but most robust Riemann solver.
"""
struct LaxFriedrichsSolver <: AbstractRiemannSolver end

function solve_riemann(::LaxFriedrichsSolver, law, wL::SVector{N}, wR::SVector{N}, dir::Int) where {N}
    fL = physical_flux(law, wL, dir)
    fR = physical_flux(law, wR, dir)
    uL = primitive_to_conserved(law, wL)
    uR = primitive_to_conserved(law, wR)
    λ_max = max(max_wave_speed(law, wL, dir), max_wave_speed(law, wR, dir))
    return 0.5 * (fL + fR) - 0.5 * λ_max * (uR - uL)
end

# ============================================================
# HLL Solver
# ============================================================

"""
    HLLSolver <: AbstractRiemannSolver

The Harten-Lax-van Leer (HLL) approximate Riemann solver.

Uses estimates of the fastest left-going and right-going wave speeds `SL`, `SR` to
compute a two-wave flux:

If `SL ≥ 0`: `F* = FL`
If `SR ≤ 0`: `F* = FR`
Otherwise: `F* = (SR*FL - SL*FR + SL*SR*(UR - UL)) / (SR - SL)`
"""
struct HLLSolver <: AbstractRiemannSolver end

function solve_riemann(::HLLSolver, law, wL::SVector{N}, wR::SVector{N}, dir::Int) where {N}
    fL = physical_flux(law, wL, dir)
    fR = physical_flux(law, wR, dir)
    uL = primitive_to_conserved(law, wL)
    uR = primitive_to_conserved(law, wR)

    # Wave speed estimates (Davis estimates)
    λ_minL, λ_maxL = wave_speeds(law, wL, dir)
    λ_minR, λ_maxR = wave_speeds(law, wR, dir)
    SL = min(λ_minL, λ_minR)
    SR = max(λ_maxL, λ_maxR)

    if SL >= zero(SL)
        return fL
    elseif SR <= zero(SR)
        return fR
    else
        return (SR * fL - SL * fR + SL * SR * (uR - uL)) / (SR - SL)
    end
end

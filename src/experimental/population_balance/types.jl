# population_balance/types.jl — Shared types for Population Balance Modelling
#
# Solvers of the Population Balance Equation (PBE)
#
#   ∂n(L, x, t)/∂t + ∇·(U·n) = S(L, x, t)
#
# close the infinite moment hierarchy (or avoid it entirely) with one of
# three methods exposed here:
#
# - `QMoM`  — Quadrature Method of Moments (transport 2N moments,
#              recover (w_i, L_i) via Wheeler / product-difference).
# - `DQMoM` — Direct QMoM (transport N weights + N weighted abscissae,
#              relate sources to moment sources via a linear solve).
# - `ClassMethod` — Discrete Class / sectional method (transport one
#                    scalar per size-class bin).
#
# Reference: Marchisio & Fox (2013), Computational Models for
# Polydisperse Particulate and Multiphase Systems (Cambridge).

"""
    PBMMethod

Trait supertype for all population-balance closure methods. Concrete
subtypes are `QMoM`, `DQMoM`, and `ClassMethod`.
"""
abstract type PBMMethod end

"""
    QMoM{T}(N::Int)

Quadrature Method of Moments with `N` quadrature nodes. Tracks `2N`
moments `m_0, m_1, …, m_{2N-1}` per CFD cell and reconstructs the
quadrature pairs `(w_i, L_i)` on demand via the Wheeler algorithm.
"""
struct QMoM{T} <: PBMMethod
    N::Int
end
QMoM(N::Int) = QMoM{Float64}(N)

"""
    DQMoM{T}(N::Int)

Direct Quadrature Method of Moments. Transports `N` weights `w_i` and
`N` weighted abscissae `ζ_i = w_i · L_i` directly; quadrature sources
`(a_i, b_i)` are recovered from moment sources via the linear system

    A * x = s

where A is the `2N × 2N` Vandermonde-like kernel and `s` contains the
first `2N` moment sources. See `dqmom_sources` for the kernel.
"""
struct DQMoM{T} <: PBMMethod
    N::Int
end
DQMoM(N::Int) = DQMoM{Float64}(N)

"""
    ClassMethod{T}(N_class::Int, L_edges::Vector{T})

Sectional (class / discrete) method with `N_class` size-class bins whose
edges are `L_edges` (length `N_class + 1`, monotonically increasing).
Each bin tracks a scalar number density `n_i` (per CFD cell) and has a
representative length `L_i = 0.5 (L_edges[i] + L_edges[i+1])` and volume
`V_i = (π/6) · L_i^3` (sphere-equivalent).
"""
struct ClassMethod{T} <: PBMMethod
    N_class::Int
    L_edges::Vector{T}
    L_centers::Vector{T}
    V_centers::Vector{T}

    function ClassMethod{T}(N_class::Int, L_edges::AbstractVector{T}) where {T}
        length(L_edges) == N_class + 1 ||
            error("ClassMethod: L_edges must have length N_class+1")
        issorted(L_edges) || error("ClassMethod: L_edges must be sorted")
        L_centers = T[(L_edges[i] + L_edges[i + 1]) / 2 for i in 1:N_class]
        V_centers = T[(pi / 6) * L^3 for L in L_centers]
        return new{T}(N_class, collect(L_edges), L_centers, V_centers)
    end
end
function ClassMethod(N_class::Int, L_min::Real, L_max::Real; spacing::Symbol = :geometric)
    T = promote_type(typeof(float(L_min)), typeof(float(L_max)))
    L_min > 0 || error("ClassMethod: L_min must be positive")
    L_max > L_min || error("ClassMethod: L_max must exceed L_min")
    edges = if spacing === :geometric
        T[L_min * (L_max / L_min)^((i - 1) / N_class) for i in 1:(N_class + 1)]
    elseif spacing === :linear
        T[L_min + (L_max - L_min) * (i - 1) / N_class for i in 1:(N_class + 1)]
    else
        error("ClassMethod: unknown spacing $(spacing) (use :linear or :geometric)")
    end
    return ClassMethod{T}(N_class, edges)
end

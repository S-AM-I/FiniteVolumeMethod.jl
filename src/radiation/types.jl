# radiation/types.jl — Core types for radiation modeling
#
# Defines the radiation model hierarchy, the P1 model, radiation state,
# and boundary condition convenience constructors.

"""Stefan-Boltzmann constant [W/(m^2 K^4)]."""
const STEFAN_BOLTZMANN = 5.670374419e-8

"""
    AbstractRadiationModel

Supertype for radiation models.
"""
abstract type AbstractRadiationModel end

"""
    P1Model{T, A} <: AbstractRadiationModel

P1 radiation approximation. Solves a single diffusion equation for the
incident radiation field G:

    -div(Gamma * grad(G)) + a * G = 4 * a * sigma * T^4

where `Gamma = 1/(3a)` and `a` is the absorption coefficient.

# Fields
- `a::A` --- absorption coefficient [1/m]: scalar `T` for uniform,
  or `Vector{T}` for per-cell spatially varying absorption.
"""
struct P1Model{T, A <: Union{T, Vector{T}}} <: AbstractRadiationModel
    a::A
end

"""
    P1Model(; a = 0.1)

Construct a P1 radiation model. `a` may be a scalar (uniform) or a
`Vector` (per-cell) absorption coefficient.
"""
function P1Model(; a = 0.1)
    if a isa AbstractVector
        T = eltype(a)
        return P1Model{T, Vector{T}}(Vector{T}(a))
    else
        return P1Model{Float64, Float64}(Float64(a))
    end
end

"""
    RadiationState{T}

Mutable state for radiation models. Holds the incident radiation field.

# Fields
- `G::CollocatedScalarField{T}` --- incident radiation [W/m^2]
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

# -- WSGGM (Weighted-Sum-of-Grey-Gases) --------------------------------------

"""
    WSGGMModel{NB, T} <: AbstractRadiationModel

Weighted-Sum-of-Grey-Gases Model. Represents the non-grey emissivity of a
combustion-product mixture (CO2 + H2O) as a weighted sum of `NB` grey
gases each with absorption coefficient `kappa_i` and temperature-dependent
weight `a_i(T)`:

    epsilon(T, L) = Σ_i a_i(T) * (1 - exp(-kappa_i * L))

with the non-participating ("window") band carrying the complementary
weight so `Σ_i a_i(T) = 1` for every `T`.

The default coefficients are Smith, Shen & Friedman (1982), "Evaluation of
Coefficients for the Weighted Sum of Gray Gases Model", *J. Heat Transfer*
104, 602-608, for a pressure-path product `p_w + p_c = 1 atm·m` and
`p_w / p_c = 2`. Three grey gases plus the window band (NB = 4).

# Fields
- `kappa::NTuple{NB, T}` --- absorption coefficients [1/m]. The first
  entry is 0 (window band).
- `b::NTuple{NB, NTuple{4, T}}` --- polynomial coefficients so
  `a_i(T) = Σ_{j=0}^{3} b[i][j+1] * T^j`. The window weight is derived as
  `a_w(T) = 1 - Σ_{i≠w} a_i(T)` at runtime but the polynomial coeffs are
  also stored for the three non-window bands.
"""
struct WSGGMModel{NB, T} <: AbstractRadiationModel
    kappa::NTuple{NB, T}
    b::NTuple{NB, NTuple{4, T}}
end

"""
    WSGGMModel(; bands = :smith1982)

Construct a WSGGM model. `bands = :smith1982` selects the Smith, Shen &
Friedman (1982) 3-grey-gas + window-band set for `p_w / p_c = 2`.

Custom coefficient sets can be supplied by passing `kappa::NTuple` and
`b::NTuple{NB, NTuple{4, T}}` directly via the inner constructor.
"""
function WSGGMModel(; bands::Symbol = :smith1982)
    if bands === :smith1982
        # Smith, Shen & Friedman (1982), Table 1, p_w/p_c = 2, p·L = 1 atm·m.
        # Coefficients b_{i,j} so a_i(T) = b_{i,0} + b_{i,1}·T + b_{i,2}·T^2 + b_{i,3}·T^3
        # with T in Kelvin. The four tuple entries are (b0, b1, b2, b3).
        # Index 1 = window (kappa = 0, derived weight so Σ a = 1).
        # Indices 2-4 = three participating bands.
        T = Float64
        kappa = (T(0.0), T(0.4303), T(7.055), T(178.1))
        # Placeholder polynomial for the window band; in `compute_band_weight`
        # we derive a_w(T) = 1 - Σ_{i>1} a_i(T) so these coefficients are
        # unused, but we store them to keep the polynomial dispatch uniform.
        b_window = (T(0.0), T(0.0), T(0.0), T(0.0))
        b_band1 = (T(5.15e-1), T(-2.303e-4), T(9.779e-8), T(-1.494e-11))
        b_band2 = (T(7.749e-2), T(3.399e-4), T(-2.297e-7), T(3.77e-11))
        b_band3 = (T(1.907e-1), T(-1.824e-4), T(5.608e-8), T(-5.122e-12))
        b = (b_window, b_band1, b_band2, b_band3)
        return WSGGMModel{4, T}(kappa, b)
    else
        error("Unknown WSGGM coefficient set :$bands. Supported: :smith1982")
    end
end

"""
    compute_band_weight(model::WSGGMModel, T, i)

Polynomial evaluation `a_i(T) = Σ_j b[i][j+1] * T^j` for band `i > 1`.
For `i = 1` (the window band) returns `1 - Σ_{j>1} a_j(T)` so the full
weight vector sums to exactly one. Weights are clamped to be non-negative
for numerical robustness at extrapolated temperatures.
"""
function compute_band_weight(model::WSGGMModel{NB, T}, Tval::Real, i::Int) where {NB, T}
    Tf = T(Tval)
    if i == 1
        s = zero(T)
        for j in 2:NB
            s += _eval_poly(model.b[j], Tf)
        end
        return max(one(T) - s, zero(T))
    else
        return max(_eval_poly(model.b[i], Tf), zero(T))
    end
end

"""Evaluate `b[1] + b[2]*T + b[3]*T^2 + b[4]*T^3` via Horner."""
function _eval_poly(b::NTuple{4, T}, Tval::T) where {T}
    return b[1] + Tval * (b[2] + Tval * (b[3] + Tval * b[4]))
end

"""
    compute_band_emissivity(model::WSGGMModel, T, path)

Total emissivity at temperature `T` [K] for path length `path` [m]:

    epsilon(T, L) = Σ_i a_i(T) * (1 - exp(-kappa_i * L))

The window band (`kappa = 0`) contributes zero emissivity regardless of
its weight, consistent with Smith et al. 1982.
"""
function compute_band_emissivity(model::WSGGMModel{NB, T}, Tval::Real, path::Real) where {NB, T}
    Tf = T(Tval)
    L = T(path)
    eps = zero(T)
    for i in 1:NB
        a_i = compute_band_weight(model, Tf, i)
        kappa_i = model.kappa[i]
        eps += a_i * (one(T) - exp(-kappa_i * L))
    end
    return eps
end

# -- BC convenience constructors ------------------------------------------------

"""
    marshak_wall_bc(rad_model::P1Model, T_wall)

Marshak boundary condition for an opaque wall at temperature `T_wall`:
`G + (2/(3a)) * dG/dn = 4 * sigma * T_wall^4`

Implemented as `ParabolicRobin(1, 2/(3a), 4 * sigma * T^4)`.
"""
function marshak_wall_bc(rad_model::P1Model, T_wall::Real)
    a_val = rad_model.a isa AbstractVector ? sum(rad_model.a) / length(rad_model.a) : rad_model.a
    T_fl = typeof(Float64(a_val))
    b_coeff = T_fl(2) / (T_fl(3) * a_val)
    c_val = T_fl(4) * T_fl(STEFAN_BOLTZMANN) * T_fl(T_wall)^4
    return ParabolicRobin(one(T_fl), b_coeff, c_val)
end

"""
    radiation_inlet_bc(T_inlet)

Fixed incident radiation BC from a known temperature:
`G = 4 * sigma * T^4`
"""
function radiation_inlet_bc(T_inlet::Real)
    G_val = 4.0 * STEFAN_BOLTZMANN * Float64(T_inlet)^4
    return ParabolicDirichlet(G_val)
end

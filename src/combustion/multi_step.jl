# combustion/multi_step.jl — Multi-step Arrhenius mechanism for collocated solver
#
# Generalises the one-step `CollocatedArrheniusReaction` closure to an
# arbitrary list of elementary Arrhenius reactions:
#
#   Reaction r:  Σ_i ν_{r,i}^reactants S_i  →  Σ_i ν_{r,i}^products S_i
#   k_{f,r}(T)  = A_r · T^{b_r} · exp(-E_{a,r} / (R · T))
#   ω_i(x)      = Σ_r (ν_{r,i}^products - ν_{r,i}^reactants) · k_{f,r}(T) · ∏_j Y_j^{ν_{r,j}^reactants}
#
# Rates are mass-based (kg/(m³·s)) so the downstream species-transport
# integrator does not need to change. Stoichiometric coefficients are
# stored as floating-point matrices to keep gradient workflows
# (e.g. `SciMLSensitivity.jl`) differentiable.

# ── Type ──────────────────────────────────────────────────────────

"""
    MultiStepMechanism{NR, NS, T}

Multi-step Arrhenius reaction mechanism with `NR` elementary reactions
involving `NS` species.

The forward rate coefficient of reaction `r` is
`k_{f,r}(T) = A_r · T^{b_r} · exp(-E_{a,r} / (R · T))`. The net
production rate of species `i` is
`ω_i = Σ_r (ν_{r,i}^p - ν_{r,i}^r) · k_{f,r}(T) · ∏_j Y_j^{ν_{r,j}^r}`.

# Fields
- `A::NTuple{NR, T}` — pre-exponential factors
- `b::NTuple{NR, T}` — temperature exponents
- `E_a::NTuple{NR, T}` — activation energies [J/mol]
- `nu_reactants::Matrix{T}` — `NR × NS` reactant stoichiometric coefficients
- `nu_products::Matrix{T}` — `NR × NS` product stoichiometric coefficients
"""
struct MultiStepMechanism{NR, NS, T}
    A::NTuple{NR, T}
    b::NTuple{NR, T}
    E_a::NTuple{NR, T}
    nu_reactants::Matrix{T}
    nu_products::Matrix{T}
end

"""
    MultiStepMechanism(; A, b, E_a, nu_reactants, nu_products)

Construct a [`MultiStepMechanism`](@ref).

`A`, `b`, `E_a` must each be an `NTuple{NR}` of reals and
`nu_reactants`, `nu_products` must each be `NR × NS` matrices of
stoichiometric coefficients.
"""
function MultiStepMechanism(;
        A::NTuple{NR, <:Real},
        b::NTuple{NR, <:Real},
        E_a::NTuple{NR, <:Real},
        nu_reactants::AbstractMatrix{<:Real},
        nu_products::AbstractMatrix{<:Real},
    ) where {NR}
    size(nu_reactants) == size(nu_products) ||
        error("nu_reactants and nu_products must share the same shape")
    size(nu_reactants, 1) == NR ||
        error("stoichiometry matrix must have NR=$(NR) rows, got $(size(nu_reactants, 1))")
    NS = size(nu_reactants, 2)
    T = promote_type(
        eltype(A), eltype(b), eltype(E_a),
        eltype(nu_reactants), eltype(nu_products),
    )
    return MultiStepMechanism{NR, NS, T}(
        NTuple{NR, T}(A),
        NTuple{NR, T}(b),
        NTuple{NR, T}(E_a),
        Matrix{T}(nu_reactants),
        Matrix{T}(nu_products),
    )
end

"""
    one_step_arrhenius_mechanism(A, b, E_a, s; n_fuel = 1.0, n_ox = 1.0) -> MultiStepMechanism{1, 3, T}

Build a single-reaction [`MultiStepMechanism`](@ref) equivalent to the
existing one-step `CollocatedArrheniusReaction` closure for the
three-species `(fuel, oxidizer, product)` system with mass
stoichiometric ratio `s` (kg oxidizer per kg fuel).

Reactant stoichiometry uses the reaction order exponents
`(n_fuel, n_ox, 0)` so concentration products match the one-step form
`Y_fuel^{n_fuel} · Y_ox^{n_ox}`. Net product stoichiometry is
`(-1, -s, (1 + s))` per unit mass of fuel consumed, matching the
existing closure's sign conventions.
"""
function one_step_arrhenius_mechanism(
        A::Real, b::Real, E_a::Real, s::Real;
        n_fuel::Real = 1.0, n_ox::Real = 1.0,
    )
    T = promote_type(typeof(A), typeof(b), typeof(E_a), typeof(s), typeof(n_fuel), typeof(n_ox))
    nu_r = reshape(T[n_fuel, n_ox, zero(T)], 1, 3)
    # Net species rate for fuel = -1 (fuel consumed), for oxidizer = -s,
    # for product = +(1 + s). The rate expression uses (ν_p - ν_r) as
    # the net coefficient; with ν_r chosen above the net coefficient is
    # ν_p - ν_r. We pick ν_p = ν_r + net so the rate-evaluator algebra
    # reproduces the one-step closure exactly.
    net_fuel = -one(T)
    net_ox = -T(s)
    net_prod = one(T) + T(s)
    nu_p = similar(nu_r)
    nu_p[1, 1] = nu_r[1, 1] + net_fuel
    nu_p[1, 2] = nu_r[1, 2] + net_ox
    nu_p[1, 3] = nu_r[1, 3] + net_prod
    return MultiStepMechanism(;
        A = (T(A),), b = (T(b),), E_a = (T(E_a),),
        nu_reactants = nu_r, nu_products = nu_p,
    )
end

# ── Rate evaluation ──────────────────────────────────────────────

"""
    compute_multi_step_rates!(
        omega, mechanism, species_state, T_field, density, mesh,
    ) -> NTuple{NS, Vector{T}}

Fill `omega` in-place with the per-species per-cell mass-based
production rates from `mechanism`.

For each cell `c` and reaction `r`:

```
k_{f,r} = A_r · T_c^{b_r} · exp(-E_{a,r} / (R · T_c))
rate_r   = k_{f,r} · ∏_i max(Y_i[c], 0)^{ν_{r,i}^reactants}
ω_i[c] += ρ · (ν_{r,i}^products - ν_{r,i}^reactants) · rate_r
```

The temperature is clamped at 200 K to avoid underflow in `exp`, matching
the one-step Arrhenius closure.
"""
function compute_multi_step_rates!(
        omega::NTuple{NS, Vector{T}},
        mechanism::MultiStepMechanism{NR, NS, T},
        species_state::SpeciesState{NS, T},
        T_field::Union{CollocatedScalarField{T}, Vector{T}},
        density::T,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T, NR, NS}
    nc = length(mesh.cell_volumes)
    R = T(_R_UNIVERSAL)
    T_vals = T_field isa CollocatedScalarField ? T_field.internal : T_field

    for i in 1:NS
        fill!(omega[i], zero(T))
    end

    @inbounds for c in 1:nc
        T_c = _cell_T(T_vals, c)
        for r in 1:NR
            k_f = mechanism.A[r] * T_c^mechanism.b[r] *
                exp(-mechanism.E_a[r] / (R * T_c))
            rate = k_f
            for j in 1:NS
                nu_rj = mechanism.nu_reactants[r, j]
                if nu_rj > zero(T)
                    Y_j = max(species_state.Y[j].internal[c], zero(T))
                    rate *= Y_j^nu_rj
                end
            end
            for i in 1:NS
                net = mechanism.nu_products[r, i] - mechanism.nu_reactants[r, i]
                if net != zero(T)
                    omega[i][c] += density * net * rate
                end
            end
        end
    end

    return omega
end

"""
    compute_multi_step_rates(mechanism, species_state, T_field, density, mesh) -> NTuple{NS, Vector{T}}

Allocating variant of [`compute_multi_step_rates!`](@ref). Returns a
fresh `NTuple{NS, Vector{T}}` of reaction rates [kg/(m³·s)].
"""
function compute_multi_step_rates(
        mechanism::MultiStepMechanism{NR, NS, T},
        species_state::SpeciesState{NS, T},
        T_field::Union{CollocatedScalarField{T}, Vector{T}},
        density::T,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T, NR, NS}
    nc = length(mesh.cell_volumes)
    omega = ntuple(_ -> zeros(T, nc), Val(NS))
    return compute_multi_step_rates!(omega, mechanism, species_state, T_field, density, mesh)
end

# ── Shared helper ─────────────────────────────────────────────────

"""
    _cell_T(T_vals, c) -> T

Return the cell-`c` temperature clamped at 200 K. Shared with the
one-step Arrhenius closure for consistent low-temperature behaviour.
"""
@inline function _cell_T(T_vals::AbstractVector{T}, c::Int) where {T}
    return max(T_vals[c], T(200))
end

# ── Cantera stub ──────────────────────────────────────────────────

"""
    read_chemkin_mechanism(path) -> MultiStepMechanism

Parse a CHEMKIN-format mechanism file and build a
[`MultiStepMechanism`](@ref). Requires the weak dependency
`Cantera.jl` to be loaded; otherwise this stub errors. The real
implementation lives in `ext/FVMCanteraExt.jl`.
"""
function read_chemkin_mechanism(path::AbstractString)
    return error(
        "read_chemkin_mechanism requires Cantera.jl — add `using Cantera` to enable the FVMCanteraExt extension.",
    )
end

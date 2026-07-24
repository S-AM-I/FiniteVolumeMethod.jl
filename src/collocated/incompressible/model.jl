# incompressible/model.jl — physics composition for the collocated
# incompressible family.
#
# A single `IncompressibleModel` replaces the `turb_model=` / `thermal_props=` /
# `bcs_T=` / `bcs_G=` / `bcs_species=` / `rad_model=` / `combustion_props=`
# kwarg sprawl that `CommonSolve.solve` used to carry.  Each component bundles
# its own model object, boundary conditions, and initial data, and takes those
# boundary conditions as a *mandatory* keyword — so "thermal properties supplied
# but no temperature boundary conditions" is unrepresentable rather than caught
# by a hand-written check at solve time.
#
# The model rides in the problem (`prob.model`), matching the SciML convention
# that the problem fully specifies the physics while the algorithm specifies
# only the numerics.

# ── Components ──────────────────────────────────────────────────────

@doc """
    TurbulenceComponent(model; bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}())

Turbulence closure together with the boundary conditions for its transported
scalars.

`bcs` is keyed first by field name (`:k`, `:epsilon`, `:omega`, `:nuTilda`)
and then by patch name.  The default empty dictionary means every turbulence
field falls back to its solver default treatment.

# Fields
- `model` — RANS/LES/hybrid closure (e.g. `StandardKEpsilon()`, `WALE()`)
- `bcs` — nested boundary-condition dictionary
"""
struct TurbulenceComponent{M, B}
    model::M
    bcs::B
end

function TurbulenceComponent(
        model;
        bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
    )
    return TurbulenceComponent(model, bcs)
end

@doc """
    ThermalComponent(properties; bcs, T_init = properties.T_ref)

Energy equation: fluid thermal properties, temperature boundary conditions,
and the initial temperature field.

`bcs` is mandatory.  Solving an energy equation without temperature boundary
conditions is not a meaningful problem, so the requirement is enforced by the
constructor rather than checked at solve time.

# Fields
- `properties` — `FluidThermalProperties`
- `bcs` — temperature boundary conditions keyed by patch name
- `T_init` — uniform initial temperature
"""
struct ThermalComponent{P, B, T}
    properties::P
    bcs::B
    T_init::T
end

function ThermalComponent(properties; bcs, T_init = properties.T_ref)
    return ThermalComponent(properties, bcs, T_init)
end

@doc """
    RadiationComponent(model; bcs)

Radiative transfer model and the boundary conditions for incident radiation
`G`.

Requires a [`ThermalComponent`](@ref) alongside it in the
[`IncompressibleModel`](@ref): the radiative source term enters the energy
equation, so there is nothing for it to couple to without one.

# Fields
- `model` — `P1Model` or an fvDOM model
- `bcs` — incident-radiation boundary conditions keyed by patch name
"""
struct RadiationComponent{M, B}
    model::M
    bcs::B
end

RadiationComponent(model; bcs) = RadiationComponent(model, bcs)

@doc """
    CombustionComponent(properties, reaction; bcs, Y_init = Dict{Symbol, Float64}(), lewis = nothing)

Reacting-flow component: thermochemistry, a reaction-rate closure, species
boundary conditions, and initial mass fractions.

`reaction` selects the rate closure and therefore the solver path — an
`EddyDissipationModel` gives the mixing-limited EDM formulation, a
`MultiStepMechanism` the temperature-dependent finite-rate Arrhenius one.

`lewis` supplies a `VariableLewis` non-unity-Lewis-number species transport
closure; it is only available with a `MultiStepMechanism`.

# Fields
- `properties` — `CombustionProperties`
- `reaction` — `EddyDissipationModel` or `MultiStepMechanism`
- `bcs` — species boundary conditions, keyed by species then patch
- `Y_init` — initial mass fractions keyed by species name
- `lewis` — `VariableLewis` closure, or `nothing`
"""
struct CombustionComponent{P, R, B, Y, L}
    properties::P
    reaction::R
    bcs::B
    Y_init::Y
    lewis::L
end

function CombustionComponent(
        properties, reaction;
        bcs,
        Y_init = Dict{Symbol, Float64}(),
        lewis = nothing,
    )
    return CombustionComponent(properties, reaction, bcs, Y_init, lewis)
end

# ── Model ───────────────────────────────────────────────────────────

@doc """
    IncompressibleModel(; turbulence = nothing, thermal = nothing, radiation = nothing,
                          combustion = nothing, porous_zones = nothing, mrf_zones = nothing)

Composable physics attached to an [`IncompressibleProblem`](@ref).  Every
component is optional; the default model is plain incompressible flow.

Component dependencies are validated on construction, so an unsolvable
combination fails where it is written rather than deep inside a solve.

# Fields
- `turbulence` — [`TurbulenceComponent`](@ref)
- `thermal` — [`ThermalComponent`](@ref)
- `radiation` — [`RadiationComponent`](@ref), requires `thermal`
- `combustion` — [`CombustionComponent`](@ref), requires `thermal`
- `porous_zones` — `Vector{PorousZone}` for Darcy-Forchheimer resistance
- `mrf_zones` — `Vector{MRFZone}` for rotating reference frames

# Example
```julia
model = IncompressibleModel(
    thermal = ThermalComponent(props; bcs = bcs_T, T_init = 300.0),
    turbulence = TurbulenceComponent(StandardKEpsilon(); bcs = turb_bcs),
)
prob = IncompressibleProblem(mesh, bcs, SIMPLE(); nu = 1.0e-3, model = model)
```
"""
struct IncompressibleModel{Tu, Th, Ra, Co, PZ, MZ}
    turbulence::Tu
    thermal::Th
    radiation::Ra
    combustion::Co
    porous_zones::PZ
    mrf_zones::MZ
end

function IncompressibleModel(;
        turbulence = nothing,
        thermal = nothing,
        radiation = nothing,
        combustion = nothing,
        porous_zones = nothing,
        mrf_zones = nothing,
    )
    _validate_model(turbulence, thermal, radiation, combustion, porous_zones, mrf_zones)
    return IncompressibleModel(
        turbulence, thermal, radiation, combustion, porous_zones, mrf_zones,
    )
end

function _validate_model(turbulence, thermal, radiation, combustion, porous_zones, mrf_zones)
    if radiation !== nothing && thermal === nothing
        throw(
            ArgumentError(
                "radiation requires thermal: the radiative source term enters the " *
                    "energy equation, so a RadiationComponent has nothing to couple to " *
                    "without a ThermalComponent."
            )
        )
    end
    if combustion !== nothing && thermal === nothing
        throw(
            ArgumentError(
                "combustion requires thermal: heat release is applied to the energy " *
                    "equation, so a CombustionComponent needs a ThermalComponent."
            )
        )
    end
    if combustion !== nothing && radiation !== nothing
        throw(
            ArgumentError(
                "combustion and radiation cannot be combined: no reacting-flow solver " *
                    "in this package accepts a radiation model, so the radiative loss " *
                    "would be dropped silently — which for a flame is typically the " *
                    "dominant heat-loss term."
            )
        )
    end
    zones_requested = porous_zones !== nothing || mrf_zones !== nothing
    if zones_requested && (combustion !== nothing || radiation !== nothing)
        throw(
            ArgumentError(
                "porous_zones/mrf_zones are threaded through the plain, turbulent, and " *
                    "thermal solve paths only — not the combustion or radiation paths."
            )
        )
    end
    return nothing
end

# ── Traits ──────────────────────────────────────────────────────────
#
# Assembly hooks dispatch on these rather than on `!== nothing` checks
# scattered through the solver loops.

@doc """
    has_turbulence(model) -> Bool

Whether `model` carries a [`TurbulenceComponent`](@ref).
"""
has_turbulence(model::IncompressibleModel) = model.turbulence !== nothing

@doc """
    has_thermal(model) -> Bool

Whether `model` carries a [`ThermalComponent`](@ref).
"""
has_thermal(model::IncompressibleModel) = model.thermal !== nothing

@doc """
    has_radiation(model) -> Bool

Whether `model` carries a [`RadiationComponent`](@ref).
"""
has_radiation(model::IncompressibleModel) = model.radiation !== nothing

@doc """
    has_combustion(model) -> Bool

Whether `model` carries a [`CombustionComponent`](@ref).
"""
has_combustion(model::IncompressibleModel) = model.combustion !== nothing

@doc """
    has_porous_zones(model) -> Bool

Whether `model` carries Darcy-Forchheimer porous zones.
"""
has_porous_zones(model::IncompressibleModel) = model.porous_zones !== nothing

@doc """
    has_mrf_zones(model) -> Bool

Whether `model` carries rotating-reference-frame zones.
"""
has_mrf_zones(model::IncompressibleModel) = model.mrf_zones !== nothing

@doc """
    is_plain_flow(model) -> Bool

Whether `model` requests no physics beyond incompressible momentum and
continuity.  Zones do not count — they are momentum-source modifications
handled on the plain path.
"""
function is_plain_flow(model::IncompressibleModel)
    return !has_turbulence(model) && !has_thermal(model) &&
        !has_radiation(model) && !has_combustion(model)
end

# ── Component accessors (internal) ──────────────────────────────────
#
# The solver entry points still take separate `turb_model=`/`turb_bcs=` pairs,
# so the CommonSolve façade has to unpack the component; these keep that
# unpacking in one place until Stage 5e consolidates the loops.

"""
    turbulence_model(model) -> closure or `nothing`

The turbulence closure carried by `model`, or `nothing` when it has no
[`TurbulenceComponent`](@ref).
"""
turbulence_model(model::IncompressibleModel) =
    has_turbulence(model) ? model.turbulence.model : nothing

"""
    turbulence_bcs(model) -> Dict

Boundary conditions for `model`'s turbulence fields, or an empty dictionary
when it has no [`TurbulenceComponent`](@ref).
"""
function turbulence_bcs(model::IncompressibleModel)
    has_turbulence(model) && return model.turbulence.bcs
    return Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}()
end

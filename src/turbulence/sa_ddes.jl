# turbulence/sa_ddes.jl — Spalart-Allmaras-based DDES variant
#
# The original `DDES` (see `ddes.jl`) is a generic wrapper that can
# operate on top of any RANS base model; when instantiated with a
# `SpalartAllmaras` base model it produces the standard SA-DDES closure
# from Spalart et al. (2006). That generic path runs through a fallback
# on non-SA base models (which we never recommend) and bakes the length
# scale into a fresh `SpalartAllmaras` instance each call. For the v3.0
# correctness wave we ship a dedicated SA-specific DDES type that:
#
# 1. Locks the base model to `SpalartAllmaras` so the SA-specific length
#    scale `l_RANS = d_wall` is guaranteed correct (no generic fallback).
# 2. Uses the same `_ddes_shielding` / `_ddes_length_scale` primitives as
#    the parent DDES — no duplicated physics, just a strict typed wrapper.
# 3. Exposes `sa_ddes_blended_length` so the V&V suite can test the
#    blended length scale in isolation without running a full RANS solve.
#
# This sits *alongside* `DDES` (both are shipped) and serves as the
# foundation for the IDDES stub in `iddes.jl`.

"""
    SADDES{T} <: AbstractHybridModel

Spalart-Allmaras Delayed Detached Eddy Simulation. Type-locked
variant of the generic `DDES` wrapper that specifically requires a
`SpalartAllmaras` base model — protects downstream code from being
handed a DDES with an incompatible base (e.g. k-ε, for which the
length-scale blend in `ddes.jl` would silently reuse `d_wall`).

# Fields
- `base_model::SpalartAllmaras{T}` — SA RANS core
- `C_DES::T` — DES constant (default 0.65)
- `delta::Vector{T}` — grid filter width per cell
- `d_wall::Vector{T}` — wall distance per cell

# Example

```julia
sa = SpalartAllmaras(mesh, [:wall])
ddes = SADDES(sa, mesh, [:wall]; C_DES = 0.65)
```
"""
struct SADDES{T} <: AbstractHybridModel
    base_model::SpalartAllmaras{T}
    C_DES::T
    delta::Vector{T}
    d_wall::Vector{T}
end

"""
    SADDES(base, mesh, wall_patches; C_DES = 0.65)

Construct an SA-DDES wrapper with precomputed filter width and wall
distance.
"""
function SADDES(
        base::SpalartAllmaras{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol};
        C_DES::Real = 0.65,
    ) where {Dim, T}
    delta = compute_filter_width(mesh)
    d_wall = compute_wall_distance(mesh, wall_patches)
    return SADDES{T}(base, T(C_DES), delta, T.(d_wall))
end

# ── Interface implementation ─────────────────────────────────────────

n_turbulence_fields(model::SADDES) = n_turbulence_fields(model.base_model)
turbulence_field_names(model::SADDES) = turbulence_field_names(model.base_model)

"""
    sa_ddes_blended_length(d_wall, delta, f_d; C_DES = 0.65) -> l_blend

Pure-algebra helper computing the SA-DDES blended length scale
```
l_blend = d_wall − f_d · max(0, d_wall − C_DES·Δ)
```
Reused both by `solve_turbulence!` below and by the V&V test suite.
"""
function sa_ddes_blended_length(
        d_wall::T, delta::T, f_d::T;
        C_DES::T = T(0.65),
    ) where {T}
    l_LES = C_DES * delta
    return _ddes_length_scale(d_wall, l_LES, f_d)
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::SADDES{T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    turbulent_viscosity!(nu_t, model.base_model, turb_state, mesh)
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::SADDES{T},
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    kappa = model.base_model.kappa

    S_mag = compute_strain_rate(U, mesh)

    # Per-cell blended length scale
    l_blend = Vector{T}(undef, nc)
    for c in 1:nc
        f_d = _ddes_shielding(
            nu, turb_state.nu_t[c], model.d_wall[c], S_mag[c], kappa,
        )
        l_blend[c] = sa_ddes_blended_length(
            model.d_wall[c], model.delta[c], f_d; C_DES = model.C_DES,
        )
    end

    # Build a per-call SA model carrying the blended length scale as the
    # effective wall distance. Immutable — creates a fresh struct.
    sa = model.base_model
    modified = SpalartAllmaras{T}(
        sa.cb1, sa.cb2, sa.sigma, sa.kappa,
        sa.cw2, sa.cw3, sa.cv1, sa.ct3, sa.ct4,
        l_blend,
    )
    solve_turbulence!(
        turb_state, modified, U, phi, nu, mesh, bcs_turb;
        dt = dt, linear_solver = linear_solver,
        solver_config = solver_config,
    )
    return nothing
end

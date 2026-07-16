# turbulence/ddes.jl — Delayed Detached Eddy Simulation
#
# Hybrid RANS/LES that wraps a base RANS model (Spalart-Allmaras)
# and modifies the length scale to switch from RANS in the boundary
# layer to LES in separated regions.

"""
    DDES{B, T} <: AbstractHybridModel

Delayed Detached Eddy Simulation.

Wraps a base RANS model and modifies its turbulent length scale:
`l_DDES = l_RANS - f_d · max(0, l_RANS - l_LES)`

The shielding function `f_d` protects the boundary layer from
premature LES switching.

# Fields
- `base_model::B` — base RANS model (e.g., SpalartAllmaras)
- `C_DES::T` — DES constant (default 0.65)
- `delta::Vector{T}` — grid filter width per cell
- `d_wall::Vector{T}` — wall distance per cell
"""
struct DDES{B, T} <: AbstractHybridModel
    base_model::B
    C_DES::T
    delta::Vector{T}
    d_wall::Vector{T}
end

"""
    DDES(base_model, mesh, wall_patches; C_DES = 0.65)

Construct a DDES model from a base RANS model.
"""
function DDES(
        base_model,
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol};
        C_DES::Real = 0.65,
    ) where {Dim, T}
    delta = compute_filter_width(mesh)
    d_wall = compute_wall_distance(mesh, wall_patches)
    return DDES{typeof(base_model), T}(base_model, T(C_DES), delta, T.(d_wall))
end

# ── DDES interface ───────────────────────────────────────────────────

n_turbulence_fields(model::DDES) = n_turbulence_fields(model.base_model)
turbulence_field_names(model::DDES) = turbulence_field_names(model.base_model)

"""
    _ddes_shielding(nu, nu_t, d, S, kappa) -> f_d

Compute the DDES shielding function. Returns ~0 in boundary layer
(RANS mode) and ~1 in separated regions (LES mode).
"""
function _ddes_shielding(nu::T, nu_t::T, d::T, S::T, kappa::T) where {T}
    d_safe = max(d, T(1.0e-10))
    S_safe = max(S, T(1.0e-10))
    r_d = (nu + nu_t) / (kappa^2 * d_safe^2 * S_safe)
    return one(T) - tanh((T(8) * r_d)^3)
end

"""
    _ddes_length_scale(l_RANS, l_LES, f_d) -> l_DDES

Compute the DDES modified length scale.
"""
function _ddes_length_scale(l_RANS::T, l_LES::T, f_d::T) where {T}
    return l_RANS - f_d * max(zero(T), l_RANS - l_LES)
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::DDES{B, T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {B, Dim, T}
    # Delegate to base model for viscosity computation
    turbulent_viscosity!(nu_t, model.base_model, turb_state, mesh)
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::DDES{B, T},
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
    ) where {B, Dim, T}
    nc = length(mesh.cell_volumes)
    kappa = T(0.41)

    # Compute strain rate for shielding function
    S_mag = compute_strain_rate(U, mesh)

    # Compute DDES length scale per cell
    l_ddes = Vector{T}(undef, nc)
    for c in 1:nc
        l_RANS = model.d_wall[c]  # For SA-based DDES, l_RANS = d
        l_LES = model.C_DES * model.delta[c]
        f_d = _ddes_shielding(nu, turb_state.nu_t[c], model.d_wall[c], S_mag[c], kappa)
        l_ddes[c] = _ddes_length_scale(l_RANS, l_LES, f_d)
    end

    # Solve base RANS model with modified wall distance
    # For SA: create a new instance with the DDES length scale (no mutation)
    if model.base_model isa SpalartAllmaras
        modified_model = SpalartAllmaras{T}(
            model.base_model.cb1, model.base_model.cb2, model.base_model.sigma,
            model.base_model.kappa, model.base_model.cw2, model.base_model.cw3,
            model.base_model.cv1, model.base_model.ct3, model.base_model.ct4,
            l_ddes,
        )
        solve_turbulence!(
            turb_state, modified_model, U, phi, nu, mesh, bcs_turb;
            dt = dt, linear_solver = linear_solver
        )
    else
        # Generic fallback: just solve the base model
        solve_turbulence!(
            turb_state, model.base_model, U, phi, nu, mesh, bcs_turb;
            dt = dt, linear_solver = linear_solver
        )
    end

    return nothing
end

# ── v3.0 Wave 1: WMLES and dedicated SA-DDES / IDDES variants ──────
# Loaded via this trailing include chain so the parent layer file
# (`src/layers/discretization_assembly_kernels.jl`) does not need to be
# updated in the same commit. The wave's main-thread pass will lift
# these into the layer file once the rest of Wave 1 lands.

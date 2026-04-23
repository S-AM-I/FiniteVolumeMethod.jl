# turbulence/iddes.jl — Improved Delayed Detached Eddy Simulation (stub)
#
# STATUS (v3.0 Wave 1 / 2026-04-23): TIME-BOXED STUB.
#
# Full IDDES (Shur et al. 2008) replaces the DDES shielding function with a
# more elaborate blend of three length scales (RANS, LES, and a
# wall-modeled LES branch activated by a resolved-content sensor) and
# uses a modified `f_d` that does not simply shut off at the wall. A
# faithful implementation requires:
#
#   - velocity-divergence sensor `α = 1 − max(0, (1 − f_d) − f_restore)`
#   - wall-modeled LES branch with `f_B = min(2 exp(−9·α²), 1)`
#   - length-scale blend `l_IDDES = f_hyb·(1 + f_restore)·l_RANS + (1 − f_hyb)·l_LES`
#
# That level of detail needs substantive validation against LES reference
# data to ensure the sensor does not flip prematurely — publication-grade
# work, not a day of integration. We therefore ship this module as a
# stub that emits a single `@warn` on construction and falls back to
# `SADDES` for the solve. The public API is in place so downstream code
# (e.g. the capability matrix, the incompressible solver wrapper) can
# reference `IDDES` today and get the SA-DDES behaviour until v3.1.

"""
    IDDES{T} <: AbstractHybridModel

Improved Delayed Detached Eddy Simulation. **Current implementation
status: stub.** On construction emits a one-time `@warn` and falls
back to [`SADDES`](@ref) for all transport-equation assembly. The
struct carries the fully-populated SA-DDES state so that the full
IDDES blending logic can be layered on in v3.1 without an API break.

# Fields
- `saddes::SADDES{T}` — backing SA-DDES model used for the fallback
  solve. All heavy state (base SA coefficients, filter width, wall
  distance) lives here.
- `f_restore::T` — reserved for the IDDES restoration function
  (unused by the stub; default 0).

# Construction

```julia
iddes = IDDES(sa, mesh, [:wall]; C_DES = 0.65)
```
"""
struct IDDES{T} <: AbstractHybridModel
    saddes::SADDES{T}
    f_restore::T
end

"""
    IDDES(base, mesh, wall_patches; C_DES = 0.65)

Construct an IDDES stub. Emits a deprecation-style warning announcing
the fallback to SA-DDES; the warning is marked `maxlog = 1` so it
fires only once per Julia session.
"""
function IDDES(
        base::SpalartAllmaras{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol};
        C_DES::Real = 0.65,
    ) where {Dim, T}
    @warn(
        "IDDES: using SA-DDES fallback; full IDDES deferred to v3.1",
        maxlog = 1,
    )
    saddes = SADDES(base, mesh, wall_patches; C_DES = C_DES)
    return IDDES{T}(saddes, zero(T))
end

# ── Interface implementation — all calls delegate to SA-DDES ─────────

n_turbulence_fields(model::IDDES) = n_turbulence_fields(model.saddes)
turbulence_field_names(model::IDDES) = turbulence_field_names(model.saddes)

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::IDDES{T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    turbulent_viscosity!(nu_t, model.saddes, turb_state, mesh)
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::IDDES{T},
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    solve_turbulence!(
        turb_state, model.saddes, U, phi, nu, mesh, bcs_turb;
        dt = dt, linear_solver = linear_solver, solver_config = solver_config,
    )
    return nothing
end

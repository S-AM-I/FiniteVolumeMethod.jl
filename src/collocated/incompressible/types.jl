# incompressible/types.jl — Core types for incompressible Navier-Stokes solver
#
# Defines the pressure-velocity coupling algorithms (SIMPLE, PISO, PIMPLE),
# the incompressible problem definition, mutable solver state, and result
# types.  These are the foundational types used by all other files in
# `src/incompressible/`.


# ── Abstract coupling hierarchy ─────────────────────────────────────

@doc """
    AbstractPVCoupling

Supertype for pressure-velocity coupling algorithms used in the
incompressible Navier-Stokes solver.  Concrete subtypes include
[`SIMPLE`](@ref), [`PISO`](@ref), and [`PIMPLE`](@ref).
"""
abstract type AbstractPVCoupling end

# ── SIMPLE ──────────────────────────────────────────────────────────

@doc """
    SIMPLE{T} <: AbstractPVCoupling

Semi-Implicit Method for Pressure-Linked Equations.

Outer iteration with under-relaxation of velocity and pressure.
Suitable for steady-state problems.

# Fields
- `alpha_U::T` — velocity under-relaxation factor (0 < alpha_U <= 1)
- `alpha_p::T` — pressure under-relaxation factor (0 < alpha_p <= 1)
- `max_iterations::Int` — maximum number of outer iterations
- `tolerance::T` — convergence tolerance on residual norms
"""
struct SIMPLE{T} <: AbstractPVCoupling
    alpha_U::T
    alpha_p::T
    max_iterations::Int
    tolerance::T
end

@doc """
    SIMPLE(; alpha_U = 0.7, alpha_p = 0.3, max_iterations = 1000, tolerance = 1e-6)

Construct a [`SIMPLE`](@ref) algorithm with default under-relaxation parameters.
"""
function SIMPLE(;
        alpha_U::T = 0.7,
        alpha_p::T = 0.3,
        max_iterations::Int = 1000,
        tolerance::T = 1.0e-6,
    ) where {T}
    return SIMPLE{T}(alpha_U, alpha_p, max_iterations, tolerance)
end

# ── PISO ────────────────────────────────────────────────────────────

@doc """
    PISO{T} <: AbstractPVCoupling

Pressure Implicit with Splitting of Operators.

Non-iterative algorithm with multiple pressure correction steps per
time step.  Suitable for transient problems.

# Fields
- `n_correctors::Int` — number of pressure correction steps (typically 2)
"""
struct PISO{T} <: AbstractPVCoupling
    n_correctors::Int
end

@doc """
    PISO(; n_correctors = 2)
    PISO{T}(; n_correctors = 2)

Construct a [`PISO`](@ref) algorithm with the given number of corrector steps.
The unparameterized form defaults to `Float64`; use `PISO{T}(...)` for other
floating-point types (matching `SIMPLE`'s type flexibility).
"""
function PISO(; n_correctors::Int = 2)
    return PISO{Float64}(n_correctors)
end

function PISO{T}(; n_correctors::Int = 2) where {T}
    return PISO{T}(n_correctors)
end

# ── PIMPLE ──────────────────────────────────────────────────────────

@doc """
    PIMPLE{T} <: AbstractPVCoupling

Merged PISO-SIMPLE algorithm.  Combines outer SIMPLE iterations with
inner PISO correctors.  Works for both steady and transient problems.

# Fields
- `n_outer::Int` — number of outer (SIMPLE-like) iterations
- `n_correctors::Int` — number of inner (PISO-like) pressure corrections
- `alpha_U::T` — velocity under-relaxation factor
- `alpha_p::T` — pressure under-relaxation factor
- `tolerance::T` — convergence tolerance for outer loop
"""
struct PIMPLE{T} <: AbstractPVCoupling
    n_outer::Int
    n_correctors::Int
    alpha_U::T
    alpha_p::T
    tolerance::T
end

@doc """
    PIMPLE(; n_outer = 2, n_correctors = 1, alpha_U = 0.7, alpha_p = 0.3, tolerance = 1e-6)

Construct a [`PIMPLE`](@ref) algorithm with default parameters.
"""
function PIMPLE(;
        n_outer::Int = 2,
        n_correctors::Int = 1,
        alpha_U::T = 0.7,
        alpha_p::T = 0.3,
        tolerance::T = 1.0e-6,
    ) where {T}
    return PIMPLE{T}(n_outer, n_correctors, alpha_U, alpha_p, tolerance)
end

# ── Incompressible problem ──────────────────────────────────────────

@doc """
    IncompressibleProblem{Dim, T, Mesh, BC, Algo <: AbstractPVCoupling, Model}

Complete specification of an incompressible Navier-Stokes problem on an
unstructured mesh.

# Fields
- `mesh::Mesh` — unstructured FVM mesh (typically `UnstructuredFVMMesh{Dim, T}`)
- `bcs::BC` — boundary conditions dictionary `Dict{Symbol, AbstractBoundaryCondition}`
- `algorithm::Algo` — pressure-velocity coupling algorithm
- `nu::T` — kinematic viscosity
- `density::T` — fluid density (constant, incompressible)
- `model::Model` — [`IncompressibleModel`](@ref) selecting the additional
  physics (turbulence, thermal, radiation, combustion, zones)
"""
struct IncompressibleProblem{Dim, T, Mesh, BC, Algo <: AbstractPVCoupling, Model}
    mesh::Mesh
    bcs::BC
    algorithm::Algo
    nu::T
    density::T
    model::Model
end

@doc """
    IncompressibleProblem(mesh, bcs, algorithm; nu, density = 1.0, model = IncompressibleModel())

Construct an [`IncompressibleProblem`](@ref) from a mesh, boundary conditions,
and coupling algorithm.

# Arguments
- `mesh` — `UnstructuredFVMMesh{Dim, T}`
- `bcs` — `Dict{Symbol, <:AbstractBoundaryCondition}` keyed by patch name
- `algorithm` — [`SIMPLE`](@ref), [`PISO`](@ref), or [`PIMPLE`](@ref)
- `nu` — kinematic viscosity
- `density` — fluid density (default `1.0`)
- `model` — [`IncompressibleModel`](@ref) (default: plain incompressible flow)
"""
function IncompressibleProblem(
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs,
        algorithm::Algo;
        nu::T,
        density::T = one(T),
        model = IncompressibleModel(),
    ) where {Dim, T, Algo <: AbstractPVCoupling}
    return IncompressibleProblem{
        Dim, T, typeof(mesh), typeof(bcs), Algo, typeof(model),
    }(
        mesh, bcs, algorithm, nu, density, model,
    )
end

# Problem-level forwarding of the model traits, so assembly hooks can ask the
# problem directly instead of reaching through `prob.model` each time.
has_turbulence(prob::IncompressibleProblem) = has_turbulence(prob.model)
has_thermal(prob::IncompressibleProblem) = has_thermal(prob.model)
has_radiation(prob::IncompressibleProblem) = has_radiation(prob.model)
has_combustion(prob::IncompressibleProblem) = has_combustion(prob.model)
has_porous_zones(prob::IncompressibleProblem) = has_porous_zones(prob.model)
has_mrf_zones(prob::IncompressibleProblem) = has_mrf_zones(prob.model)
is_plain_flow(prob::IncompressibleProblem) = is_plain_flow(prob.model)

# ── Solver state ────────────────────────────────────────────────────

@doc """
    IncompressibleState{Dim, T}

Mutable solver state carrying all field data needed by the
pressure-velocity coupling loop.

# Fields
- `U::CollocatedVectorField{Dim, T}` — cell-centered velocity
- `p::CollocatedScalarField{T}` — cell-centered pressure
- `phi::FaceFluxField{T}` — volumetric face flux
- `A_P::Vector{T}` — diagonal momentum coefficients (per cell)
- `H_U::Vector{SVector{Dim, T}}` — momentum H-operator values (per cell)
- `U_old::Vector{SVector{Dim, T}}` — old-time-level velocity (per cell),
  snapshot at the start of each transient time step via
  [`_snapshot_old_time!`](@ref).  The `ddt` term in `assemble_momentum!`
  is assembled against this field so that repeated assemblies within a
  time step (PISO correctors, PIMPLE outer iterations) all discretize
  `(Uⁿ⁺¹ - Uⁿ)/Δt` rather than drifting toward the previous iterate.
"""
# The first three fields are abstractly typed: the concrete types carry a
# trailing container parameter (`CollocatedVectorField{Dim,T,A}` and friends)
# that is not fixed here. Adding it would cascade — `SolveResult` declares
# `state::IncompressibleState{Dim,T}` and `snapshots::Vector{...}`, which would
# become UnionAll fields in turn — and Stage 5f replaces this storage with a
# flat vector of views, changing these declarations anyway. Left alone
# deliberately so the types are reworked once rather than twice.
mutable struct IncompressibleState{Dim, T}
    U::CollocatedVectorField{Dim, T}
    p::CollocatedScalarField{T}
    phi::FaceFluxField{T}
    A_P::Vector{T}
    H_U::Vector{SVector{Dim, T}}
    U_old::Vector{SVector{Dim, T}}
end

@doc """
    IncompressibleState{Dim, T}(U, p, phi, A_P, H_U)

Backward-compatible 5-argument constructor: initializes `U_old` as a copy
of `U.internal`.
"""
function IncompressibleState{Dim, T}(
        U::CollocatedVectorField{Dim, T},
        p::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        A_P::Vector{T},
        H_U::Vector{SVector{Dim, T}},
    ) where {Dim, T}
    return IncompressibleState{Dim, T}(U, p, phi, A_P, H_U, copy(U.internal))
end

@doc """
    _snapshot_old_time!(state::IncompressibleState)

Copy the current velocity into `state.U_old`.  Must be called exactly once
at the start of each transient time step, before the first momentum
assembly of that step.
"""
function _snapshot_old_time!(state::IncompressibleState{Dim, T}) where {Dim, T}
    copyto!(state.U_old, state.U.internal)
    return nothing
end

@doc """
    IncompressibleState(mesh::UnstructuredFVMMesh{Dim, T})

Construct a zero-initialized [`IncompressibleState`](@ref) for the given mesh.
`A_P` is initialized to ones to avoid division-by-zero before the first
momentum solve.
"""
function IncompressibleState(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}
    nc = length(mesh.cell_volumes)
    U = CollocatedVectorField(:U, mesh)
    p = CollocatedScalarField(:p, mesh)
    phi = FaceFluxField(:phi, mesh)
    A_P = ones(T, nc)
    H_U = fill(zero(SVector{Dim, T}), nc)
    return IncompressibleState{Dim, T}(U, p, phi, A_P, H_U)
end

# ── Solve result ────────────────────────────────────────────────────

@doc """
    SolveResult{Dim, T}

Output of an incompressible solver run, containing convergence info
and the final state.

# Fields
- `converged::Bool` — whether the solver met the tolerance criterion
  (for transient solvers: whether the run completed with finite residuals)
- `iterations::Int` — number of outer iterations performed
- `residuals::Dict{Symbol, Vector{T}}` — residual history per equation
- `state::IncompressibleState{Dim, T}` — final solver state
- `snapshots::Vector{IncompressibleState{Dim, T}}` — saved state snapshots
  from transient solvers (every `save_every` steps); empty for steady solvers
"""
struct SolveResult{Dim, T}
    converged::Bool
    iterations::Int
    residuals::Dict{Symbol, Vector{T}}
    state::IncompressibleState{Dim, T}
    snapshots::Vector{IncompressibleState{Dim, T}}
end

@doc """
    SolveResult{Dim, T}(converged, iterations, residuals, state)

Backward-compatible 4-argument constructor with no snapshots.
"""
function SolveResult{Dim, T}(
        converged::Bool,
        iterations::Int,
        residuals::Dict{Symbol, Vector{T}},
        state::IncompressibleState{Dim, T},
    ) where {Dim, T}
    return SolveResult{Dim, T}(
        converged, iterations, residuals, state,
        IncompressibleState{Dim, T}[],
    )
end

# ── Component extract / set helpers ─────────────────────────────────

@doc """
    _extract_component(U::CollocatedVectorField{Dim, T}, d::Int) -> Vector{T}

Extract the `d`-th spatial component of the vector field `U` as a
plain `Vector{T}` over interior cells.
"""
function _extract_component(
        U::CollocatedVectorField{Dim, T}, d::Int,
    ) where {Dim, T}
    return T[u[d] for u in U.internal]
end

@doc """
    _set_component!(U::CollocatedVectorField{Dim, T}, d::Int, vals::Vector{T})

Overwrite the `d`-th spatial component of `U.internal` with `vals`.
"""
function _set_component!(
        U::CollocatedVectorField{Dim, T}, d::Int, vals::Vector{T},
    ) where {Dim, T}
    for c in eachindex(U.internal)
        old = U.internal[c]
        U.internal[c] = Base.setindex(old, vals[c], d)
    end
    return nothing
end

# ============================================================
# Stiff Source Terms for IMEX Time Integration
# ============================================================
#
# Abstract interface and concrete implementations of stiff
# source terms that will be treated implicitly in the IMEX
# time integration loop.

"""
    AbstractStiffSource

Abstract supertype for stiff source terms treated implicitly
in IMEX time integration.

All subtypes must implement:
- `evaluate_stiff_source(src, law, w, u) -> SVector`
- `stiff_source_jacobian(src, law, w, u) -> SMatrix`

where `w` is the primitive state and `u` is the conserved state.
"""
abstract type AbstractStiffSource end

"""
    evaluate_stiff_source(src::AbstractStiffSource, law, w, u) -> SVector

Evaluate the stiff source term `S(U)` at the given state.

# Arguments
- `src`: The stiff source term.
- `law`: The conservation law.
- `w`: Primitive variable vector.
- `u`: Conserved variable vector.

# Returns
Source vector in conserved variable space (same length as `u`).
"""
function evaluate_stiff_source end

"""
    stiff_source_jacobian(src::AbstractStiffSource, law, w, u) -> SMatrix

Compute the Jacobian `∂S/∂U` of the stiff source term with respect
to the conserved variables.

# Arguments
- `src`: The stiff source term.
- `law`: The conservation law.
- `w`: Primitive variable vector.
- `u`: Conserved variable vector.

# Returns
Jacobian matrix `∂S/∂U` as an SMatrix.
"""
function stiff_source_jacobian end

# ============================================================
# ResistiveSource: Ohmic dissipation for MHD
# ============================================================

"""
    ResistiveSource{FT} <: AbstractStiffSource

Resistive (Ohmic) source term for ideal MHD equations.

In the resistive MHD framework, the magnetic field evolves as:
  `∂B/∂t = ∇×(v×B) + η ∇²B`

The diffusive part `η ∇²B` is stiff when η is large. In a
spatially discretised form, this gives a source:
  `S = η / dx² * (B_{i-1} - 2B_i + B_{i+1})`

For the IMEX framework, we treat the local contribution as:
  `S_i = -2η / dx² * B_i` (implicit part, stiff)
  and the neighbour contributions explicitly.

# Fields
- `eta::FT`: Magnetic diffusivity (resistivity / μ₀).
"""
struct ResistiveSource{FT} <: AbstractStiffSource
    eta::FT
end

@inline function evaluate_stiff_source(src::ResistiveSource, law::IdealMHDEquations, w::SVector{8}, u::SVector{8})
    # The implicit part is the diagonal (local) contribution of the
    # Laplacian. The stiff source acts only on the magnetic field
    # components (indices 6, 7, 8 in the conserved vector).
    # S = -2η/dx² * [0, 0, 0, 0, 0, Bx, By, Bz]
    # dx is not known here; the IMEX solver will scale by dx².
    # We return the per-unit source: S/η = -[0, 0, 0, 0, 0, Bx, By, Bz] * 2/dx²
    # Actually, for the local-in-cell formulation, we return:
    #   S = -η_eff * u_B  where η_eff encodes the spatial discretisation.
    # For simplicity in the IMEX interface, we return the source evaluated
    # with unit spatial factor. The IMEX solve loop multiplies by 1/dx².
    η = src.eta
    Bx = u[6]
    By = u[7]
    Bz = u[8]
    return SVector(
        zero(η), zero(η), zero(η), zero(η), zero(η),
        -η * Bx, -η * By, -η * Bz
    )
end

@inline function stiff_source_jacobian(src::ResistiveSource, law::IdealMHDEquations, w::SVector{8}, u::SVector{8})
    η = src.eta
    z = zero(η)
    # Jacobian ∂S/∂U: only the B components have non-zero derivatives
    # ∂S_6/∂U_6 = -η, ∂S_7/∂U_7 = -η, ∂S_8/∂U_8 = -η
    return StaticArrays.SMatrix{8, 8}(
        z, z, z, z, z, z, z, z,    # column 1 (∂S/∂ρ)
        z, z, z, z, z, z, z, z,    # column 2 (∂S/∂ρvx)
        z, z, z, z, z, z, z, z,    # column 3 (∂S/∂ρvy)
        z, z, z, z, z, z, z, z,    # column 4 (∂S/∂ρvz)
        z, z, z, z, z, z, z, z,    # column 5 (∂S/∂E)
        z, z, z, z, z, -η, z, z,   # column 6 (∂S/∂Bx)
        z, z, z, z, z, z, -η, z,   # column 7 (∂S/∂By)
        z, z, z, z, z, z, z, -η    # column 8 (∂S/∂Bz)
    )
end

# ============================================================
# CoolingSource: optically thin radiative cooling
# ============================================================

"""
    CoolingSource{FT, F} <: AbstractStiffSource

Optically thin radiative cooling source term for the energy equation.

The cooling rate depends on local density and temperature:
  `S_E = -ρ² Λ(T)`

where `Λ(T)` is the cooling function.

# Fields
- `cooling_function::F`: Function `T -> Λ(T)` returning the cooling rate coefficient.
- `mu_mol::FT`: Mean molecular weight (for temperature computation).
"""
struct CoolingSource{FT, F} <: AbstractStiffSource
    cooling_function::F
    mu_mol::FT
end

CoolingSource(cooling_fn; mu_mol = 0.6) = CoolingSource(cooling_fn, mu_mol)

@inline function evaluate_stiff_source(src::CoolingSource, law::EulerEquations{1}, w::SVector{3}, u::SVector{3})
    ρ, v, P = w
    # Temperature from ideal gas: T = P * mu / (ρ * k_B)
    # For dimensionless problems, T ~ P/ρ
    T = P / ρ * src.mu_mol
    Λ = src.cooling_function(T)
    S_E = -ρ^2 * Λ
    return SVector(zero(ρ), zero(ρ), S_E)
end

@inline function evaluate_stiff_source(src::CoolingSource, law::EulerEquations{2}, w::SVector{4}, u::SVector{4})
    ρ, vx, vy, P = w
    T = P / ρ * src.mu_mol
    Λ = src.cooling_function(T)
    S_E = -ρ^2 * Λ
    return SVector(zero(ρ), zero(ρ), zero(ρ), S_E)
end

@inline function stiff_source_jacobian(src::CoolingSource, law::EulerEquations{1}, w::SVector{3}, u::SVector{3})
    # Approximate Jacobian: linearise S_E around current state
    # S_E = -ρ² Λ(T), T = P/(ρ) * μ
    # ∂S_E/∂E ≈ S_E / (E) (simple diagonal approximation)
    ρ, v, P = w
    T = P / ρ * src.mu_mol
    Λ = src.cooling_function(T)
    E = u[3]
    z = zero(ρ)
    dSdE = abs(E) > eps(typeof(E)) ? -ρ^2 * Λ / E : z
    return StaticArrays.SMatrix{3, 3}(
        z, z, z,       # column 1
        z, z, z,       # column 2
        z, z, dSdE     # column 3
    )
end

@inline function stiff_source_jacobian(src::CoolingSource, law::EulerEquations{2}, w::SVector{4}, u::SVector{4})
    ρ, vx, vy, P = w
    T = P / ρ * src.mu_mol
    Λ = src.cooling_function(T)
    E = u[4]
    z = zero(ρ)
    dSdE = abs(E) > eps(typeof(E)) ? -ρ^2 * Λ / E : z
    return StaticArrays.SMatrix{4, 4}(
        z, z, z, z,       # column 1
        z, z, z, z,       # column 2
        z, z, z, z,       # column 3
        z, z, z, dSdE     # column 4
    )
end

# ============================================================
# NullSource: no stiff source (for testing / pure explicit)
# ============================================================

"""
    NullSource <: AbstractStiffSource

A no-op stiff source that returns zero. Useful for testing the IMEX
framework in purely explicit mode.
"""
struct NullSource <: AbstractStiffSource end

@inline function evaluate_stiff_source(::NullSource, law, w::SVector{N}, u::SVector{N}) where {N}
    return zero(u)
end

@inline function stiff_source_jacobian(::NullSource, law, w::SVector{N}, u::SVector{N}) where {N}
    FT = eltype(u)
    return zero(StaticArrays.SMatrix{N, N, FT})
end

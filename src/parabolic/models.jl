# Model type definitions for Parabolic FVM Solver
# Migrated from Simu.jl SimuFVM/models.jl

# --- Physics Model Abstract Types ---

"""Abstract supertype for all parabolic equation models."""
abstract type AbstractEquationModel end

"""Abstract supertype for diffusion equation models."""
abstract type AbstractDiffusion <: AbstractEquationModel end

"""Abstract supertype for advection equation models."""
abstract type AbstractAdvection <: AbstractEquationModel end

"""Abstract supertype for combined advection-diffusion equation models."""
abstract type AbstractAdvectionDiffusion <: AbstractEquationModel end

# --- Diffusion Models ---

"""Constant-coefficient diffusion equation model in 1D Cartesian coordinates."""
struct Diffusion1D <: AbstractDiffusion
    gamma::Float64 # Diffusion coefficient
    scheme::Symbol # :first_order or :second_order

    Diffusion1D(gamma::Float64) = new(gamma, :first_order)
    Diffusion1D(gamma::Float64, scheme::Symbol) = new(gamma, scheme)
end

"""Constant-coefficient diffusion equation model in 2D Cartesian coordinates."""
struct Diffusion2D <: AbstractDiffusion
    gamma::Float64 # Diffusion coefficient
    scheme::Symbol # :first_order or :second_order

    Diffusion2D(gamma::Float64) = new(gamma, :first_order)
    Diffusion2D(gamma::Float64, scheme::Symbol) = new(gamma, scheme)
end

"""Constant-coefficient diffusion equation model in 3D Cartesian coordinates."""
struct Diffusion3D <: AbstractDiffusion
    gamma::Float64 # Diffusion coefficient
    scheme::Symbol # :first_order or :second_order

    Diffusion3D(gamma::Float64) = new(gamma, :first_order)
    Diffusion3D(gamma::Float64, scheme::Symbol) = new(gamma, scheme)
end

# --- Variable Diffusion Models ---

"""Variable-coefficient diffusion equation model in 1D Cartesian coordinates."""
struct VariableDiffusion1D <: AbstractDiffusion
    gamma::Union{Function, Vector{Float64}} # Diffusion coefficient (function or array)
    scheme::Symbol

    VariableDiffusion1D(gamma) = new(gamma, :first_order)
    VariableDiffusion1D(gamma, scheme::Symbol) = new(gamma, scheme)
end

"""Variable-coefficient diffusion equation model in 2D Cartesian coordinates."""
struct VariableDiffusion2D <: AbstractDiffusion
    gamma::Union{Function, Matrix{Float64}} # Diffusion coefficient (function or array)
end

"""Variable-coefficient diffusion equation model in 3D Cartesian coordinates."""
struct VariableDiffusion3D <: AbstractDiffusion
    gamma::Union{Function, Array{Float64, 3}} # Diffusion coefficient (function or 3D array)
end

# --- Anisotropic Diffusion Models ---

"""Anisotropic (tensor) diffusion equation model in 1D Cartesian coordinates."""
struct AnisotropicDiffusion1D <: AbstractDiffusion
    D::Float64  # 1x1 diffusion tensor (scalar for 1D)
end

"""Anisotropic (tensor) diffusion equation model in 2D Cartesian coordinates."""
struct AnisotropicDiffusion2D <: AbstractDiffusion
    D::Matrix{Float64}  # 2x2 diffusion tensor
end

"""Anisotropic (tensor) diffusion equation model in 3D Cartesian coordinates."""
struct AnisotropicDiffusion3D <: AbstractDiffusion
    D::Union{Array{Float64, 2}, Array{Float64, 5}}  # 3x3 diffusion tensor (3x3 matrix for constant, or 3x3xNxMxK array for spatially varying)
end

# --- Cylindrical Diffusion Models ---

"""Constant-coefficient diffusion equation model in 1D cylindrical (radial) coordinates."""
struct CylindricalDiffusion1D <: AbstractDiffusion
    gamma::Float64 # Diffusion coefficient
    scheme::Symbol # :first_order or :second_order

    CylindricalDiffusion1D(gamma::Float64) = new(gamma, :first_order)
    CylindricalDiffusion1D(gamma::Float64, scheme::Symbol) = new(gamma, scheme)
end

"""Constant-coefficient diffusion equation model in 2D cylindrical (r-z) coordinates."""
struct CylindricalDiffusion2D <: AbstractDiffusion
    gamma::Float64 # Diffusion coefficient
    scheme::Symbol # :first_order or :second_order

    CylindricalDiffusion2D(gamma::Float64) = new(gamma, :first_order)
    CylindricalDiffusion2D(gamma::Float64, scheme::Symbol) = new(gamma, scheme)
end

"""Variable-coefficient diffusion equation model in 1D cylindrical (radial)
coordinates. `gamma` is either a function `gamma(r)` evaluated at the cell
centre or a per-cell `Vector{Float64}`. Face coefficients use the harmonic
mean of the two adjacent cell values, matching the Cartesian
`VariableDiffusion1D` discretisation."""
struct VariableCylindricalDiffusion1D <: AbstractDiffusion
    gamma::Union{Function, Vector{Float64}}
    scheme::Symbol

    VariableCylindricalDiffusion1D(gamma) = new(gamma, :first_order)
    VariableCylindricalDiffusion1D(gamma, scheme::Symbol) = new(gamma, scheme)
end

"""Variable-coefficient diffusion equation model in 2D cylindrical (r-z)
coordinates. `gamma` is either a function `gamma(r, z)` evaluated at the
cell centre or a `Matrix{Float64}` of size `(n_cells_r, n_cells_z)`. Face
coefficients use the harmonic mean of the two adjacent cell values."""
struct VariableCylindricalDiffusion2D <: AbstractDiffusion
    gamma::Union{Function, Matrix{Float64}}
    scheme::Symbol

    VariableCylindricalDiffusion2D(gamma) = new(gamma, :first_order)
    VariableCylindricalDiffusion2D(gamma, scheme::Symbol) = new(gamma, scheme)
end

# --- Spherical Diffusion Models ---

"""Constant-coefficient diffusion equation model in 1D spherical (radial) coordinates."""
struct SphericalDiffusion1D <: AbstractDiffusion
    gamma::Float64 # Diffusion coefficient
    scheme::Symbol # :first_order or :second_order

    SphericalDiffusion1D(gamma::Float64) = new(gamma, :first_order)
    SphericalDiffusion1D(gamma::Float64, scheme::Symbol) = new(gamma, scheme)
end

# --- Spherical Advection Models ---

"""Advection equation model in 1D spherical (radial) coordinates."""
struct SphericalAdvection1D <: AbstractAdvection
    v::Float64 # Radial velocity
    scheme::Symbol # :upwind (default)

    SphericalAdvection1D(v::Float64) = new(v, :upwind)
    SphericalAdvection1D(v::Float64, scheme::Symbol) = new(v, scheme)
end

"""Combined advection-diffusion equation model in 1D spherical (radial) coordinates."""
struct SphericalAdvectionDiffusion1D <: AbstractAdvectionDiffusion
    advection::SphericalAdvection1D
    diffusion::SphericalDiffusion1D
end

# --- Cylindrical Advection Models ---

"""Advection equation model in 1D cylindrical (radial) coordinates."""
struct CylindricalAdvection1D <: AbstractAdvection
    v::Float64 # Radial advection velocity
    scheme::Symbol # :upwind (default)

    CylindricalAdvection1D(v::Float64) = new(v, :upwind)
    CylindricalAdvection1D(v::Float64, scheme::Symbol) = new(v, scheme)
end

"""Advection equation model in 2D cylindrical (r-z) coordinates."""
struct CylindricalAdvection2D <: AbstractAdvection
    vr::Float64 # Radial velocity
    vz::Float64 # Axial velocity
    scheme::Symbol # :upwind (default)

    CylindricalAdvection2D(vr::Float64, vz::Float64) = new(vr, vz, :upwind)
    CylindricalAdvection2D(vr::Float64, vz::Float64, scheme::Symbol) = new(vr, vz, scheme)
end

# --- Advection Models ---

"""Constant-velocity advection equation model in 1D Cartesian coordinates."""
struct Advection1D <: AbstractAdvection
    v::Float64 # Advection velocity (positive = rightward)
    scheme::Symbol # :upwind, :central, :muscl, or :quick (default: :upwind)

    Advection1D(v::Float64) = new(v, :upwind)
    Advection1D(v::Float64, scheme::Symbol) = new(v, scheme)
end

"""Constant-velocity advection equation model in 2D Cartesian coordinates."""
struct Advection2D <: AbstractAdvection
    vx::Float64 # x-component of the advection velocity
    vy::Float64 # y-component of the advection velocity
    scheme::Symbol # :upwind, :central, :muscl, or :quick (default: :upwind)

    Advection2D(vx::Float64, vy::Float64) = new(vx, vy, :upwind)
    Advection2D(vx::Float64, vy::Float64, scheme::Symbol) = new(vx, vy, scheme)
end

"""Constant-velocity advection equation model in 3D Cartesian coordinates."""
struct Advection3D <: AbstractAdvection
    vx::Float64 # x-component of the advection velocity
    vy::Float64 # y-component of the advection velocity
    vz::Float64 # z-component of the advection velocity
    scheme::Symbol # :upwind, :central, :muscl, or :quick (default: :upwind)

    Advection3D(vx::Float64, vy::Float64, vz::Float64) = new(vx, vy, vz, :upwind)
    Advection3D(vx::Float64, vy::Float64, vz::Float64, scheme::Symbol) = new(vx, vy, vz, scheme)
end

# --- Variable Advection Models ---

"""Variable-velocity advection equation model in 1D Cartesian coordinates."""
struct VariableAdvection1D <: AbstractAdvection
    v::Union{Function, Vector{Float64}} # Advection velocity (function or array)
    scheme::Symbol

    VariableAdvection1D(v) = new(v, :upwind)
    VariableAdvection1D(v, scheme::Symbol) = new(v, scheme)
end

"""Variable-velocity advection equation model in 2D Cartesian coordinates."""
struct VariableAdvection2D <: AbstractAdvection
    vx::Union{Function, Matrix{Float64}} # x-component velocity (function or array)
    vy::Union{Function, Matrix{Float64}} # y-component velocity (function or array)
end

"""Variable-velocity advection equation model in 3D Cartesian coordinates."""
struct VariableAdvection3D <: AbstractAdvection
    vx::Union{Function, Array{Float64, 3}} # x-component velocity (function or 3D array)
    vy::Union{Function, Array{Float64, 3}} # y-component velocity (function or 3D array)
    vz::Union{Function, Array{Float64, 3}} # z-component velocity (function or 3D array)
end

# --- Combined Models ---

"""Combined advection-diffusion equation model in 1D Cartesian coordinates."""
struct AdvectionDiffusion1D <: AbstractAdvectionDiffusion
    advection::Advection1D
    diffusion::Diffusion1D
end

"""Combined advection-diffusion equation model in 2D Cartesian coordinates."""
struct AdvectionDiffusion2D <: AbstractAdvectionDiffusion
    advection::Advection2D
    diffusion::Diffusion2D
end

# --- Variable Combined Models ---

"""Combined variable-coefficient advection-diffusion equation model in 1D Cartesian coordinates."""
struct VariableAdvectionDiffusion1D <: AbstractAdvectionDiffusion
    advection::VariableAdvection1D
    diffusion::VariableDiffusion1D
end

"""Combined variable-coefficient advection-diffusion equation model in 2D Cartesian coordinates."""
struct VariableAdvectionDiffusion2D <: AbstractAdvectionDiffusion
    advection::VariableAdvection2D
    diffusion::VariableDiffusion2D
end

"""Combined advection-diffusion equation model in 1D cylindrical (radial) coordinates."""
struct CylindricalAdvectionDiffusion1D <: AbstractAdvectionDiffusion
    advection::CylindricalAdvection1D
    diffusion::CylindricalDiffusion1D
end

"""Combined advection-diffusion equation model in 2D cylindrical (r-z) coordinates."""
struct CylindricalAdvectionDiffusion2D <: AbstractAdvectionDiffusion
    advection::CylindricalAdvection2D
    diffusion::CylindricalDiffusion2D
end

"""Combined advection-diffusion equation model in 3D Cartesian coordinates."""
struct AdvectionDiffusion3D <: AbstractAdvectionDiffusion
    advection::Advection3D
    diffusion::Diffusion3D
end

"""Combined variable-coefficient advection-diffusion equation model in 3D Cartesian coordinates."""
struct VariableAdvectionDiffusion3D <: AbstractAdvectionDiffusion
    advection::VariableAdvection3D
    diffusion::VariableDiffusion3D
end

# --- Source Terms ---

"""Abstract supertype for source term models."""
abstract type AbstractSourceTerm end

"""
    ConstantSource(value)

Uniform source term with constant value.
"""
struct ConstantSource <: AbstractSourceTerm
    value::Float64
end

"""
    SpatialSource(values)

Source term with cell-wise values.
For 1D: values is a Vector{Float64}
For 2D: values is a Matrix{Float64} or Vector{Float64} (flattened)
"""
struct SpatialSource{T} <: AbstractSourceTerm
    values::T
end

"""
    FunctionSource(f)

Source term defined by a function.
For 1D: f(x) -> Float64
For 2D: f(x, y) -> Float64
"""
struct FunctionSource <: AbstractSourceTerm
    f::Function
end

"""
    evaluate_source(source, mesh, i)

Evaluate source term at cell i for 1D mesh.
"""
function evaluate_source(source::ConstantSource, mesh::Mesh1D, i)
    return source.value
end

function evaluate_source(source::SpatialSource{Vector{Float64}}, mesh::Mesh1D, i)
    return source.values[i]
end

function evaluate_source(source::FunctionSource, mesh::Mesh1D, i)
    return source.f(mesh.cells[i].center)
end

"""
    evaluate_source(source, mesh, i, j)

Evaluate source term at cell (i,j) for 2D mesh.
"""
function evaluate_source(source::ConstantSource, mesh::Mesh2D, i, j)
    return source.value
end

function evaluate_source(source::SpatialSource{Matrix{Float64}}, mesh::Mesh2D, i, j)
    return source.values[i, j]
end

function evaluate_source(source::SpatialSource{Vector{Float64}}, mesh::Mesh2D, i, j)
    k = (i - 1) * mesh.ny + j
    return source.values[k]
end

function evaluate_source(source::FunctionSource, mesh::Mesh2D, i, j)
    cell = mesh.cells[(i - 1) * mesh.ny + j]
    return source.f(cell.center[1], cell.center[2])
end

"""
    evaluate_source(source, mesh, i, j)

Evaluate source term at cell (i,j) for CurvilinearMesh2D.
"""
function evaluate_source(source::ConstantSource, mesh::CurvilinearMesh2D, i, j)
    return source.value
end

function evaluate_source(source::FunctionSource, mesh::CurvilinearMesh2D, i, j)
    k = (i - 1) * mesh.ny + j
    cell = mesh.cells[k]
    return source.f(cell.center[1], cell.center[2])
end

function evaluate_source(source::SpatialSource{Vector{Float64}}, mesh::CurvilinearMesh2D, i, j)
    k = (i - 1) * mesh.ny + j
    return source.values[k]
end

"""
    evaluate_source(source, mesh, i, j, k)

Evaluate source term at cell (i,j,k) for 3D mesh.
"""
function evaluate_source(source::ConstantSource, mesh::Mesh3D, i, j, k)
    return source.value
end

function evaluate_source(source::SpatialSource{Array{Float64, 3}}, mesh::Mesh3D, i, j, k)
    return source.values[i, j, k]
end

function evaluate_source(source::SpatialSource{Vector{Float64}}, mesh::Mesh3D, i, j, k)
    idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
    return source.values[idx]
end

function evaluate_source(source::FunctionSource, mesh::Mesh3D, i, j, k)
    idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
    cell = mesh.cells[idx]
    return source.f(cell.center[1], cell.center[2], cell.center[3])
end

# --- Turbulence Models ---

"""Abstract supertype for turbulence closure models."""
abstract type AbstractTurbulenceModel <: AbstractEquationModel end

"""
    ParabolicKEpsilon

Standard k-ε turbulence model for the parabolic (cell-centered) solver.
Renamed from StandardKEpsilon to avoid conflict with the vertex-centered
turbulence model in `src/physics/turbulence/k_epsilon.jl`.

The same physical constants are used by both types. Convert between them via:
- `ParabolicKEpsilon(model::StandardKEpsilon)`
- `StandardKEpsilon(model::ParabolicKEpsilon)`
"""
struct ParabolicKEpsilon <: AbstractTurbulenceModel
    C_mu::Float64
    sigma_k::Float64
    sigma_epsilon::Float64
    C1_epsilon::Float64
    C2_epsilon::Float64
end

function ParabolicKEpsilon(;
        C_mu = 0.09, sigma_k = 1.0, sigma_epsilon = 1.3,
        C1_epsilon = 1.44, C2_epsilon = 1.92
    )
    return ParabolicKEpsilon(C_mu, sigma_k, sigma_epsilon, C1_epsilon, C2_epsilon)
end

# --- Linearized Source Term ---
"""
    LinearizedSource(sc, sp)

Linearized source term S = Sc + Sp * phi.
Sc: Constant part (added to RHS).
Sp: Linear part (subtracted from LHS diagonal). Sp should generally be negative for stability.
"""
struct LinearizedSource{T1, T2} <: AbstractSourceTerm
    sc::T1 # Vector or scalar
    sp::T2 # Vector or scalar
end

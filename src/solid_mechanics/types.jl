# solid_mechanics/types.jl — Linear elasticity on unstructured mesh (Stage 7a)
#
# Displacement-based FVM assembly of the Cauchy momentum equation for
# small-strain isotropic linear elasticity:
#
#   ∇ · σ + f = 0,  σ_ij = λ δ_ij (∇·u) + μ (∂_j u_i + ∂_i u_j)
#
# Each momentum component is assembled as a separate `CollocatedEquation`
# — the full block-coupled solve is a Stage 7b FSI-follow-up.
# The Laplacian of each u_i and the coupling terms are added explicitly
# using the existing collocated operators.
#
# This is the MVP cantilever-beam / plate-bending / Cook's-membrane
# benchmark infrastructure. Finite-strain + contact + plasticity are
# deferred (Stage 7a follow-ups).

using StaticArrays: SVector
using LinearAlgebra: dot

"""
    IsotropicElastic{T}(; E, nu)

Isotropic linear-elastic material with Young's modulus `E` (Pa) and
Poisson's ratio `nu`. Derives Lamé constants:

    λ = E · nu / ((1 + nu)(1 - 2 nu))     (first Lamé parameter)
    μ = E / (2 (1 + nu))                  (shear modulus)
"""
struct IsotropicElastic{T}
    E::T
    nu::T
    lambda::T
    mu::T
end
function IsotropicElastic(; E::Real = 1.0, nu::Real = 0.3)
    T = promote_type(typeof(float(E)), typeof(float(nu)))
    E_T = T(E); nu_T = T(nu)
    lambda = E_T * nu_T / ((one(T) + nu_T) * (one(T) - T(2) * nu_T))
    mu = E_T / (T(2) * (one(T) + nu_T))
    return IsotropicElastic{T}(E_T, nu_T, lambda, mu)
end

"""
    SolidDisplacementProblem{Dim, T, Mesh, Mat}

Displacement-formulation solid-mechanics problem.

# Fields
- `mesh::Mesh` — structural mesh (UnstructuredFVMMesh).
- `material::Mat` — elastic constitutive model.
- `body_force::SVector{Dim, T}` — per-cell body force density (e.g. gravity).
- `displacement_bcs::Dict{Symbol, SVector{Dim, T}}` — patch → Dirichlet displacement.
- `traction_bcs::Dict{Symbol, SVector{Dim, T}}` — patch → prescribed surface traction.
"""
struct SolidDisplacementProblem{Dim, T, Mesh, Mat}
    mesh::Mesh
    material::Mat
    body_force::SVector{Dim, T}
    displacement_bcs::Dict{Symbol, SVector{Dim, T}}
    traction_bcs::Dict{Symbol, SVector{Dim, T}}
end

function SolidDisplacementProblem(
        mesh::UnstructuredFVMMesh{Dim, T}, material::IsotropicElastic{T};
        body_force::SVector{Dim, T} = zero(SVector{Dim, T}),
        displacement_bcs = Dict{Symbol, SVector{Dim, T}}(),
        traction_bcs = Dict{Symbol, SVector{Dim, T}}(),
    ) where {Dim, T}
    return SolidDisplacementProblem{Dim, T, typeof(mesh), typeof(material)}(
        mesh, material, body_force, displacement_bcs, traction_bcs,
    )
end

"""
    stress_tensor(material, strain_tensor) -> SMatrix{Dim, Dim, T}

Compute Cauchy stress from small-strain tensor via isotropic elasticity:
`σ = λ tr(ε) I + 2μ ε`.
"""
function stress_tensor(
        mat::IsotropicElastic{T}, strain::AbstractMatrix{T},
    ) where {T}
    Dim = size(strain, 1)
    tr_eps = zero(T)
    for i in 1:Dim
        tr_eps += strain[i, i]
    end
    sigma = zeros(T, Dim, Dim)
    for j in 1:Dim, i in 1:Dim
        sigma[i, j] = T(2) * mat.mu * strain[i, j]
    end
    for i in 1:Dim
        sigma[i, i] += mat.lambda * tr_eps
    end
    return sigma
end

"""
    small_strain_tensor(grad_u::AbstractMatrix) -> AbstractMatrix

Symmetrize the displacement gradient: `ε = (∇u + ∇u^T) / 2`.
"""
function small_strain_tensor(grad_u::AbstractMatrix{T}) where {T}
    Dim = size(grad_u, 1)
    eps_tensor = similar(grad_u)
    for j in 1:Dim, i in 1:Dim
        eps_tensor[i, j] = T(0.5) * (grad_u[i, j] + grad_u[j, i])
    end
    return eps_tensor
end

"""
    cantilever_tip_deflection(E, I, L, P) -> T

Analytical Euler-Bernoulli tip deflection of a cantilever of length `L`,
flexural rigidity `E·I`, with point load `P` at the tip:
`δ = P L³ / (3 E I)`. Used as a reference value in Stage 7 benchmark
tests against the FVM solver.
"""
cantilever_tip_deflection(E, I, L, P) = P * L^3 / (3 * E * I)

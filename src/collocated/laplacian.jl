# collocated/laplacian.jl — Implicit Laplacian operator for collocated FVM
#
# Assembles `div(Γ * grad(φ))` into a sparse matrix equation suitable for
# LinearProblem from SciMLBase.  Supports non-orthogonal correction and
# both constant and spatially varying diffusivity.
#
# Follows OpenFOAM `fvm::laplacian(gamma, phi)` semantics.
#
# Non-orthogonal correction modes (Stage 3c, Jasak 1996 Ch. 4):
#   NON_ORTHO_NONE       — E = (S·d̂) d̂, T_f forced to zero (orthogonal only,
#                          no tangential correction term even if caller sets
#                          `non_ortho_correction = true`).
#   NON_ORTHO_MINIMUM    — E = (S·d̂) d̂; smallest |E|, gentlest implicit term.
#                          T_f = S_f − E_f (the parallel-to-face component).
#   NON_ORTHO_ORTHOGONAL — E = |S| d̂; fixed magnitude.
#   NON_ORTHO_OVER_RELAXED — E = |S|² / (S·d̂) · d̂; largest |E|, best
#                             convergence on skewed meshes (OpenFOAM default).
#
# The over-relaxed variant increases the implicit orthogonal coefficient
# by a factor 1/cosθ (θ = angle between S_f and d̂), which accelerates
# convergence of the iterative non-orthogonal correction on non-orthogonal
# meshes. All variants converge to the same continuous Laplacian.

"""
    NonOrthoCorrectionMode

Selector for the implicit-vs-explicit split of the face flux in
`assemble_laplacian!`. See file header for details.

Variants:

  * `NON_ORTHO_NONE` — orthogonal-only, skip tangential correction.
  * `NON_ORTHO_MINIMUM` — minimum-correction (Jasak 1996).
  * `NON_ORTHO_ORTHOGONAL` — classical orthogonal correction.
  * `NON_ORTHO_OVER_RELAXED` — over-relaxed (Jasak 1996, OpenFOAM default).
"""
@enum NonOrthoCorrectionMode begin
    NON_ORTHO_NONE
    NON_ORTHO_MINIMUM
    NON_ORTHO_ORTHOGONAL
    NON_ORTHO_OVER_RELAXED
end

# ── Core assembly ────────────────────────────────────────────────────

"""
    assemble_laplacian!(
        eq::CollocatedEquation{T},
        gamma::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition};
        non_ortho_correction::Bool = false,
        grad_phi::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
    )

Assemble the implicit Laplacian operator `div(Γ * grad(φ))` into `eq`.

For each internal face, the orthogonal contribution is:
```
Γ_f * |S_f| / |d|  * (φ_N - φ_P)
```
where `d = x_N - x_P` and `Γ_f` is the harmonic mean of cell diffusivities.

If `non_ortho_correction = true` and `grad_phi` is provided, an explicit
non-orthogonal correction is added to the RHS:
```
Γ_f * (∇φ)_f · (S_f - |S_f|²/|S_f·d̂| * d̂)
```

# Arguments
- `eq` — equation object (matrix and RHS modified in-place)
- `gamma` — diffusivity: scalar (constant) or `Vector{T}` (per-cell)
- `mesh` — `UnstructuredFVMMesh`
- `bcs` — boundary conditions keyed by patch name
- `non_ortho_correction` — enable explicit non-orthogonal correction
- `grad_phi` — current gradient field (required if `non_ortho_correction = true`)
"""
function assemble_laplacian!(
        eq::CollocatedEquation{T},
        gamma::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition};
        non_ortho_correction::Bool = false,
        grad_phi::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        correction_mode::NonOrthoCorrectionMode = NON_ORTHO_OVER_RELAXED,
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)

    @inbounds for f in 1:nf
        P = owner(mesh, f)
        S_f = face_normal_area(mesh, f)
        A_f = mesh.face_areas[f]

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            d_vec, d_mag = owner_neighbour_distance(mesh, f)

            # Diffusivity at face: harmonic mean for variable gamma
            gamma_f = _face_diffusivity(gamma, mesh, f)

            # Implicit / explicit split of S_f along d̂ (Jasak 1996).
            d_hat = d_vec / d_mag
            S_mag2 = dot(S_f, S_f)
            S_dot_d = dot(S_f, d_hat)

            # Magnitude of the implicit component |E| chosen by mode.
            E_mag = _non_ortho_E_magnitude(correction_mode, S_mag2, S_dot_d)
            flux_coeff = gamma_f * E_mag / d_mag

            # Implicit contribution: flux_coeff * (φ_N - φ_P)
            add_face_coeffs_PN!(
                eq, f, P, N,
                flux_coeff, -flux_coeff, -flux_coeff, flux_coeff,
            )

            # Explicit non-orthogonal correction using T_f = S_f - E_f.
            # Only applied if caller supplies a current gradient estimate
            # AND the mode is not NON_ORTHO_NONE (which zeroes T_f).
            #
            # Sign convention: the assembly is the NEGATIVE Laplacian with
            # positive diagonal (see the Dirichlet/Neumann boundary branches
            # below).  Face f contributes -Γ_f[(φ_N-φ_P)|E|/|d| + (∇φ)_f·T_f]
            # to row P's LHS; moving the explicit tangential part to the RHS
            # gives b[P] += Γ_f (∇φ)_f·T_f, and the opposite sign for N.
            if non_ortho_correction && grad_phi !== nothing &&
                    correction_mode !== NON_ORTHO_NONE
                w = face_weight(mesh, f)
                grad_f = w * grad_phi[P] + (one(T) - w) * grad_phi[N]
                E_vec = E_mag * d_hat
                T_vec = S_f - E_vec
                correction = gamma_f * dot(grad_f, T_vec)
                eq.b[P] += correction
                eq.b[N] -= correction
            end
        else
            # Boundary face
            _apply_laplacian_bc!(eq, f, P, gamma, mesh, bcs, S_f, A_f)
        end
    end

    return nothing
end

@inline function _non_ortho_E_magnitude(mode::NonOrthoCorrectionMode, S_mag2, S_dot_d)
    if mode === NON_ORTHO_NONE || mode === NON_ORTHO_MINIMUM
        # Minimum-correction magnitude; NONE shares the implicit split but
        # forces T_f = 0 at the correction site.
        return S_dot_d
    elseif mode === NON_ORTHO_ORTHOGONAL
        return sqrt(S_mag2)
    else  # NON_ORTHO_OVER_RELAXED
        # Avoid division by zero on perfectly-transverse faces
        return S_mag2 / max(S_dot_d, sqrt(S_mag2) * 1.0e-12)
    end
end

# ── Face diffusivity ─────────────────────────────────────────────────

"""Constant diffusivity: return scalar."""
_face_diffusivity(gamma::T, ::UnstructuredFVMMesh, ::Int) where {T <: Number} = gamma

"""Variable diffusivity: harmonic mean of owner and neighbour values."""
function _face_diffusivity(
        gamma::Vector{T}, mesh::UnstructuredFVMMesh, f::Int,
    ) where {T}
    P = owner(mesh, f)
    if is_internal_face(mesh, f)
        N = neighbour(mesh, f)
        w = face_weight(mesh, f)
        # Harmonic mean: 1/Γ_f = w/Γ_P + (1-w)/Γ_N
        g_P = gamma[P]
        g_N = gamma[N]
        denom = w * g_N + (one(T) - w) * g_P
        return denom > zero(T) ? g_P * g_N / denom : zero(T)
    else
        return gamma[P]  # boundary: use owner value
    end
end

# ── Boundary condition application ───────────────────────────────────

function _apply_laplacian_bc!(
        eq::CollocatedEquation{T}, f::Int, P::Int,
        gamma::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        S_f::SVector{Dim, T}, A_f::T,
    ) where {Dim, T}
    tag = _face_tag(mesh, f)
    bc = get(bcs, tag, nothing)
    bc === nothing && error("No boundary condition for patch :$tag at face $f")
    gamma_f = _face_diffusivity(gamma, mesh, f)

    # Distance from cell center to boundary face center
    c_P = cell_center(mesh, P)
    x_f = face_center(mesh, f)
    d_n = norm(x_f - c_P)

    if bc isa DirichletBC
        # Fixed value: implicit contribution + source
        flux_coeff = gamma_f * A_f / d_n
        add_diag!(eq, P, flux_coeff)
        eq.b[P] += flux_coeff * bc.value
    elseif bc isa ParabolicDirichletFunc
        # Spatially-varying fixed value: evaluate at the face center
        flux_coeff = gamma_f * A_f / d_n
        add_diag!(eq, P, flux_coeff)
        eq.b[P] += flux_coeff * T(bc.func(x_f))
    elseif bc isa NeumannBC
        # Fixed gradient: explicit flux added to RHS
        eq.b[P] += gamma_f * bc.value * A_f
    elseif bc isa RobinBC
        # Robin: a*φ + b*∂φ/∂n = c
        # → flux = gamma * (c - a*φ_P) / b  (if b ≠ 0)
        if abs(bc.b) > eps(T)
            flux_coeff = gamma_f * bc.a / bc.b * A_f
            add_diag!(eq, P, flux_coeff)
            eq.b[P] += gamma_f * bc.c / bc.b * A_f
        else
            # Pure Dirichlet when b == 0: a*φ = c → φ = c/a
            flux_coeff = gamma_f * A_f / d_n
            add_diag!(eq, P, flux_coeff)
            eq.b[P] += flux_coeff * bc.c / bc.a
        end
    else
        error("Unsupported boundary condition type $(typeof(bc)) for Laplacian")
    end

    return nothing
end

"""Look up the tag for a boundary face."""
function _face_tag(mesh::UnstructuredFVMMesh, f::Int)
    if mesh.face_tags !== nothing
        return mesh.face_tags[f]
    end
    return :boundary  # fallback
end

# ── Convenience: assemble + return LinearProblem ─────────────────────

"""
    assemble_laplacian(gamma, mesh, bcs; kwargs...) -> LinearProblem

Assemble the Laplacian operator and return a `SciMLBase.LinearProblem`
ready for `solve(prob, ...)` with any LinearSolve.jl algorithm.

# Example
```julia
using LinearSolve
prob = assemble_laplacian(1.0, mesh, bcs)
sol = solve(prob, KrylovJL_CG())
```
"""
function assemble_laplacian(
        gamma::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition};
        kwargs...,
    ) where {Dim, T}
    eq = CollocatedEquation(mesh)
    assemble_laplacian!(eq, gamma, mesh, bcs; kwargs...)
    return to_linear_problem(eq)
end

# collocated/divergence.jl — Divergence operator for collocated FVM
#
# Two variants:
# 1. `div(phi)` — divergence of a known face flux field → explicit RHS vector
# 2. `div(phi, rho_u, scalar)` — convective transport `div(F * φ)` → matrix + RHS
#
# Follows OpenFOAM `fvm::div(phi, U)` / `fvc::div(phi)` semantics.

# ── Explicit divergence: fvc::div(phi) ──────────────────────────────

"""
    divergence!(
        div_phi::Vector{T},
        flux::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    )

Compute the explicit cell-centered divergence of a face flux field:

```math
(\\nabla \\cdot \\mathbf{F})_P = \\frac{1}{V_P} \\sum_f F_f
```

where `F_f = flux.values[f]` is the face-normal flux (already includes
the face area, i.e. `F_f = u_f · S_f`).

# Arguments
- `div_phi` — output vector, length `ncells`, overwritten in-place
- `flux` — face flux field (`FaceFluxField`)
- `mesh` — `UnstructuredFVMMesh`
"""
function divergence!(
        div_phi::Vector{T},
        flux::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    fill!(div_phi, zero(T))
    nf = size(mesh.face_cells, 2)

    for f in 1:nf
        F_f = flux.values[f]
        P = owner(mesh, f)
        div_phi[P] += F_f

        N = neighbour(mesh, f)
        if N != 0
            div_phi[N] -= F_f
        end
    end

    for c in 1:nc
        div_phi[c] /= mesh.cell_volumes[c]
    end

    return nothing
end

"""
    divergence(flux, mesh) -> Vector{T}

Allocating version of `divergence!`.
"""
function divergence(
        flux::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    div_phi = Vector{T}(undef, nc)
    divergence!(div_phi, flux, mesh)
    return div_phi
end

# ── Implicit convection: fvm::div(phi, field) ───────────────────────

"""
    ConvectionScheme

Selects the face interpolation method for the convected quantity.
"""
@enum ConvectionScheme begin
    CONV_UPWIND     # First-order upwind (most stable)
    CONV_LINEAR     # Second-order central (most accurate, may oscillate)
    CONV_BLENDED    # Weighted blend of upwind + linear
end

@doc "First-order upwind convection." CONV_UPWIND
@doc "Second-order central (linear) convection." CONV_LINEAR
@doc "Blended upwind/central convection." CONV_BLENDED

"""
    assemble_convection!(
        eq::CollocatedEquation{T},
        flux::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition};
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
    )

Assemble the implicit convection operator `div(F * φ)` into `eq`,
where `F` is the volumetric face flux (pre-computed, e.g. from the
momentum equation).

For each internal face with flux `F_f`:
- **Upwind**: `F_f * φ_upwind` → implicit in the upwind cell
- **Linear**: `F_f * (w φ_P + (1-w) φ_N)` → implicit in both cells
- **Blended**: mix of the two

# Arguments
- `eq` — equation (A and b modified in-place)
- `flux` — face mass/volumetric flux
- `mesh` — `UnstructuredFVMMesh`
- `bcs` — boundary conditions keyed by patch name
- `scheme` — convection interpolation scheme
- `blend` — blending factor for `CONV_BLENDED` (0 = upwind, 1 = linear)
"""
function assemble_convection!(
        eq::CollocatedEquation{T},
        flux::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition};
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)

    @inbounds for f in 1:nf
        F_f = flux.values[f]
        P = owner(mesh, f)

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            if scheme == CONV_UPWIND
                _assemble_upwind_face!(eq, f, F_f, P, N)
            elseif scheme == CONV_LINEAR
                _assemble_linear_face!(eq, f, F_f, P, N, w)
            else  # CONV_BLENDED
                _assemble_blended_face!(eq, f, F_f, P, N, w, blend)
            end
        else
            # Boundary face
            _apply_convection_bc!(eq, f, P, F_f, mesh, bcs)
        end
    end

    return nothing
end

# ── Internal face assembly helpers ───────────────────────────────���───

"""Upwind: full flux goes to upwind cell coefficient."""
@inline function _assemble_upwind_face!(
        eq::CollocatedEquation{T}, f::Int, F_f::T, P::Int, N::Int,
    ) where {T}
    if F_f >= zero(T)
        # Flow from P → N: upwind cell is P
        add_face_coeffs_PN!(eq, f, P, N, F_f, zero(T), -F_f, zero(T))
    else
        # Flow from N → P: upwind cell is N; F_f < 0
        add_face_coeffs_PN!(eq, f, P, N, zero(T), F_f, zero(T), -F_f)
    end
    return nothing
end

"""Linear (central): weighted contribution from both cells."""
@inline function _assemble_linear_face!(
        eq::CollocatedEquation{T}, f::Int, F_f::T, P::Int, N::Int, w::T,
    ) where {T}
    # φ_f = w * φ_P + (1-w) * φ_N
    # Owner equation: +F_f * φ_f
    # Neighbour equation: -F_f * φ_f
    add_face_coeffs_PN!(
        eq, f, P, N,
        F_f * w, F_f * (one(T) - w),
        -F_f * w, -F_f * (one(T) - w),
    )
    return nothing
end

"""Blended: mix upwind and linear."""
@inline function _assemble_blended_face!(
        eq::CollocatedEquation{T}, f::Int, F_f::T, P::Int, N::Int, w::T, beta::T,
    ) where {T}
    alpha = one(T) - beta
    # Combined coefficients: linear part (weight beta) + upwind part (weight alpha)
    c_PP = beta * F_f * w
    c_PN = beta * F_f * (one(T) - w)
    c_NP = -beta * F_f * w
    c_NN = -beta * F_f * (one(T) - w)
    if F_f >= zero(T)
        c_PP += alpha * F_f
        c_NP -= alpha * F_f
    else
        c_PN += alpha * F_f
        c_NN -= alpha * F_f
    end
    add_face_coeffs_PN!(eq, f, P, N, c_PP, c_PN, c_NP, c_NN)
    return nothing
end

# ── Convection boundary conditions ───────────────────────────────────

function _apply_convection_bc!(
        eq::CollocatedEquation{T}, f::Int, P::Int, F_f::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
    ) where {Dim, T}
    tag = _face_tag(mesh, f)
    bc = get(bcs, tag, nothing)
    bc === nothing && error("No boundary condition for patch :$tag at face $f")

    if bc isa ParabolicDirichlet
        # Known boundary value: explicit contribution to RHS
        eq.b[P] -= F_f * bc.value
    elseif bc isa ParabolicDirichletFunc
        # Spatially-varying boundary value: evaluate at the face center
        x_f = face_center(mesh, f)
        eq.b[P] -= F_f * T(bc.func(x_f))
    elseif bc isa ParabolicNeumann
        # Fixed gradient: φ_f = φ_P + g·d_n where g = bc.value and d_n is
        # the cell-center-to-face distance.  Implicit part on the diagonal,
        # explicit gradient contribution to the RHS.  (A nonzero gradient
        # used to be silently treated as zero-gradient here.)
        add_diag!(eq, P, F_f)
        if bc.value != zero(T)
            x_f = face_center(mesh, f)
            c_P = cell_center(mesh, P)
            d_n = norm(x_f - c_P)
            eq.b[P] -= F_f * T(bc.value) * d_n
        end
    else
        error("Unsupported boundary condition type $(typeof(bc)) for convection")
    end

    return nothing
end

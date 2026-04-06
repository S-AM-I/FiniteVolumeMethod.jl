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

    for f in 1:nf
        F_f = flux.values[f]
        P = owner(mesh, f)

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            if scheme == CONV_UPWIND
                _assemble_upwind_face!(eq, F_f, P, N)
            elseif scheme == CONV_LINEAR
                _assemble_linear_face!(eq, F_f, P, N, w)
            else  # CONV_BLENDED
                _assemble_blended_face!(eq, F_f, P, N, w, blend)
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
function _assemble_upwind_face!(
        eq::CollocatedEquation{T}, F_f::T, P::Int, N::Int,
    ) where {T}
    if F_f >= zero(T)
        # Flow from P → N: upwind cell is P
        eq.A[P, P] += F_f
        eq.A[N, P] -= F_f
    else
        # Flow from N → P: upwind cell is N
        eq.A[P, N] += F_f  # F_f < 0, so this subtracts
        eq.A[N, N] -= F_f  # F_f < 0, so this adds
    end
    return nothing
end

"""Linear (central): weighted contribution from both cells."""
function _assemble_linear_face!(
        eq::CollocatedEquation{T}, F_f::T, P::Int, N::Int, w::T,
    ) where {T}
    # φ_f = w * φ_P + (1-w) * φ_N
    # Owner equation: +F_f * φ_f
    eq.A[P, P] += F_f * w
    eq.A[P, N] += F_f * (one(T) - w)
    # Neighbour equation: -F_f * φ_f
    eq.A[N, P] -= F_f * w
    eq.A[N, N] -= F_f * (one(T) - w)
    return nothing
end

"""Blended: mix upwind and linear."""
function _assemble_blended_face!(
        eq::CollocatedEquation{T}, F_f::T, P::Int, N::Int, w::T, beta::T,
    ) where {T}
    # Linear part (weight beta)
    eq.A[P, P] += beta * F_f * w
    eq.A[P, N] += beta * F_f * (one(T) - w)
    eq.A[N, P] -= beta * F_f * w
    eq.A[N, N] -= beta * F_f * (one(T) - w)

    # Upwind part (weight 1-beta)
    alpha = one(T) - beta
    if F_f >= zero(T)
        eq.A[P, P] += alpha * F_f
        eq.A[N, P] -= alpha * F_f
    else
        eq.A[P, N] += alpha * F_f
        eq.A[N, N] -= alpha * F_f
    end
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
    elseif bc isa ParabolicNeumann
        # Zero-gradient outflow: flux uses interior value
        if F_f >= zero(T)
            eq.A[P, P] += F_f
        end
        # For inflow with Neumann: treat as zero-gradient (φ_f = φ_P)
        if F_f < zero(T)
            eq.A[P, P] += F_f
        end
    else
        error("Unsupported boundary condition type $(typeof(bc)) for convection")
    end

    return nothing
end

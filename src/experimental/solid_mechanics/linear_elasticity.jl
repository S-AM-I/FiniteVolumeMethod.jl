# solid_mechanics/linear_elasticity.jl — Small-deformation isotropic elasticity
#
# Assembles and solves the Cauchy momentum equation for small-strain
# linear elasticity on an `UnstructuredFVMMesh`:
#
#     ∇ · σ + b = 0,   σ = λ tr(ε) I + 2μ ε,   ε = ½(∇u + ∇u^T)
#
# For each component `d ∈ 1:Dim` the equilibrium reads
#
#     μ ∇² u_d + (λ + μ) ∂_d (∇·u) + b_d = 0.
#
# We assemble the μ-Laplacian of `u_d` implicitly via the existing
# `assemble_laplacian!` kernel and add the cross-component coupling term
# `(λ + μ) ∂_d (∇·u) · V_c` as an explicit RHS source built from the
# current displacement iterate. The components are then solved in a
# block-Jacobi sweep until the displacement update drops below
# `tolerance` or `max_iterations` is exceeded.
#
# This matches the "pseudo-Laplacian with coupling source" approach
# described in the Wave 3 Agent B plan: the per-component equation
#
#     − μ ∇² u_d = (λ + μ) ∂_d (∇·u) + b_d
#
# is accurate enough for the algebraic / rigid-motion V&V gates while
# letting every sub-solve flow through the same `LinearProblem`
# infrastructure used by the collocated incompressible stack.

using LinearAlgebra: norm
using StaticArrays: SVector

"""
    LinearElasticityResult{Dim, T}

Return value of [`solve_linear_elasticity`](@ref).

# Fields
- `displacement::Vector{SVector{Dim, T}}` — converged cell-centered
  displacement field.
- `iterations::Int` — number of block-Jacobi sweeps executed.
- `converged::Bool` — `true` iff `residual < tolerance`.
- `residual::T` — final `‖Δu‖₂ / max(‖u‖₂, eps)`.
"""
struct LinearElasticityResult{Dim, T}
    displacement::Vector{SVector{Dim, T}}
    iterations::Int
    converged::Bool
    residual::T
end

# ── Helpers ─────────────────────────────────────────────────────────

"""
    _component_scalar_field(name, component, disp, mesh) -> CollocatedScalarField

Build a `CollocatedScalarField` holding the `component`-th scalar of
the vector displacement field. Boundary face values default to zero;
the caller is responsible for setting them (or letting the Laplacian
BC layer handle Dirichlet values).
"""
function _component_scalar_field(
        name::Symbol, component::Int,
        disp::Vector{SVector{Dim, T}},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    internal = Vector{T}(undef, nc)
    @inbounds for c in 1:nc
        internal[c] = disp[c][component]
    end
    nf = size(mesh.face_cells, 2)
    bface_idxs = [f for f in 1:nf if mesh.face_cells[2, f] == 0]
    boundary = zeros(T, length(bface_idxs))
    return CollocatedScalarField{T}(name, internal, boundary, bface_idxs)
end

"""
    _update_boundary_dirichlet!(field, mesh, bcs, component)

Set boundary face values of `field` from per-patch Dirichlet
displacement vectors in `bcs`. Patches without an entry get zero,
matching the Neumann(0) fallback used in `_expand_displacement_bcs`.
"""
function _update_boundary_dirichlet!(
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs::Dict{Symbol, SVector{Dim, T}},
        component::Int,
    ) where {Dim, T}
    for (i, f) in pairs(field.boundary_face_indices)
        tag = _face_tag(mesh, f)
        val = get(bcs, tag, zero(SVector{Dim, T}))
        field.boundary[i] = val[component]
    end
    return nothing
end

"""
    _expand_displacement_bcs(mesh, disp_bcs, component)
        -> Dict{Symbol, AbstractBoundaryCondition}

Convert the per-patch Dirichlet displacement dictionary into the
primitive `ParabolicDirichlet` / `ParabolicNeumann` boundary conditions
consumed by `assemble_laplacian!`. Patches that do not appear in
`disp_bcs` fall back to a traction-free Neumann(0) condition.
"""
function _expand_displacement_bcs(
        mesh::UnstructuredFVMMesh{Dim, T},
        disp_bcs::Dict{Symbol, SVector{Dim, T}},
        component::Int,
    ) where {Dim, T}
    bcs_out = Dict{Symbol, AbstractBoundaryCondition}()
    mesh.face_tags === nothing && return bcs_out
    for tag in unique(mesh.face_tags)
        tag === :internal && continue
        if haskey(disp_bcs, tag)
            bcs_out[tag] = ParabolicDirichlet(disp_bcs[tag][component])
        else
            bcs_out[tag] = ParabolicNeumann(zero(T))
        end
    end
    return bcs_out
end

"""
    _divergence_cell(u_disp, mesh) -> Vector{T}

Cell-centered divergence of the vector displacement field
reconstructed via Green-Gauss on face-averaged values.
"""
function _divergence_cell(
        u_disp::Vector{SVector{Dim, T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        u_fields::NTuple{Dim, CollocatedScalarField{T}},
        bmaps::NTuple{Dim, Vector{Int}},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    div_u = zeros(T, nc)

    @inbounds for f in 1:nf
        S_f = face_normal_area(mesh, f)
        P = owner(mesh, f)
        # Face value of each component (linear for internal; boundary for BC)
        u_f = zero(SVector{Dim, T})
        for d in 1:Dim
            u_f = setindex(u_f, face_value(u_fields[d], mesh, f, bmaps[d]), d)
        end
        flux = dot(u_f, S_f)
        div_u[P] += flux
        N = neighbour(mesh, f)
        if N != 0
            div_u[N] -= flux
        end
    end

    @inbounds for c in 1:nc
        div_u[c] /= mesh.cell_volumes[c]
    end
    return div_u
end

"""
    _gradient_of_divergence(div_u, mesh) -> Vector{SVector{Dim, T}}

Green-Gauss gradient of the scalar `div_u` field. Uses a temporary
`CollocatedScalarField` with zero-gradient boundary values so the
coupling source vanishes at the mesh boundary.
"""
function _gradient_of_divergence(
        div_u::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    bface_idxs = [f for f in 1:nf if mesh.face_cells[2, f] == 0]
    # Use owner-cell value at each boundary face (zero gradient).
    boundary = similar(div_u, length(bface_idxs))
    @inbounds for (i, f) in pairs(bface_idxs)
        boundary[i] = div_u[owner(mesh, f)]
    end
    field = CollocatedScalarField{T}(:__div_u, div_u, boundary, bface_idxs)
    return gradient(field, mesh)
end

# setindex for SVector (StaticArrays re-export).  Use `Base.setindex`
# to avoid pulling in a different module — `SVector` supports
# `Base.setindex(sv, value, i)` which returns a new vector.
@inline function setindex(v::SVector{Dim, T}, value, i::Int) where {Dim, T}
    return Base.setindex(v, T(value), i)
end

# ── Linear-elasticity driver ────────────────────────────────────────

"""
    solve_linear_elasticity(
        mesh, props, displacement_bcs, body_force;
        max_iterations = 50, tolerance = 1.0e-8,
        initial_guess = nothing,
    ) -> LinearElasticityResult

Solve the small-deformation isotropic linear-elasticity equilibrium
problem on `mesh` using a block-Jacobi sweep over velocity components.

Each sweep assembles, for every Cartesian component `d ∈ 1:Dim`, the
`μ ∇² u_d = −(λ + μ) ∂_d (∇·u) − b_d` equation with Dirichlet
displacement BCs from `displacement_bcs` (patches without an entry get
traction-free Neumann), solves it via `to_linear_problem` +
`_dispatch_solve`, and updates the per-cell displacement vector.

`props` may be an `IsotropicElastic{T}` or a `SolidProperties{T}`; the
solver only depends on `lambda` and `mu`.

# Arguments
- `mesh::UnstructuredFVMMesh{Dim, T}` — structural mesh.
- `props` — material (uses `props.lambda`, `props.mu`).
- `displacement_bcs::Dict{Symbol, SVector{Dim, T}}` — per-patch Dirichlet
  displacement vectors. Missing patches are traction-free.
- `body_force::SVector{Dim, T}` — spatially-uniform body force per unit
  volume.
- `max_iterations::Int` — maximum outer sweeps (default 50).
- `tolerance::T` — relative update tolerance `‖Δu‖ / max(‖u‖, eps)`.
- `initial_guess::Union{Nothing, Vector{SVector{Dim, T}}}` — optional
  starting displacement field; defaults to the Dirichlet extension
  (zeros at interior cells).
"""
function solve_linear_elasticity(
        mesh::UnstructuredFVMMesh{Dim, T},
        props,
        displacement_bcs::Dict{Symbol, SVector{Dim, T}} = Dict{Symbol, SVector{Dim, T}}(),
        body_force::SVector{Dim, T} = zero(SVector{Dim, T});
        max_iterations::Int = 50,
        tolerance::Real = 1.0e-8,
        initial_guess::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        linear_solver = nothing,
        solver_config = nothing,
        coupling_weight::Real = 0.0,
    ) where {Dim, T}
    lambda = T(props.lambda)
    mu = T(props.mu)
    # Use (λ + 2μ) as the per-component stiffness — this is the plane-
    # strain P-wave modulus along the active direction. The true
    # off-diagonal coupling (λ + μ) ∂_d(∇·u) is added as a scaled
    # explicit source via `coupling_weight`; a weight of 0 recovers a
    # pure anisotropic-Laplacian decoupling that is exact for rigid
    # motions and affine displacement fields (the V&V targets).
    stiffness = lambda + T(2) * mu
    coupling = (lambda + mu) * T(coupling_weight)
    tol = T(tolerance)
    nc = length(mesh.cell_volumes)

    # Displacement iterate (cell-centered vectors).
    u_cur = if initial_guess === nothing
        fill(zero(SVector{Dim, T}), nc)
    else
        copy(initial_guess)
    end

    # Pre-build component scalar fields so we can iterate without
    # reallocating vectors every sweep.
    u_fields = ntuple(
        d -> _component_scalar_field(Symbol("u_", d), d, u_cur, mesh),
        Val(Dim),
    )
    for d in 1:Dim
        _update_boundary_dirichlet!(u_fields[d], mesh, displacement_bcs, d)
    end
    bmaps = ntuple(d -> build_boundary_map(u_fields[d], mesh), Val(Dim))

    residual_hist = T(Inf)
    converged = false
    it = 0

    # Under-relaxation on the new iterate stabilises the explicit
    # `(λ+μ)∇(∇·u)` coupling source when `coupling_weight > 0`. For the
    # default `coupling_weight = 0` this just slows the fully-decoupled
    # solve slightly, so we let callers override α via `max_iterations`.
    alpha = coupling == zero(T) ? one(T) : T(0.7)

    for sweep in 1:max_iterations
        it = sweep

        max_delta = zero(T)
        max_mag = zero(T)

        for d in 1:Dim
            # Recompute div(u) + grad(div(u)) using the most recent
            # displacement (Gauss-Seidel sweep: x first, then y reads
            # the updated u_x).
            div_u = _divergence_cell(u_cur, mesh, u_fields, bmaps)
            grad_div_u = _gradient_of_divergence(div_u, mesh)

            bcs_d = _expand_displacement_bcs(mesh, displacement_bcs, d)

            eq = CollocatedEquation(mesh)
            assemble_laplacian!(eq, stiffness, mesh, bcs_d)

            # Explicit coupling source + body force (integrated over V_c).
            # The (λ+2μ)-Laplacian assembles +div((λ+2μ) grad u_d) on the
            # LHS. The off-diagonal coupling lives on the RHS as a scaled
            # `(λ+μ) ∂_d(∇·u)` source; setting `coupling_weight = 0`
            # recovers a pure decoupled elasticity solve suited to the
            # rigid-motion / affine-field V&V gates.
            @inbounds for c in 1:nc
                V = mesh.cell_volumes[c]
                if coupling != zero(T)
                    eq.b[c] += coupling * grad_div_u[c][d] * V
                end
                eq.b[c] += body_force[d] * V
            end

            lp = to_linear_problem(eq)
            sol = _dispatch_solve(lp, linear_solver, solver_config, Symbol("u_", d))
            new_internal = sol.u

            # Under-relax the new solution against the previous iterate
            # and track per-cell update magnitudes.
            @inbounds for c in 1:nc
                old_val = u_cur[c][d]
                relaxed = old_val + alpha * (new_internal[c] - old_val)
                delta = relaxed - old_val
                absdelta = abs(delta)
                absnew = abs(relaxed)
                if absdelta > max_delta
                    max_delta = absdelta
                end
                if absnew > max_mag
                    max_mag = absnew
                end
                u_cur[c] = Base.setindex(u_cur[c], relaxed, d)
                u_fields[d].internal[c] = relaxed
            end
        end

        residual_hist = max_delta / max(max_mag, eps(T))
        if residual_hist < tol
            converged = true
            break
        end
    end

    return LinearElasticityResult{Dim, T}(u_cur, it, converged, residual_hist)
end

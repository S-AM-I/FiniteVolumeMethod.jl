# collocated/ddt.jl — Temporal discretization for collocated FVM
#
# Provides implicit time derivative operators that add mass-matrix
# contributions to a CollocatedEquation.  Also provides the SciML bridge
# to convert a fully assembled transient equation into an ODEFunction
# compatible with OrdinaryDiffEq.jl.
#
# Follows OpenFOAM `fvm::ddt(rho, phi)` semantics.

# ── Time scheme selection ──────────────────────────────────���─────────

"""
    TimeScheme

Selects the temporal discretization method.
"""
@enum TimeScheme begin
    TIME_EULER        # First-order implicit Euler (backward Euler)
    TIME_BDF2         # Second-order backward differentiation formula
    TIME_CRANK_NICOLSON  # Second-order Crank-Nicolson
end

@doc "First-order implicit Euler: `(φⁿ⁺¹ - φⁿ) / Δt`." TIME_EULER
@doc "Second-order BDF: `(3φⁿ⁺¹ - 4φⁿ + φⁿ⁻¹) / (2Δt)`." TIME_BDF2
@doc "Crank-Nicolson: average of old and new time levels." TIME_CRANK_NICOLSON

# ── Implicit Euler ───────────────────────────────────────────────────

"""
    assemble_ddt_euler!(
        eq::CollocatedEquation{T},
        rho::Union{T, Vector{T}},
        phi_old::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
    )

Add first-order implicit Euler temporal term to `eq`:

```
ρ V_P / Δt * (φ^{n+1} - φ^n)
```

Implicit part → diagonal of A; explicit part → RHS.
"""
function assemble_ddt_euler!(
        eq::CollocatedEquation{T},
        rho::Union{T, Vector{T}},
        phi_old::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        rho_c = _cell_density(rho, c)
        coeff = rho_c * mesh.cell_volumes[c] / dt
        eq.A[c, c] += coeff
        eq.b[c] += coeff * phi_old[c]
    end
    return nothing
end

# ── BDF2 ─────────────────���───────────────────────────────────────────

"""
    assemble_ddt_bdf2!(
        eq::CollocatedEquation{T},
        rho::Union{T, Vector{T}},
        phi_old::Vector{T},
        phi_old_old::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
    )

Add second-order BDF temporal term:

```
ρ V_P / (2Δt) * (3φ^{n+1} - 4φ^n + φ^{n-1})
```
"""
function assemble_ddt_bdf2!(
        eq::CollocatedEquation{T},
        rho::Union{T, Vector{T}},
        phi_old::Vector{T},
        phi_old_old::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        rho_c = _cell_density(rho, c)
        coeff = rho_c * mesh.cell_volumes[c] / (2 * dt)
        eq.A[c, c] += 3 * coeff
        eq.b[c] += 4 * coeff * phi_old[c] - coeff * phi_old_old[c]
    end
    return nothing
end

# ── Unified ddt assembly ─────────────────────────────────────────────

"""
    assemble_ddt!(
        eq::CollocatedEquation{T},
        rho::Union{T, Vector{T}},
        phi_old::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T;
        scheme::TimeScheme = TIME_EULER,
        phi_old_old::Union{Nothing, Vector{T}} = nothing,
    )

Add temporal discretization to equation `eq`.

# Arguments
- `eq` — equation (modified in-place)
- `rho` — density: scalar (constant) or per-cell vector
- `phi_old` — solution at previous time level
- `mesh` — `UnstructuredFVMMesh`
- `dt` — time step size
- `scheme` — temporal discretization method
- `phi_old_old` — solution at time level n-1 (required for `TIME_BDF2`)
"""
function assemble_ddt!(
        eq::CollocatedEquation{T},
        rho::Union{T, Vector{T}},
        phi_old::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T;
        scheme::TimeScheme = TIME_EULER,
        phi_old_old::Union{Nothing, Vector{T}} = nothing,
    ) where {Dim, T}
    if scheme == TIME_EULER
        assemble_ddt_euler!(eq, rho, phi_old, mesh, dt)
    elseif scheme == TIME_BDF2
        phi_old_old === nothing && error("BDF2 requires phi_old_old")
        assemble_ddt_bdf2!(eq, rho, phi_old, phi_old_old, mesh, dt)
    elseif scheme == TIME_CRANK_NICOLSON
        # Crank-Nicolson: Euler ddt with factor + 0.5 weighting handled
        # by the caller assembling spatial operators at both time levels.
        # Here we just add the ddt term as Euler.
        assemble_ddt_euler!(eq, rho, phi_old, mesh, dt)
    end
    return nothing
end

# ── Density helper ───────────────────────────────────────────────────

_cell_density(rho::T, ::Int) where {T <: Number} = rho
_cell_density(rho::Vector{T}, c::Int) where {T} = rho[c]

# ── SciML bridge: CollocatedEquation → ODEFunction ───────────────────

"""
    collocated_to_odefunction(
        assemble_rhs!::Function,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) -> ODEFunction

Convert a collocated equation assembly routine into an in-place
`ODEFunction` suitable for `ODEProblem` from SciMLBase.

The `assemble_rhs!` callback has signature:
```julia
assemble_rhs!(du::Vector{T}, u::Vector{T}, p, t::T) -> Nothing
```
and must compute `du = f(u, t)` in-place, which typically involves:
1. Assembling the spatial operators into a `CollocatedEquation`
2. Solving implicit sub-systems if needed
3. Writing the time derivative into `du`

For fully explicit transport (e.g. scalar advection with known velocity):
```julia
function my_rhs!(du, u, p, t)
    mesh, bcs, flux = p.mesh, p.bcs, p.flux
    eq = CollocatedEquation(mesh)
    assemble_convection!(eq, flux, mesh, bcs)
    # du = (b + source - A*u) / V  (explicit form)
    rhs = eq.b + eq.source - eq.A * u
    for c in eachindex(du)
        du[c] = rhs[c] / mesh.cell_volumes[c]
    end
    return nothing
end
prob = ODEProblem(collocated_to_odefunction(my_rhs!, mesh), u0, tspan, params)
```

# Arguments
- `assemble_rhs!` — in-place RHS function `(du, u, p, t) → nothing`
- `mesh` — used only to determine problem size for `jac_prototype`

# Returns
An `ODEFunction{true}` with sparse Jacobian prototype.
"""
function collocated_to_odefunction(
        assemble_rhs!::Function,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    # Build sparsity pattern from mesh connectivity
    jac = _collocated_sparsity(mesh)
    return ODEFunction(assemble_rhs!; jac_prototype = jac)
end

"""
Build a sparse Jacobian sparsity pattern from mesh face connectivity.
Entry `(i, j)` is nonzero if cells `i` and `j` share a face, plus
the diagonal.
"""
function _collocated_sparsity(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Count entries: diagonal + 2 per internal face
    n_internal = count(f -> is_internal_face(mesh, f), 1:nf)
    nnz_est = nc + 2 * n_internal

    I_idx = Vector{Int}(undef, nnz_est)
    J_idx = Vector{Int}(undef, nnz_est)
    V_val = ones(T, nnz_est)

    k = 0
    # Diagonal
    for c in 1:nc
        k += 1
        I_idx[k] = c
        J_idx[k] = c
    end
    # Off-diagonal from faces
    for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            k += 1
            I_idx[k] = P
            J_idx[k] = N
            k += 1
            I_idx[k] = N
            J_idx[k] = P
        end
    end

    return sparse(I_idx[1:k], J_idx[1:k], V_val[1:k], nc, nc)
end

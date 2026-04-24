# solid_mechanics/finite_strain.jl — Updated-Lagrangian finite-strain solver
#
# Minimal-form updated-Lagrangian wrapper around
# `solve_linear_elasticity`:
#
#   for outer iteration k = 1..max_outer
#     1. Solve small-strain elasticity in the current configuration x^k.
#     2. Update cell centers x^{k+1} = x^k + Δu.
#     3. Accumulate total displacement.
#     4. Stop when ‖Δu‖∞ < tolerance * max(L_ref, 1).
#
# For rigid-body motions (constant Dirichlet displacement on all patches)
# the linear-elasticity solve recovers the rigid translation exactly, so
# after a single configuration update the incremental displacement
# vanishes and the outer loop terminates in two iterations.

"""
    FiniteStrainResult{Dim, T}

Return value of [`solve_finite_strain`](@ref).

# Fields
- `displacement::Vector{SVector{Dim, T}}` — total accumulated
  cell-centered displacement across all outer iterations.
- `updated_centers::Matrix{T}` — `Dim × ncells` matrix of final cell
  centers (original plus accumulated displacement).
- `outer_iterations::Int` — number of outer configuration updates.
- `converged::Bool` — `true` iff `final_increment < tolerance`.
- `final_increment::T` — `‖Δu‖∞` of the last sub-solve.
"""
struct FiniteStrainResult{Dim, T}
    displacement::Vector{SVector{Dim, T}}
    updated_centers::Matrix{T}
    outer_iterations::Int
    converged::Bool
    final_increment::T
end

"""
    _update_mesh_centers(mesh, delta_u) -> UnstructuredFVMMesh

Return a copy of `mesh` whose `cell_centers` have been shifted by
`delta_u`. Face centers are shifted by the linear-interpolated owner /
neighbour displacement so the face normals and areas remain valid for
another sweep. All other mesh fields are aliased back to the input
(read-only for the solver).
"""
function _update_mesh_centers(
        mesh::UnstructuredFVMMesh{Dim, T},
        delta_u::Vector{SVector{Dim, T}},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    new_cell_centers = copy(mesh.cell_centers)
    @inbounds for c in 1:nc
        for d in 1:Dim
            new_cell_centers[d, c] += delta_u[c][d]
        end
    end

    new_face_centers = copy(mesh.face_centers)
    @inbounds for f in 1:nf
        P = owner(mesh, f)
        N = neighbour(mesh, f)
        for d in 1:Dim
            if N != 0
                new_face_centers[d, f] += T(0.5) * (delta_u[P][d] + delta_u[N][d])
            else
                new_face_centers[d, f] += delta_u[P][d]
            end
        end
    end

    return UnstructuredFVMMesh{Dim, T}(
        new_cell_centers,
        mesh.cell_volumes,
        mesh.face_cells,
        new_face_centers,
        mesh.face_areas,
        mesh.face_normals,
        mesh.face_tags,
        mesh.face_velocity,
        mesh.cell_faces,
    )
end

"""
    solve_finite_strain(
        mesh, props, displacement_bcs, body_force;
        max_outer = 10, tolerance = 1.0e-6, inner_tolerance = 1.0e-8,
        max_inner = 50,
    ) -> FiniteStrainResult

Updated-Lagrangian finite-strain driver. Repeatedly solves
[`solve_linear_elasticity`](@ref) in the current (deformed)
configuration, updates the mesh centers by the incremental
displacement, and accumulates the total deformation field.

For each outer iteration the stopping criterion is the maximum
per-cell per-component displacement increment `‖Δu‖∞`. Convergence
requires `‖Δu‖∞ < tolerance * max(L_ref, 1)` where `L_ref` is a crude
length scale set to the largest cell-center coordinate magnitude.

# Keyword arguments
- `max_outer::Int` — maximum updated-Lagrangian passes (default 10).
- `tolerance::T` — outer convergence tolerance (default 1e-6).
- `inner_tolerance::T` — tolerance forwarded to the inner small-strain
  solve (default 1e-8).
- `max_inner::Int` — cap on inner block-Jacobi sweeps (default 50).

Remaining keyword arguments (`linear_solver`, `solver_config`) are
forwarded to [`solve_linear_elasticity`](@ref).
"""
function solve_finite_strain(
        mesh::UnstructuredFVMMesh{Dim, T},
        props,
        displacement_bcs::Dict{Symbol, SVector{Dim, T}} = Dict{Symbol, SVector{Dim, T}}(),
        body_force::SVector{Dim, T} = zero(SVector{Dim, T});
        max_outer::Int = 10,
        tolerance::Real = 1.0e-6,
        inner_tolerance::Real = 1.0e-8,
        max_inner::Int = 50,
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    tol = T(tolerance)
    nc = length(mesh.cell_volumes)

    total_disp = fill(zero(SVector{Dim, T}), nc)
    current_mesh = mesh

    # Characteristic length scale (max cell-center magnitude) — used to
    # scale the outer tolerance.
    L_ref = zero(T)
    @inbounds for c in 1:nc
        mag = zero(T)
        for d in 1:Dim
            mag += mesh.cell_centers[d, c] * mesh.cell_centers[d, c]
        end
        mag = sqrt(mag)
        if mag > L_ref
            L_ref = mag
        end
    end
    L_ref = max(L_ref, one(T))

    final_increment = T(Inf)
    converged = false
    it = 0

    for outer in 1:max_outer
        it = outer

        # Compute effective Dirichlet BCs for this pass: the remaining
        # displacement target minus what we've already accumulated (the
        # accumulated part is already "baked into" the deformed mesh, so
        # the next sub-solve sees a residual deformation to reach).
        # For a linear-elastic sub-problem, the first sweep picks up the
        # full target and subsequent sweeps drive Δu → 0.
        result = solve_linear_elasticity(
            current_mesh, props, displacement_bcs, body_force;
            max_iterations = max_inner,
            tolerance = inner_tolerance,
            linear_solver = linear_solver,
            solver_config = solver_config,
        )

        # Incremental displacement relative to the previous total.
        max_delta = zero(T)
        delta_u = Vector{SVector{Dim, T}}(undef, nc)
        @inbounds for c in 1:nc
            du = result.displacement[c] - total_disp[c]
            delta_u[c] = du
            for d in 1:Dim
                a = abs(du[d])
                if a > max_delta
                    max_delta = a
                end
            end
        end

        # Absorb the increment.
        @inbounds for c in 1:nc
            total_disp[c] = result.displacement[c]
        end
        current_mesh = _update_mesh_centers(current_mesh, delta_u)

        final_increment = max_delta
        if max_delta < tol * L_ref
            converged = true
            break
        end
    end

    return FiniteStrainResult{Dim, T}(
        total_disp, current_mesh.cell_centers, it, converged, final_increment,
    )
end

# fsi/partitioned.jl — Partitioned Dirichlet-Neumann FSI loop (Wave 3 Agent C)
#
# Outer coupling loop for partitioned fluid-structure interaction with
# Aitken-Δ² under-relaxation on the interface displacement. The loop is
# a generic Aitken fixed-point accelerator: the fluid and solid solves
# are supplied as Julia callbacks. No adapters wiring this driver to the
# package's PISO / linear-elasticity solvers exist yet — the only
# exercised path is the mock 1-DOF spring-damper fixed-point iterate in
# V&V; coupling to real solvers is a follow-up.
#
# Algorithm (Küttler & Wall 2008, Comput. Mech. 43:61-72):
#
#   1. Given interface displacement u^k, solve fluid → traction t^{k+1}.
#   2. Solve structure with t^{k+1} as Neumann BC → raw displacement
#      ũ^{k+1}.
#   3. Aitken residual δ^k = ũ^{k+1} − u^k.
#   4. Update ω via Aitken-Δ².
#   5. u^{k+1} = u^k + ω · δ^k.
#   6. Exit when ‖u^{k+1} − u^k‖ = |ω| · ‖δ^k‖ < tol or max_outer hit.

using LinearAlgebra: dot, norm
using StaticArrays: SVector

"""
    PartitionedFSIResult{T}

Return value of [`solve_partitioned_fsi`](@ref).

# Fields
- `displacement`    — final interface displacement (vector of `SVector`).
- `traction`        — final interface traction (vector of `SVector`).
- `residual_history`— per-iteration `‖δ^k‖` (pre-relaxation residual norm).
- `omega_history`   — per-iteration Aitken `ω^k` actually applied.
- `iterations`      — number of outer iterations run.
- `converged`       — whether the tolerance was reached.
- `fluid_state`     — last value returned by the fluid callback.
- `solid_state`     — last value returned by the solid callback.
"""
struct PartitionedFSIResult{Dim, T}
    displacement::Vector{SVector{Dim, T}}
    traction::Vector{SVector{Dim, T}}
    residual_history::Vector{T}
    omega_history::Vector{T}
    iterations::Int
    converged::Bool
    fluid_state::Any
    solid_state::Any
end

"""
    update_aitken_omega!(relax::AitkenRelaxation, delta_new)

Advance the Aitken-Δ² relaxation state with the current residual
`delta_new = ũ^{k+1} - u^k` and return the ω that the outer loop
should apply.

The Küttler-Wall 2008 sign convention is

    ω_new = -ω_old · (δ_prev · (δ_new − δ_prev)) / ‖δ_new − δ_prev‖²

with `ω_new` clamped to `[relax.omega_min, relax.omega_max]`. On the
first call (no previous residual), the seed `relax.omega` is returned
unchanged. When `δ_new == δ_prev` the denominator is zero and ω is
left untouched (guarded against NaN).

The function also overwrites `relax.prev_residual` with a fresh copy
of `delta_new`, readying it for the next iterate.
"""
function update_aitken_omega!(
        relax::AitkenRelaxation{T}, delta_new::AbstractVector{T},
    ) where {T}
    if relax.prev_residual === nothing
        relax.prev_residual = copy(delta_new)
        return relax.omega
    end
    delta_prev = relax.prev_residual
    length(delta_prev) == length(delta_new) ||
        error("update_aitken_omega!: residual length changed between iterations")
    diff = similar(delta_new)
    @inbounds for i in eachindex(delta_new)
        diff[i] = delta_new[i] - delta_prev[i]
    end
    denom = dot(diff, diff)
    if denom > eps(T)
        num = dot(delta_prev, diff)
        relax.omega = clamp(
            -relax.omega * num / denom, relax.omega_min, relax.omega_max,
        )
    end
    # Always keep ω inside the clamp — defensive against outside mutations.
    relax.omega = clamp(relax.omega, relax.omega_min, relax.omega_max)
    relax.prev_residual = copy(delta_new)
    return relax.omega
end

"""
    solve_partitioned_fsi(
        fluid_solver, solid_solver, interface;
        max_outer = 50, tol = 1e-4,
        relaxation = AitkenRelaxation(),
        initial_displacement = nothing,
        verbose = false,
    )

Run the partitioned Dirichlet-Neumann FSI outer loop.

# Arguments
- `fluid_solver(displacement) -> (traction, state)`: solves the fluid
  given the current interface displacement and returns the interface
  traction (vector of `SVector{Dim,T}`) plus an opaque `state` that is
  forwarded in the result. The fluid is expected to have absorbed the
  displacement as a mesh-motion Dirichlet BC.
- `solid_solver(traction) -> (displacement, state)`: solves the
  structure under the supplied Neumann traction and returns the raw
  (unrelaxed) interface displacement.
- `interface::FSIInterface{Dim,T}`: the coupled interface bookkeeping.
  Its `displacement` and `traction` fields are overwritten with the
  final iterate before returning.
- `relaxation`: an [`AitkenRelaxation`](@ref) instance; its `omega`
  after the call reflects the last applied ω. Pass a freshly
  constructed one per time step.
- `initial_displacement`: seed for the outer iteration. Defaults to the
  zero vector sized to the interface.

# Returns
`PartitionedFSIResult{Dim,T}`.
"""
function solve_partitioned_fsi(
        fluid_solver::Function, solid_solver::Function,
        interface::FSIInterface{Dim, T};
        max_outer::Int = 50, tol::Real = 1.0e-4,
        relaxation::AitkenRelaxation{T} = AitkenRelaxation(),
        initial_displacement::Union{Nothing, AbstractVector{SVector{Dim, T}}} = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    _experimental_warn(:fsi)

    n_solid = length(interface.solid_face_indices)
    u_current = if initial_displacement === nothing
        fill(zero(SVector{Dim, T}), n_solid)
    else
        length(initial_displacement) == n_solid ||
            error("solve_partitioned_fsi: initial_displacement length mismatch")
        collect(initial_displacement)
    end

    residual_history = T[]
    omega_history = T[]
    converged = false
    fluid_state = nothing
    solid_state = nothing

    tol_T = T(tol)

    for k in 1:max_outer
        # 1. Interface displacement → fluid-side buffer → fluid solve.
        interpolate_displacement_to_fluid!(interface, u_current)
        traction_fluid, fluid_state = fluid_solver(interface.displacement)
        length(traction_fluid) == length(interface.fluid_face_indices) || error(
            "fluid_solver must return one traction per fluid interface face",
        )

        # 2. Fluid traction → solid-side buffer → structure solve.
        interpolate_traction_to_structure!(interface, traction_fluid)
        u_tilde, solid_state = solid_solver(interface.traction)
        length(u_tilde) == n_solid || error(
            "solid_solver must return one displacement per solid interface face",
        )

        # 3. Aitken residual δ = ũ − u.
        delta = Vector{T}(undef, Dim * n_solid)
        @inbounds for i in 1:n_solid
            for c in 1:Dim
                delta[(i - 1) * Dim + c] = u_tilde[i][c] - u_current[i][c]
            end
        end
        delta_norm = norm(delta)
        push!(residual_history, delta_norm)

        # 4. Update ω.
        omega = update_aitken_omega!(relaxation, delta)
        push!(omega_history, omega)

        # 5. u ← u + ω · δ.
        u_next = similar(u_current)
        @inbounds for i in 1:n_solid
            step = SVector{Dim, T}(
                ntuple(c -> omega * delta[(i - 1) * Dim + c], Dim),
            )
            u_next[i] = u_current[i] + step
        end

        update_norm = abs(omega) * delta_norm
        if verbose
            @info "FSI outer iter" k δ = delta_norm ω = omega Δu = update_norm
        end

        u_current = u_next

        if update_norm < tol_T
            converged = true
            # Refresh interface displacement with converged iterate.
            interpolate_displacement_to_fluid!(interface, u_current)
            return PartitionedFSIResult{Dim, T}(
                u_current, copy(interface.traction),
                residual_history, omega_history,
                k, converged, fluid_state, solid_state,
            )
        end
    end

    # Non-converged exit: mirror the final iterate into the interface.
    interpolate_displacement_to_fluid!(interface, u_current)
    return PartitionedFSIResult{Dim, T}(
        u_current, copy(interface.traction),
        residual_history, omega_history,
        max_outer, converged, fluid_state, solid_state,
    )
end

# ============================================================
# Method of Lines (MOL) Semi-discretization
# ============================================================
#
# The spatial FVM discretization converts the hyperbolic PDE
#   ∂U/∂t + ∇·F(U) = 0
# into a system of ODEs
#   dU/dt = -R(U)
# where R is the finite-volume residual (flux differences + source terms).
#
# This file constructs the SciML ODEProblem from that semi-discrete form,
# pairing the RHS function with a pre-allocated cache and CFL callback.

# ============================================================
# Standard Hyperbolic (1D)
# ============================================================

"""
    SciMLBase.ODEProblem(prob::HyperbolicProblem; backend=CPUBackend(), kwargs...)

Create an `ODEProblem` from a 1D hyperbolic problem.

The ODE state is a flat `Vector{FT}` of length `nc * N`.
Ghost cells are managed in the cache (stored as `p`).
A CFL callback controls the timestep.

# Example
```julia
ode_prob = ODEProblem(prob)
sol = solve(ode_prob, SSPRK33(); adaptive = false, dt = 1e-3)
```
"""
function SciMLBase.ODEProblem(
        prob::HyperbolicProblem;
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    cache = build_cache(prob, backend)
    u0 = initial_state_flat(prob, cache)
    tspan = (prob.initial_time, prob.final_time)

    function rhs!(du, u, p, t)
        unfold_to_padded!(p, u)
        hyperbolic_rhs!(p.padded_dU, p.padded_U, p.prob, t)
        return fold_from_padded!(du, p)
    end

    sys = fvm_symbolic_index(prob)
    f = ODEFunction{true, SciMLBase.AutoSpecialize}(rhs!; sys)
    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return ODEProblem{true}(f, u0, tspan, cache; callback = cb, kwargs...)
end

# ============================================================
# Standard Hyperbolic (2D)
# ============================================================

"""
    SciMLBase.ODEProblem(prob::HyperbolicProblem2D; backend=CPUBackend(), kwargs...)

Create an `ODEProblem` from a 2D hyperbolic problem.

The ODE state is a flat `Vector{FT}` of length `nx * ny * N`.
"""
function SciMLBase.ODEProblem(
        prob::HyperbolicProblem2D;
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    cache = build_cache(prob, backend)
    u0 = initial_state_flat(prob, cache)
    tspan = (prob.initial_time, prob.final_time)

    function rhs!(du, u, p, t)
        unfold_to_padded!(p, u)
        hyperbolic_rhs_2d!(p.padded_dU, p.padded_U, p.prob, t)
        return fold_from_padded!(du, p)
    end

    sys = fvm_symbolic_index(prob)
    f = ODEFunction{true, SciMLBase.AutoSpecialize}(rhs!; sys)
    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return ODEProblem{true}(f, u0, tspan, cache; callback = cb, kwargs...)
end

# ============================================================
# Standard Hyperbolic (3D)
# ============================================================

"""
    SciMLBase.ODEProblem(prob::HyperbolicProblem3D; backend=CPUBackend(), kwargs...)

Create an `ODEProblem` from a 3D hyperbolic problem.
"""
function SciMLBase.ODEProblem(
        prob::HyperbolicProblem3D;
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    cache = build_cache(prob, backend)
    u0 = initial_state_flat(prob, cache)
    tspan = (prob.initial_time, prob.final_time)

    function rhs!(du, u, p, t)
        unfold_to_padded!(p, u)
        hyperbolic_rhs_3d!(p.padded_dU, p.padded_U, p.prob, t)
        return fold_from_padded!(du, p)
    end

    sys = fvm_symbolic_index(prob)
    f = ODEFunction{true, SciMLBase.AutoSpecialize}(rhs!; sys)
    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return ODEProblem{true}(f, u0, tspan, cache; callback = cb, kwargs...)
end

# ============================================================
# Unstructured Hyperbolic
# ============================================================

"""
    SciMLBase.ODEProblem(prob::UnstructuredHyperbolicProblem; backend=CPUBackend(), kwargs...)

Create an `ODEProblem` from an unstructured hyperbolic problem.
"""
function SciMLBase.ODEProblem(
        prob::UnstructuredHyperbolicProblem;
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    cache = build_cache(prob, backend)
    u0 = initial_state_flat(prob, cache)
    tspan = (prob.initial_time, prob.final_time)

    function rhs!(du, u, p, t)
        unfold_to_padded!(p, u)
        unstructured_rhs!(p.dU, p.U, p.prob, t)
        return fold_from_padded!(du, p)
    end

    sys = fvm_symbolic_index(prob)
    f = ODEFunction{true, SciMLBase.AutoSpecialize}(rhs!; sys)
    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return ODEProblem{true}(f, u0, tspan, cache; callback = cb, kwargs...)
end

# ============================================================
# MHD/CT via Augmented State (2D IdealMHD)
# ============================================================

"""
    SciMLBase.ODEProblem(prob::HyperbolicProblem2D{<:IdealMHDEquations{2}};
                         vector_potential=nothing, backend=CPUBackend(), kwargs...)

Create an `ODEProblem` for 2D MHD with constrained transport.

The ODE state is augmented: `[cell_conserved | Bx_face | By_face]`.
Face-centered B evolves via Faraday's law (dB/dt = curl(EMF)).

For physical accuracy, solve with a stage limiter that enforces
cell-B = avg(face-B) after each RK stage:

```julia
ode_prob = ODEProblem(prob)
limiter = mhd_stage_limiter(ode_prob.p)
sol = solve(ode_prob, SSPRK33(; stage_limiter! = limiter); adaptive = false, dt = dt0)
```
"""
function SciMLBase.ODEProblem(
        prob::HyperbolicProblem2D{<:IdealMHDEquations{2}};
        vector_potential = nothing,
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    cache = build_mhd_ct_cache(prob, backend)
    u0 = initial_mhd_augmented_state(prob, cache; vector_potential = vector_potential)
    tspan = (prob.initial_time, prob.final_time)

    function rhs!(du, u, p, t)
        unfold_mhd_augmented!(p, u)
        _mhd_compute_fluxes_2d!(p.Fx_all, p.Fy_all, p.padded_dU, p.padded_U, p.prob, t)
        _compute_emf_from_extended!(p.emf_z, p.Fx_all, p.Fy_all, p.nx, p.ny)
        return fold_mhd_augmented!(du, p)
    end

    sys = _mhd_ct_2d_symbolic_index(prob)
    f = ODEFunction{true, SciMLBase.AutoSpecialize}(rhs!; sys)
    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return ODEProblem{true}(f, u0, tspan, cache; callback = cb, kwargs...)
end

# ============================================================
# SRMHD/CT via Augmented State (2D)
# ============================================================

"""
    SciMLBase.ODEProblem(prob::HyperbolicProblem2D{<:SRMHDEquations{2}};
                         vector_potential=nothing, backend=CPUBackend(), kwargs...)

Create an `ODEProblem` for 2D SRMHD with constrained transport.
Same augmented-state approach as IdealMHD.
"""
function SciMLBase.ODEProblem(
        prob::HyperbolicProblem2D{<:SRMHDEquations{2}};
        vector_potential = nothing,
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    cache = build_mhd_ct_cache(prob, backend)
    u0 = initial_mhd_augmented_state(prob, cache; vector_potential = vector_potential)
    tspan = (prob.initial_time, prob.final_time)

    function rhs!(du, u, p, t)
        unfold_mhd_augmented!(p, u)
        _mhd_compute_fluxes_2d!(p.Fx_all, p.Fy_all, p.padded_dU, p.padded_U, p.prob, t)
        _compute_emf_from_extended!(p.emf_z, p.Fx_all, p.Fy_all, p.nx, p.ny)
        return fold_mhd_augmented!(du, p)
    end

    sys = _mhd_ct_2d_symbolic_index(prob)
    f = ODEFunction{true, SciMLBase.AutoSpecialize}(rhs!; sys)
    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return ODEProblem{true}(f, u0, tspan, cache; callback = cb, kwargs...)
end

# ============================================================
# GRMHD/CT via Augmented State (2D)
# ============================================================

"""
    SciMLBase.ODEProblem(prob::HyperbolicProblem2D{<:GRMHDEquations{2}};
                         vector_potential=nothing, backend=CPUBackend(), kwargs...)

Create an `ODEProblem` for 2D GRMHD with constrained transport
and geometric source terms.
"""
function SciMLBase.ODEProblem(
        prob::HyperbolicProblem2D{<:GRMHDEquations{2}};
        vector_potential = nothing,
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    cache = build_grmhd_ct_cache(prob, backend)
    u0 = initial_mhd_augmented_state(prob, cache; vector_potential = vector_potential)
    tspan = (prob.initial_time, prob.final_time)

    function rhs!(du, u, p, t)
        unfold_mhd_augmented!(p, u)
        _grmhd_compute_fluxes_2d!(
            p.Fx_all, p.Fy_all, p.padded_dU, p.padded_U,
            p.prob, t, p.metric_data, p.face_data
        )
        _grmhd_add_source_terms!(p.padded_dU, p.padded_U, p.prob.law, p.metric_data, p.prob.mesh, p.nx, p.ny)
        _compute_emf_from_extended!(p.emf_z, p.Fx_all, p.Fy_all, p.nx, p.ny)
        return fold_mhd_augmented!(du, p)
    end

    sys = _mhd_ct_2d_symbolic_index(prob)
    f = ODEFunction{true, SciMLBase.AutoSpecialize}(rhs!; sys)
    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return ODEProblem{true}(f, u0, tspan, cache; callback = cb, kwargs...)
end

"""
    SciMLBase.ODEProblem(prob::HyperbolicProblem3D{<:IdealMHDEquations{3}};
                         vector_potential_x=nothing, vector_potential_y=nothing,
                         vector_potential_z=nothing, backend=CPUBackend(), kwargs...)

Create an augmented-state `ODEProblem` for 3D MHD with constrained transport.
"""
function SciMLBase.ODEProblem(
        prob::HyperbolicProblem3D{<:IdealMHDEquations{3}};
        vector_potential_x = nothing,
        vector_potential_y = nothing,
        vector_potential_z = nothing,
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    cache = build_mhd_ct_cache(prob, backend)
    u0 = initial_mhd_augmented_state(
        prob, cache;
        vector_potential_x = vector_potential_x,
        vector_potential_y = vector_potential_y,
        vector_potential_z = vector_potential_z,
    )
    tspan = (prob.initial_time, prob.final_time)

    function rhs!(du, u, p, t)
        unfold_mhd_augmented!(p, u)
        _mhd_compute_fluxes_3d!(p.Fx_all, p.Fy_all, p.Fz_all, p.padded_dU, p.padded_U, p.prob, t)
        _compute_emf_3d_from_extended!(p.ct, p.Fx_all, p.Fy_all, p.Fz_all, p.nx, p.ny, p.nz)
        return fold_mhd_augmented!(du, p)
    end

    sys = _mhd_ct_3d_symbolic_index(prob)
    f = ODEFunction{true, SciMLBase.AutoSpecialize}(rhs!; sys)
    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return ODEProblem{true}(f, u0, tspan, cache; callback = cb, kwargs...)
end

"""
    mhd_stage_limiter(cache) -> Function

Create a stage limiter function for MHD/CT constrained transport.

The returned function enforces cell-centered B = avg(face-centered B)
after each RK stage. Pass it to the ODE algorithm:

```julia
ode_prob = ODEProblem(prob)
limiter = mhd_stage_limiter(ode_prob.p)
sol = solve(ode_prob, SSPRK33(; stage_limiter! = limiter); adaptive = false, dt = dt0)
```
"""
function mhd_stage_limiter(cache)
    return (u, integrator, p, t) -> _sync_cell_B_from_faces!(u, p)
end

# ============================================================
# AMR via Segmented ODEProblem
# ============================================================

"""
    SciMLBase.ODEProblem(prob::AMRProblem; kwargs...)

Create an `ODEProblem` for an AMR problem.

All active block interiors are flattened into a single state vector.
Uses finest-level CFL as the global dt (no subcycling).
For subcycled AMR, use `solve_amr` or `solve_amr_subcycled` directly.
"""
function SciMLBase.ODEProblem(prob::AMRProblem; callback = nothing, kwargs...)
    cache = build_amr_cache(prob)
    u0 = flatten_amr_state(cache)
    tspan = (prob.initial_time, prob.final_time)

    function rhs!(du, u, p, t)
        unfold_amr!(p, u)
        # Zero dU for all blocks
        for bid in p.block_ids
            du_pad = p.per_block_dU[bid]
            block = p.grid.blocks[bid]
            nx, ny = block.dims[1], block.dims[2]
            N_var = size(du_pad, 1) > 4 ? size(p.per_block_padded[bid][3, 3], 1) : 0
            zero_state = zero(eltype(du_pad))
            for j in axes(du_pad, 2), i in axes(du_pad, 1)
                du_pad[i, j] = zero_state
            end
        end
        # Compute RHS per block
        for bid in p.block_ids
            _advance_block_rhs!(
                p.per_block_dU[bid], p.per_block_padded[bid],
                p.grid.blocks[bid], p.law_ref, p.riemann_solver_ref
            )
        end
        return fold_amr!(du, p)
    end

    sys = _amr_symbolic_index(prob)
    f = ODEFunction{true}(rhs!; sys)
    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return ODEProblem{true}(f, u0, tspan, cache; callback = cb, kwargs...)
end

"""
    _advance_block_rhs!(dU_pad, U_pad, block, law, solver)

Compute the RHS for a single AMR block (2D). Uses first-order reconstruction
(piecewise constant) for simplicity, matching `_advance_block!` from amr_solve.jl.
"""
function _advance_block_rhs!(dU_pad, U_pad, block, law, solver)
    nx, ny = block.dims[1], block.dims[2]
    dx_val, dy_val = block.dx[1], block.dx[2]

    # X-sweeps
    for iy in 1:ny
        jj = iy + 2
        for ix in 0:nx
            iL = ix + 2
            iR = ix + 3
            wL = conserved_to_primitive(law, U_pad[iL, jj])
            wR = conserved_to_primitive(law, U_pad[iR, jj])
            F = solve_riemann(solver, law, wL, wR, 1)
            if ix >= 1
                dU_pad[iL, jj] = dU_pad[iL, jj] - F / dx_val
            end
            if ix < nx
                dU_pad[iR, jj] = dU_pad[iR, jj] + F / dx_val
            end
        end
    end

    # Y-sweeps
    for ix in 1:nx
        ii = ix + 2
        for iy in 0:ny
            jL = iy + 2
            jR = iy + 3
            wL = conserved_to_primitive(law, U_pad[ii, jL])
            wR = conserved_to_primitive(law, U_pad[ii, jR])
            F = solve_riemann(solver, law, wL, wR, 2)
            if iy >= 1
                dU_pad[ii, jL] = dU_pad[ii, jL] - F / dy_val
            end
            if iy < ny
                dU_pad[ii, jR] = dU_pad[ii, jR] + F / dy_val
            end
        end
    end

    return nothing
end

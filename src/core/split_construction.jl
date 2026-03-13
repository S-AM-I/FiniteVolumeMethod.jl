# ============================================================
# SplitODEProblem for IMEX Time Integration
# ============================================================
#
# Wraps hyperbolic problems with stiff sources into a
# SplitODEProblem{true}(f1, f2, u0, tspan, p) where:
#   f1: explicit hyperbolic fluxes
#   f2: implicit stiff source terms
#
# Users solve with SciML IMEX schemes (e.g. KenCarp4).

"""
    SciMLBase.SplitODEProblem(prob::HyperbolicProblem, stiff_source::AbstractStiffSource;
                              backend=CPUBackend(), kwargs...)

Create a `SplitODEProblem` for 1D IMEX time integration.

`f1` (explicit): hyperbolic flux divergence.
`f2` (implicit): stiff source term evaluated cell-by-cell.

# Example
```julia
split_prob = SplitODEProblem(prob, stiff_source)
sol = solve(split_prob, KenCarp4(); adaptive = false, dt = 1e-4)
```
"""
function SciMLBase.SplitODEProblem(
        prob::HyperbolicProblem, stiff_source::AbstractStiffSource;
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    _cpu_backend_only("SplitODEProblem(::HyperbolicProblem, ...)", backend)
    cache = build_cache(prob, backend)
    u0 = initial_state_flat(prob, cache)
    tspan = (prob.initial_time, prob.final_time)

    N = nvariables(prob.law)
    nc = ncells(prob.mesh)
    FT = eltype(u0)

    # f1: explicit hyperbolic fluxes
    function f1!(du, u, p, t)
        unfold_to_padded!(p, u)
        hyperbolic_rhs!(p.padded_dU, p.padded_U, p.prob, t)
        return fold_from_padded!(du, p)
    end

    # f2: implicit stiff source
    function f2!(du, u, p, t)
        law = p.prob.law
        u_sv = reinterpret(SVector{N, FT}, u)
        du_sv = reinterpret(SVector{N, FT}, du)
        return @inbounds for i in 1:nc
            w = conserved_to_primitive(law, u_sv[i])
            du_sv[i] = evaluate_stiff_source(stiff_source, law, w, u_sv[i])
        end
    end

    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return SplitODEProblem{true}(f1!, f2!, u0, tspan, cache; callback = cb, kwargs...)
end

"""
    SciMLBase.SplitODEProblem(prob::HyperbolicProblem2D, stiff_source::AbstractStiffSource;
                              backend=CPUBackend(), kwargs...)

Create a `SplitODEProblem` for 2D IMEX time integration.
"""
function SciMLBase.SplitODEProblem(
        prob::HyperbolicProblem2D, stiff_source::AbstractStiffSource;
        backend::AbstractBackend = CPUBackend(),
        callback = nothing,
        kwargs...
    )
    _cpu_backend_only("SplitODEProblem(::HyperbolicProblem2D, ...)", backend)
    cache = build_cache(prob, backend)
    u0 = initial_state_flat(prob, cache)
    tspan = (prob.initial_time, prob.final_time)

    N = nvariables(prob.law)
    nx, ny = prob.mesh.nx, prob.mesh.ny
    FT = eltype(u0)

    function f1!(du, u, p, t)
        unfold_to_padded!(p, u)
        hyperbolic_rhs_2d!(p.padded_dU, p.padded_U, p.prob, t)
        return fold_from_padded!(du, p)
    end

    function f2!(du, u, p, t)
        law = p.prob.law
        u_sv = reinterpret(SVector{N, FT}, u)
        du_sv = reinterpret(SVector{N, FT}, du)
        return @inbounds for iy in 1:ny, ix in 1:nx
            idx = (iy - 1) * nx + ix
            w = conserved_to_primitive(law, u_sv[idx])
            du_sv[idx] = evaluate_stiff_source(stiff_source, law, w, u_sv[idx])
        end
    end

    cfl_cb = cfl_stepsize_callback(cache)
    cb = _merge_problem_callbacks(cfl_cb, callback)
    return SplitODEProblem{true}(f1!, f2!, u0, tspan, cache; callback = cb, kwargs...)
end

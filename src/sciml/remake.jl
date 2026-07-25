# ============================================================
# SciMLBase.remake for all problem types
# ============================================================

# Sentinel type: kwarg not provided (distinct from user passing `nothing`)
struct _Unset end
const _unset = _Unset()
_replace(::_Unset, original) = original
_replace(new, _) = new

"""
    SciMLBase.remake(prob::FVMProblem; kwargs...)

Create a copy of `prob` with specified fields replaced. Fields not provided
are kept from the original problem. The `conditions` field is preserved
as-is (pre-assembled `Conditions` object).

# Keyword Arguments
Any field of [`FVMProblem`](@ref): `mesh`, `conditions`, `flux_function`,
`flux_parameters`, `source_function`, `source_parameters`,
`initial_condition`, `initial_time`, `final_time`.
"""
function SciMLBase.remake(
        prob::FVMProblem;
        mesh = _unset,
        conditions = _unset,
        flux_function = _unset,
        flux_parameters = _unset,
        source_function = _unset,
        source_parameters = _unset,
        initial_condition = _unset,
        initial_time = _unset,
        final_time = _unset,
    )
    return FVMProblem(
        _replace(mesh, prob.mesh),
        _replace(conditions, prob.conditions),
        _replace(flux_function, prob.flux_function),
        _replace(flux_parameters, prob.flux_parameters),
        _replace(source_function, prob.source_function),
        _replace(source_parameters, prob.source_parameters),
        _replace(initial_condition, prob.initial_condition),
        _replace(initial_time, prob.initial_time),
        _replace(final_time, prob.final_time),
    )
end

"""
    SciMLBase.remake(prob::SteadyFVMProblem; problem=nothing)

Create a copy of `prob`, optionally replacing the inner problem.
"""
function SciMLBase.remake(prob::SteadyFVMProblem; problem = _unset)
    return SteadyFVMProblem(_replace(problem, prob.problem))
end

"""
    SciMLBase.remake(prob::FVMSystem; problems=nothing, initial_time=nothing, final_time=nothing)

Create a copy of `prob`, optionally replacing the problems tuple and/or time span.
When `initial_time` or `final_time` are changed, the inner problems are remade
with the new time span before rebuilding the system.
"""
function SciMLBase.remake(
        prob::FVMSystem{N};
        problems = _unset,
        initial_time = _unset,
        final_time = _unset,
    ) where {N}
    ps = _replace(problems, prob.problems)
    t0 = _replace(initial_time, prob.initial_time)
    tf = _replace(final_time, prob.final_time)
    if t0 !== prob.initial_time || tf !== prob.final_time
        ps = ntuple(N) do i
            SciMLBase.remake(ps[i]; initial_time = t0, final_time = tf)
        end
    end
    return FVMSystem(ps...)
end

"""
    SciMLBase.remake(prob::HyperbolicProblem; kwargs...)

Create a copy of `prob` with specified fields replaced.
"""
function SciMLBase.remake(
        prob::HyperbolicProblem;
        law = _unset,
        mesh = _unset,
        riemann_solver = _unset,
        reconstruction = _unset,
        bc_left = _unset,
        bc_right = _unset,
        initial_condition = _unset,
        initial_time = _unset,
        final_time = _unset,
        cfl = _unset,
    )
    return HyperbolicProblem(
        _replace(law, prob.law),
        _replace(mesh, prob.mesh),
        _replace(riemann_solver, prob.riemann_solver),
        _replace(reconstruction, prob.reconstruction),
        _replace(bc_left, prob.bc_left),
        _replace(bc_right, prob.bc_right),
        _replace(initial_condition, prob.initial_condition),
        _replace(initial_time, prob.initial_time),
        _replace(final_time, prob.final_time),
        _replace(cfl, prob.cfl),
    )
end

"""
    SciMLBase.remake(prob::HyperbolicProblem2D; kwargs...)

Create a copy of `prob` with specified fields replaced.
"""
function SciMLBase.remake(
        prob::HyperbolicProblem2D;
        law = _unset,
        mesh = _unset,
        riemann_solver = _unset,
        reconstruction = _unset,
        bc_left = _unset,
        bc_right = _unset,
        bc_bottom = _unset,
        bc_top = _unset,
        initial_condition = _unset,
        initial_time = _unset,
        final_time = _unset,
        cfl = _unset,
    )
    return HyperbolicProblem2D(
        _replace(law, prob.law),
        _replace(mesh, prob.mesh),
        _replace(riemann_solver, prob.riemann_solver),
        _replace(reconstruction, prob.reconstruction),
        _replace(bc_left, prob.bc_left),
        _replace(bc_right, prob.bc_right),
        _replace(bc_bottom, prob.bc_bottom),
        _replace(bc_top, prob.bc_top),
        _replace(initial_condition, prob.initial_condition),
        _replace(initial_time, prob.initial_time),
        _replace(final_time, prob.final_time),
        _replace(cfl, prob.cfl),
    )
end

"""
    SciMLBase.remake(prob::HyperbolicProblem3D; kwargs...)

Create a copy of `prob` with specified fields replaced.
"""
function SciMLBase.remake(
        prob::HyperbolicProblem3D;
        law = _unset,
        mesh = _unset,
        riemann_solver = _unset,
        reconstruction = _unset,
        bc_left = _unset,
        bc_right = _unset,
        bc_bottom = _unset,
        bc_top = _unset,
        bc_front = _unset,
        bc_back = _unset,
        initial_condition = _unset,
        initial_time = _unset,
        final_time = _unset,
        cfl = _unset,
    )
    return HyperbolicProblem3D(
        _replace(law, prob.law),
        _replace(mesh, prob.mesh),
        _replace(riemann_solver, prob.riemann_solver),
        _replace(reconstruction, prob.reconstruction),
        _replace(bc_left, prob.bc_left),
        _replace(bc_right, prob.bc_right),
        _replace(bc_bottom, prob.bc_bottom),
        _replace(bc_top, prob.bc_top),
        _replace(bc_front, prob.bc_front),
        _replace(bc_back, prob.bc_back),
        _replace(initial_condition, prob.initial_condition),
        _replace(initial_time, prob.initial_time),
        _replace(final_time, prob.final_time),
        _replace(cfl, prob.cfl),
    )
end

"""
    SciMLBase.remake(prob::UnstructuredHyperbolicProblem; kwargs...)

Create a copy of `prob` with specified fields replaced.
"""
function SciMLBase.remake(
        prob::UnstructuredHyperbolicProblem;
        law = _unset,
        mesh = _unset,
        riemann_solver = _unset,
        reconstruction = _unset,
        boundary_conditions = _unset,
        default_bc = _unset,
        initial_condition = _unset,
        initial_time = _unset,
        final_time = _unset,
        cfl = _unset,
    )
    return UnstructuredHyperbolicProblem(
        _replace(law, prob.law),
        _replace(mesh, prob.mesh),
        _replace(riemann_solver, prob.riemann_solver),
        _replace(reconstruction, prob.reconstruction),
        _replace(boundary_conditions, prob.boundary_conditions),
        _replace(default_bc, prob.default_bc),
        _replace(initial_condition, prob.initial_condition),
        _replace(initial_time, prob.initial_time),
        _replace(final_time, prob.final_time),
        _replace(cfl, prob.cfl),
    )
end

"""
    SciMLBase.remake(prob::AMRProblem; kwargs...)

Create a copy of `prob` with specified fields replaced.
"""
function SciMLBase.remake(
        prob::AMRProblem;
        grid = _unset,
        riemann_solver = _unset,
        reconstruction = _unset,
        boundary_conditions = _unset,
        initial_time = _unset,
        final_time = _unset,
        cfl = _unset,
        regrid_interval = _unset,
    )
    return AMRProblem(
        _replace(grid, prob.grid),
        _replace(riemann_solver, prob.riemann_solver),
        _replace(reconstruction, prob.reconstruction),
        _replace(boundary_conditions, prob.boundary_conditions),
        _replace(initial_time, prob.initial_time),
        _replace(final_time, prob.final_time),
        _replace(cfl, prob.cfl),
        _replace(regrid_interval, prob.regrid_interval),
    )
end

# ============================================================
# Semidiscrete ODEProblem remake
# ============================================================
#
# Rebuild the underlying physics problem, then re-create the ODEProblem.
# Accepts keyword args for the physics problem fields (cfl, final_time, etc.)
# as well as `vector_potential` for MHD/CT problems.
#
# Standard ODEProblem kwargs are handled explicitly:
# - `u0` and `tspan` are honored: the rebuilt problem uses them directly.
# - `p` is honored only when it is a semidiscrete cache (the cache-as-parameter
#   design means `p` *is* the discretization); the physics problem inside the
#   provided cache becomes the base for the rebuild. Any other `p` throws an
#   informative `ArgumentError` — it is never silently dropped.
# - `f` cannot be replaced (the RHS closure is generated from the physics
#   problem); passing a different `f` throws.
# NOTE: SciML's `solve` may internally call `remake(prob; u0=..., p=..., tspan=...)`
# with the problem's *own* (possibly type-promoted) values to specialize the
# ODEFunction; those calls are honored by the passthrough logic above.

# Structural ODEProblem kwargs that never map to physics-problem fields
const _ODE_REMAKE_KEYS = (:f, :u0, :p, :tspan, :prob_type, :problem_type, :kwargs, :callback)

function _filter_physics_kwargs(; kwargs...)
    return pairs(NamedTuple(filter(kv -> kv.first ∉ _ODE_REMAKE_KEYS, pairs(kwargs))))
end

_callback_kwarg(kwargs) = hasproperty(kwargs, :callback) ? getproperty(kwargs, :callback) : nothing

function _rebuild_semidiscrete_problem(physics_prob; callback = nothing, kwargs...)
    if callback === nothing
        return ODEProblem(physics_prob; kwargs...)
    end
    return ODEProblem(physics_prob; callback, kwargs...)
end

# `missing` is the SciMLBase remake convention for "not provided"; `nothing`
# means "recompute from defaults", which for us is the rebuilt problem's value.
_ode_override_unset(x) = x === missing || x === nothing

# Resolve the base physics problem for a semidiscrete remake, honoring a
# user-provided `p` when it is a compatible semidiscrete cache.
function _remake_base_physics_prob(ode_prob, p)
    _ode_override_unset(p) && return ode_prob.p.prob
    p === ode_prob.p && return ode_prob.p.prob
    if p isa AbstractSemidiscreteCache && hasproperty(p, :prob)
        return p.prob
    end
    throw(
        ArgumentError(
            "remake: cannot replace `p` of a semidiscrete ODEProblem with a " *
                "$(typeof(p)). The parameter object is a pre-allocated semidiscrete " *
                "cache (cache-as-parameter design); pass a compatible cache, or " *
                "remake physics fields directly (e.g. `remake(ode_prob; cfl = ..., " *
                "law = ...)`), or rebuild via `ODEProblem(physics_prob)`."
        )
    )
end

function _remake_check_f(ode_prob, f)
    _ode_override_unset(f) && return nothing
    f === ode_prob.f && return nothing
    # SciML's solve specialization may pass a re-wrapped ODEFunction (or the
    # raw closure) around the *same* underlying RHS — accept those, since the
    # rebuild regenerates an equivalent RHS from the physics problem anyway.
    _same_underlying_rhs(ode_prob.f, f) && return nothing
    throw(
        ArgumentError(
            "remake: cannot replace `f` of a semidiscrete ODEProblem — the RHS " *
                "closure is generated from the physics problem and its cache. " *
                "Remake physics fields instead (e.g. `remake(ode_prob; law = ...)`)."
        )
    )
end

function _same_underlying_rhs(old, new)
    raw_old = SciMLBase.unwrapped_f(old)
    raw_new = SciMLBase.unwrapped_f(new)
    raw_new === raw_old && return true
    # SplitFunction (SciMLBase 3) carries the RHS pair in f1/f2 with no `f`
    # field, so compare component-wise.
    return hasfield(typeof(old), :f) && raw_new === old.f
end
function _same_underlying_rhs(old::SciMLBase.SplitFunction, new::SciMLBase.SplitFunction)
    return _same_underlying_rhs(old.f1, new.f1) && _same_underlying_rhs(old.f2, new.f2)
end

# Apply u0/tspan overrides to a freshly rebuilt semidiscrete ODEProblem.
function _apply_ode_overrides(new_ode, u0, tspan)
    _ode_override_unset(u0) && _ode_override_unset(tspan) && return new_ode
    new_u0 = _ode_override_unset(u0) ? new_ode.u0 : u0
    if new_u0 !== new_ode.u0
        length(new_u0) == length(new_ode.u0) || throw(
            ArgumentError(
                "remake: `u0` has length $(length(new_u0)) but the semidiscrete " *
                    "state has length $(length(new_ode.u0)) (flat interior-cell " *
                    "vector). Provide a flat state of matching length."
            )
        )
    end
    new_tspan = _ode_override_unset(tspan) ? new_ode.tspan : tspan
    return SciMLBase.ODEProblem{SciMLBase.isinplace(new_ode)}(
        new_ode.f, new_u0, new_tspan, new_ode.p; new_ode.kwargs...
    )
end

# Fast path for pure structural overrides (u0/tspan only, cache unchanged, no
# physics kwargs, callback unchanged): preserve `f` verbatim instead of
# regenerating the RHS. SciML's `solve` calls `remake(prob; u0, p)` internally
# on every solve, and the regeneration path returns a plain `ODEProblem` RHS —
# which would silently drop the stiff half of a `SplitFunction` (the canonical
# IMEX path).
function _remake_structural_fast_path(ode_prob, f, u0, p, tspan, callback, kwargs)
    isempty(kwargs) || return nothing
    (_ode_override_unset(p) || p === ode_prob.p) || return nothing
    callback === _callback_kwarg(ode_prob.kwargs) || return nothing
    _remake_check_f(ode_prob, f)
    new_u0 = _ode_override_unset(u0) ? ode_prob.u0 : u0
    if new_u0 !== ode_prob.u0
        length(new_u0) == length(ode_prob.u0) || throw(
            ArgumentError(
                "remake: `u0` has length $(length(new_u0)) but the semidiscrete " *
                    "state has length $(length(ode_prob.u0)) (flat interior-cell " *
                    "vector). Provide a flat state of matching length."
            )
        )
    end
    new_tspan = _ode_override_unset(tspan) ? ode_prob.tspan : tspan
    return SciMLBase.ODEProblem{SciMLBase.isinplace(ode_prob)}(
        ode_prob.f, new_u0, new_tspan, ode_prob.p; ode_prob.kwargs...
    )
end

# The RHS regeneration below produces a plain ODEProblem; rebuilding a split
# semidiscrete problem that way would silently discard the stiff source.
function _split_rebuild_guard(ode_prob)
    ode_prob.f isa SciMLBase.SplitFunction || return nothing
    throw(
        ArgumentError(
            "remake: cannot rebuild a split semidiscrete ODEProblem with physics " *
                "or callback overrides — remake the physics problem and rebuild " *
                "via `SplitODEProblem(physics_prob, source)` instead."
        )
    )
end

"""
    SciMLBase.remake(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:HyperbolicCache1D}; kwargs...)

Remake a semidiscrete 1D ODEProblem. Accepts any keyword argument valid for
`remake(::HyperbolicProblem; ...)` (e.g. `cfl`, `final_time`, `initial_condition`),
plus the standard ODEProblem kwargs `u0` and `tspan` (used directly by the
rebuilt problem) and `p` (must be a compatible semidiscrete cache).
"""
function SciMLBase.remake(
        ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:HyperbolicCache1D};
        f = missing,
        u0 = missing,
        p = missing,
        tspan = missing,
        callback = _callback_kwarg(ode_prob.kwargs),
        kwargs...
    )
    fast = _remake_structural_fast_path(ode_prob, f, u0, p, tspan, callback, kwargs)
    fast === nothing || return fast
    _split_rebuild_guard(ode_prob)
    _remake_check_f(ode_prob, f)
    base_prob = _remake_base_physics_prob(ode_prob, p)
    physics_kwargs = _filter_physics_kwargs(; kwargs...)
    physics_prob = SciMLBase.remake(base_prob; physics_kwargs...)
    new_ode = _rebuild_semidiscrete_problem(physics_prob; callback)
    return _apply_ode_overrides(new_ode, u0, tspan)
end

"""
    SciMLBase.remake(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:HyperbolicCache2D}; kwargs...)

Remake a semidiscrete 2D ODEProblem. Honors `u0`/`tspan` passthrough; see the
1D method for the `p`/`f` contract.
"""
function SciMLBase.remake(
        ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:HyperbolicCache2D};
        f = missing,
        u0 = missing,
        p = missing,
        tspan = missing,
        callback = _callback_kwarg(ode_prob.kwargs),
        kwargs...
    )
    fast = _remake_structural_fast_path(ode_prob, f, u0, p, tspan, callback, kwargs)
    fast === nothing || return fast
    _split_rebuild_guard(ode_prob)
    _remake_check_f(ode_prob, f)
    base_prob = _remake_base_physics_prob(ode_prob, p)
    physics_kwargs = _filter_physics_kwargs(; kwargs...)
    physics_prob = SciMLBase.remake(base_prob; physics_kwargs...)
    new_ode = _rebuild_semidiscrete_problem(physics_prob; callback)
    return _apply_ode_overrides(new_ode, u0, tspan)
end

"""
    SciMLBase.remake(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:HyperbolicCache3D}; kwargs...)

Remake a semidiscrete 3D ODEProblem. Honors `u0`/`tspan` passthrough; see the
1D method for the `p`/`f` contract.
"""
function SciMLBase.remake(
        ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:HyperbolicCache3D};
        f = missing,
        u0 = missing,
        p = missing,
        tspan = missing,
        callback = _callback_kwarg(ode_prob.kwargs),
        kwargs...
    )
    fast = _remake_structural_fast_path(ode_prob, f, u0, p, tspan, callback, kwargs)
    fast === nothing || return fast
    _split_rebuild_guard(ode_prob)
    _remake_check_f(ode_prob, f)
    base_prob = _remake_base_physics_prob(ode_prob, p)
    physics_kwargs = _filter_physics_kwargs(; kwargs...)
    physics_prob = SciMLBase.remake(base_prob; physics_kwargs...)
    new_ode = _rebuild_semidiscrete_problem(physics_prob; callback)
    return _apply_ode_overrides(new_ode, u0, tspan)
end

"""
    SciMLBase.remake(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:UnstructuredCache}; kwargs...)

Remake a semidiscrete unstructured ODEProblem. Honors `u0`/`tspan` passthrough;
see the 1D method for the `p`/`f` contract.
"""
function SciMLBase.remake(
        ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:UnstructuredCache};
        f = missing,
        u0 = missing,
        p = missing,
        tspan = missing,
        callback = _callback_kwarg(ode_prob.kwargs),
        kwargs...
    )
    fast = _remake_structural_fast_path(ode_prob, f, u0, p, tspan, callback, kwargs)
    fast === nothing || return fast
    _split_rebuild_guard(ode_prob)
    _remake_check_f(ode_prob, f)
    base_prob = _remake_base_physics_prob(ode_prob, p)
    physics_kwargs = _filter_physics_kwargs(; kwargs...)
    physics_prob = SciMLBase.remake(base_prob; physics_kwargs...)
    new_ode = _rebuild_semidiscrete_problem(physics_prob; callback)
    return _apply_ode_overrides(new_ode, u0, tspan)
end

"""
    SciMLBase.remake(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:MHDCTCache2D};
                     vector_potential=nothing, kwargs...)

Remake a semidiscrete MHD/CT ODEProblem.
"""
function SciMLBase.remake(
        ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:MHDCTCache2D};
        vector_potential = nothing,
        f = missing,
        u0 = missing,
        p = missing,
        tspan = missing,
        callback = _callback_kwarg(ode_prob.kwargs),
        kwargs...
    )
    fast = _remake_structural_fast_path(ode_prob, f, u0, p, tspan, callback, kwargs)
    fast === nothing || return fast
    _split_rebuild_guard(ode_prob)
    _remake_check_f(ode_prob, f)
    base_prob = _remake_base_physics_prob(ode_prob, p)
    physics_kwargs = _filter_physics_kwargs(; kwargs...)
    physics_prob = SciMLBase.remake(base_prob; physics_kwargs...)
    new_ode = _rebuild_semidiscrete_problem(physics_prob; callback, vector_potential = vector_potential)
    return _apply_ode_overrides(new_ode, u0, tspan)
end

"""
    SciMLBase.remake(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:GRMHDCTCache2D};
                     vector_potential=nothing, kwargs...)

Remake a semidiscrete GRMHD/CT ODEProblem.
"""
function SciMLBase.remake(
        ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:GRMHDCTCache2D};
        vector_potential = nothing,
        f = missing,
        u0 = missing,
        p = missing,
        tspan = missing,
        callback = _callback_kwarg(ode_prob.kwargs),
        kwargs...
    )
    fast = _remake_structural_fast_path(ode_prob, f, u0, p, tspan, callback, kwargs)
    fast === nothing || return fast
    _split_rebuild_guard(ode_prob)
    _remake_check_f(ode_prob, f)
    base_prob = _remake_base_physics_prob(ode_prob, p)
    physics_kwargs = _filter_physics_kwargs(; kwargs...)
    physics_prob = SciMLBase.remake(base_prob; physics_kwargs...)
    new_ode = _rebuild_semidiscrete_problem(physics_prob; callback, vector_potential = vector_potential)
    return _apply_ode_overrides(new_ode, u0, tspan)
end

"""
    SciMLBase.remake(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:MHDCTCache3D};
                     vector_potential_x=nothing, vector_potential_y=nothing,
                     vector_potential_z=nothing, kwargs...)

Remake a semidiscrete 3D MHD/CT `ODEProblem`.
"""
function SciMLBase.remake(
        ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:MHDCTCache3D};
        vector_potential_x = nothing,
        vector_potential_y = nothing,
        vector_potential_z = nothing,
        f = missing,
        u0 = missing,
        p = missing,
        tspan = missing,
        callback = _callback_kwarg(ode_prob.kwargs),
        kwargs...
    )
    fast = _remake_structural_fast_path(ode_prob, f, u0, p, tspan, callback, kwargs)
    fast === nothing || return fast
    _split_rebuild_guard(ode_prob)
    _remake_check_f(ode_prob, f)
    base_prob = _remake_base_physics_prob(ode_prob, p)
    physics_kwargs = _filter_physics_kwargs(; kwargs...)
    physics_prob = SciMLBase.remake(base_prob; physics_kwargs...)
    new_ode = _rebuild_semidiscrete_problem(
        physics_prob;
        callback,
        vector_potential_x = vector_potential_x,
        vector_potential_y = vector_potential_y,
        vector_potential_z = vector_potential_z,
    )
    return _apply_ode_overrides(new_ode, u0, tspan)
end

# ── Incompressible problem remake ────────────────────────────────────

"""
    SciMLBase.remake(prob::AnyIncompressibleProblem; kwargs...)

Create a copy of `prob` with specified fields replaced. The steady-vs-transient
concrete type ([`SteadyIncompressibleProblem`](@ref) or
[`IncompressibleProblem`](@ref)) is preserved.
"""
function SciMLBase.remake(
        prob::AnyIncompressibleProblem{Dim, T};
        mesh = _unset,
        bcs = _unset,
        algorithm = _unset,
        nu = _unset,
        density = _unset,
        model = _unset,
    ) where {Dim, T}
    new_mesh = _replace(mesh, prob.mesh)
    new_bcs = _replace(bcs, prob.bcs)
    new_algo = _replace(algorithm, prob.algorithm)
    new_nu = nu === _unset ? prob.nu : T(nu)
    new_density = density === _unset ? prob.density : T(density)
    new_model = _replace(model, prob.model)
    return _rebuild_incompressible(
        prob, new_mesh, new_bcs, new_algo, new_nu, new_density, new_model,
    )
end

# Rebuild the same concrete problem type via its fully-parameterised inner
# constructor (which, unlike the public constructors, accepts any algorithm —
# `remake(prob; algorithm = ...)` may swap the coupling).
function _rebuild_incompressible(
        ::IncompressibleProblem{Dim, T}, mesh, bcs, algo, nu, density, model,
    ) where {Dim, T}
    return IncompressibleProblem{
        Dim, T, typeof(mesh), typeof(bcs), typeof(algo), typeof(model),
    }(mesh, bcs, algo, nu, density, model)
end

function _rebuild_incompressible(
        ::SteadyIncompressibleProblem{Dim, T}, mesh, bcs, algo, nu, density, model,
    ) where {Dim, T}
    return SteadyIncompressibleProblem{
        Dim, T, typeof(mesh), typeof(bcs), typeof(algo), typeof(model),
    }(mesh, bcs, algo, nu, density, model)
end

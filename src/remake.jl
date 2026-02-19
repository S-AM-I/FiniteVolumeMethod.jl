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

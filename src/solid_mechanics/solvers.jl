# solid_mechanics/solvers.jl — Unified dispatch between small- and
# large-deformation solid mechanics solvers.
#
# Users instantiate a `SolidMechanicsAlgorithm` (either
# `SmallStrainElasticity` or `UpdatedLagrangian`) and call
# `solve_solid_mechanics(problem, algorithm; kwargs...)`; the dispatcher
# routes to `solve_linear_elasticity` or `solve_finite_strain`.

"""
    SolidMechanicsAlgorithm

Abstract supertype for solid-mechanics solver algorithms.
"""
abstract type SolidMechanicsAlgorithm end

"""
    SmallStrainElasticity <: SolidMechanicsAlgorithm

Small-deformation isotropic linear-elasticity algorithm marker. Routes
to [`solve_linear_elasticity`](@ref).
"""
struct SmallStrainElasticity <: SolidMechanicsAlgorithm end

"""
    UpdatedLagrangian <: SolidMechanicsAlgorithm

Updated-Lagrangian finite-strain algorithm marker. Routes to
[`solve_finite_strain`](@ref).
"""
struct UpdatedLagrangian <: SolidMechanicsAlgorithm end

"""
    solve_solid_mechanics(problem, algorithm; kwargs...) -> result

Dispatch helper that forwards `problem.mesh`, `problem.material`,
`problem.displacement_bcs`, and `problem.body_force` to the
corresponding solver based on `algorithm`.

# Examples
```julia
problem = SolidDisplacementProblem(mesh, IsotropicElastic(; E = 1e6, nu = 0.3))
result = solve_solid_mechanics(problem, SmallStrainElasticity())
large  = solve_solid_mechanics(problem, UpdatedLagrangian(); max_outer = 20)
```
"""
function _check_no_traction_bcs(problem::SolidDisplacementProblem)
    isempty(problem.traction_bcs) && return nothing
    throw(
        ArgumentError(
            "solve_solid_mechanics: traction_bcs are not supported by the current " *
                "solvers (patches $(collect(keys(problem.traction_bcs)))). They were " *
                "previously ignored silently; only displacement_bcs and body_force " *
                "are applied. Remove traction_bcs or impose the load differently.",
        ),
    )
end

function solve_solid_mechanics(
        problem::SolidDisplacementProblem{Dim, T},
        ::SmallStrainElasticity;
        kwargs...,
    ) where {Dim, T}
    _check_no_traction_bcs(problem)
    return solve_linear_elasticity(
        problem.mesh, problem.material,
        problem.displacement_bcs, problem.body_force;
        kwargs...,
    )
end

function solve_solid_mechanics(
        problem::SolidDisplacementProblem{Dim, T},
        ::UpdatedLagrangian;
        kwargs...,
    ) where {Dim, T}
    _check_no_traction_bcs(problem)
    return solve_finite_strain(
        problem.mesh, problem.material,
        problem.displacement_bcs, problem.body_force;
        kwargs...,
    )
end

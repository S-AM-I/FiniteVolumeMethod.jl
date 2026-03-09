# Parabolic Core Types - Migrated from Simu.jl SimuCore
# These types provide a foundation for parabolic PDE solvers and structured mesh support.
# Names are adjusted to avoid collisions with existing FVM.jl types:
#   AbstractMesh -> AbstractParabolicMesh
#   AbstractOperator -> AbstractPhysicsOperator
#   Dirichlet/Neumann/Robin -> ParabolicDirichlet/ParabolicNeumann/ParabolicRobin

# --- Tagging System ---

"""
    AbstractTag

Base type for tags used in multiple dispatch.
"""
abstract type AbstractTag end

"""
    AbstractTagSpatial <: AbstractTag

Tag for spatial model characteristics.
"""
abstract type AbstractTagSpatial <: AbstractTag end

"""
    AbstractTagTime <: AbstractTag

Tag for temporal model characteristics.
"""
abstract type AbstractTagTime <: AbstractTag end

"""
    AbstractTagSteady <: AbstractTag

Tag for steady-state models.
"""
abstract type AbstractTagSteady <: AbstractTag end

"""
    AbstractTagIVP <: AbstractTag

Tag for initial value problems.
"""
abstract type AbstractTagIVP <: AbstractTag end

# --- Simulation, Problem, and Solution Types ---

"""
    AbstractSimulation

Base type for all simulation objects.
"""
abstract type AbstractSimulation end

"""
    AbstractProblem

Base type for all problem definitions.
"""
abstract type AbstractProblem end

"""
    AbstractSolution

Base type for all solution containers.
"""
abstract type AbstractSolution end

"""
    AbstractProblemIVP <: AbstractProblem

Initial Value Problem.
"""
abstract type AbstractProblemIVP <: AbstractProblem end

"""
    AbstractProblemSteady <: AbstractProblem

Steady-State Problem.
"""
abstract type AbstractProblemSteady <: AbstractProblem end

"""
    AbstractProblemPDE <: AbstractProblemIVP

Partial Differential Equation Problem.
"""
abstract type AbstractProblemPDE <: AbstractProblemIVP end

# --- Geometry and Mesh Abstract Types ---

"""
    AbstractGeometry

Base type for all geometric representations in the simulation.
"""
abstract type AbstractGeometry end

"""
    AbstractGeometryComponent

Base type for components of a geometry (e.g., nodes, cells, faces).
"""
abstract type AbstractGeometryComponent end

"""
    AbstractParabolicMesh <: AbstractGeometryComponent

Abstract representation of a computational mesh for parabolic/elliptic solvers.
Renamed from Simu.jl's `AbstractMesh` to avoid collision with FVM.jl's
`AbstractMesh{Dim}` used by the hyperbolic solver framework.
"""
abstract type AbstractParabolicMesh <: AbstractGeometryComponent end

"""
    AbstractNode <: AbstractGeometryComponent

Abstract representation of a point in space (node/vertex).
"""
abstract type AbstractNode <: AbstractGeometryComponent end

"""
    AbstractCell <: AbstractGeometryComponent

Abstract representation of a computational cell (element).
"""
abstract type AbstractCell <: AbstractGeometryComponent end

"""
    AbstractFace <: AbstractGeometryComponent

Abstract representation of an interface between cells or a boundary.
"""
abstract type AbstractFace <: AbstractGeometryComponent end

@enum CellType begin
    CT_Tetrahedron
    CT_Hexahedron
    CT_Prism
    CT_Pyramid
    CT_Polyhedron # Generic fallback
end

# --- Physics and Discretization Abstract Types ---

"""
    AbstractField

Abstract type for spatially distributed fields (e.g., Temperature, Pressure).
"""
abstract type AbstractField end

"""
    AbstractPhysicsOperator

Abstract type for mathematical operators (e.g., Diffusion, Advection).
Renamed from Simu.jl's `AbstractOperator` to avoid collision with
FVM.jl's `AbstractOperator` used in coupling/operators.jl.
"""
abstract type AbstractPhysicsOperator end

"""
    AbstractBoundaryCondition

Abstract type for boundary condition specifications.
"""
abstract type AbstractBoundaryCondition end

"""
    AbstractInitialCondition

Abstract type for initial state specifications.
"""
abstract type AbstractInitialCondition end

"""
    ParabolicDirichlet <: AbstractBoundaryCondition

Fixed value boundary condition.
Renamed from Simu.jl's `Dirichlet` to avoid collision with FVM.jl's
`Dirichlet` enum member in `ConditionType`.
"""
struct ParabolicDirichlet <: AbstractBoundaryCondition
    value::Float64
end

"""
    ParabolicNeumann <: AbstractBoundaryCondition

Fixed flux (gradient) boundary condition.
Renamed from Simu.jl's `Neumann` to avoid collision with FVM.jl's
`Neumann` enum member in `ConditionType`.
"""
struct ParabolicNeumann <: AbstractBoundaryCondition
    value::Float64 # This represents the flux
end

"""
    ParabolicRobin <: AbstractBoundaryCondition

Mixed boundary condition: a*phi + b*flux = c.
Renamed from Simu.jl's `Robin` to avoid collision with FVM.jl's
`Robin` enum member in `ConditionType`.
"""
struct ParabolicRobin <: AbstractBoundaryCondition
    a::Float64
    b::Float64
    c::Float64
end

# --- Variable and Field Types ---

"""
    AbstractVariable

Abstract type for variables in the simulation.
"""
abstract type AbstractVariable end

"""
    VariableRole <: AbstractVariable

Categorization of variables (e.g., State, Property).
"""
abstract type VariableRole <: AbstractVariable end

"""
    STATEVAR

Singleton role tag for state variables.
"""
struct _StateVar <: VariableRole end
const STATEVAR = _StateVar()

"""
    Variable(name, role, unit, description)

Lightweight variable metadata used across physics models.
"""
struct Variable{R, U} <: AbstractVariable
    name::Symbol
    role::R
    unit::Symbol
    description::U
end

"""
    CellField(variable, values)

Simple cell-centered field container.
"""
struct CellField{V, T} <: AbstractField
    variable::V
    values::Vector{T}
end

"""
    SimulationState(t; fields)

State wrapper holding the current time and field data.
"""
struct SimulationState{T, F}
    t::T
    fields::F
    function SimulationState(t::T; fields::F) where {T, F}
        return new{T, F}(t, fields)
    end
end

"""
    validate_state(state)

Ensures no empty fields are present.
"""
function validate_state(state::SimulationState)
    return state
end

"""
    update_field(state, field)

Returns a new state with the given field replaced by name.
"""
function update_field(state::SimulationState, field::CellField)
    new_fields = copy(state.fields)
    new_fields[field.variable.name] = field
    return SimulationState(state.t; fields = new_fields)
end

# --- Material Types ---

"""
    AbstractMaterialProperty

Abstract base for material properties (e.g., density, viscosity).
"""
abstract type AbstractMaterialProperty end

"""
    AbstractMaterial

Abstract representation of a material composed of properties and models.
"""
abstract type AbstractMaterial end

"""
    AbstractMaterialModel

Abstract type for property correlations or constitutive equations.
"""
abstract type AbstractMaterialModel end

# --- Discretization Types ---

"""
    AbstractDiscretization

Abstract type for spatial discretization methods.
"""
abstract type AbstractDiscretization end

"""
    AbstractSemidiscretization <: AbstractDiscretization

Method that discretizes space but leaves time continuous.
"""
abstract type AbstractSemidiscretization <: AbstractDiscretization end

"""
    AbstractFluxCalculator

Logic for computing numerical fluxes across faces.
"""
abstract type AbstractFluxCalculator end

"""
    AbstractReconstruction

Logic for interpolating cell values to faces.
"""
abstract type AbstractReconstruction end

# --- Solver and Algorithm Abstract Types ---

"""
    AbstractAlgorithm

Base type for all numerical algorithms.
"""
abstract type AbstractAlgorithm end

"""
    AbstractTimeIntegrator <: AbstractAlgorithm

Solver for time-stepping (ODEs).
"""
abstract type AbstractTimeIntegrator <: AbstractAlgorithm end

"""
    AbstractNonlinearSolver <: AbstractAlgorithm

Solver for nonlinear algebraic systems (e.g., Newton-Raphson).
"""
abstract type AbstractNonlinearSolver <: AbstractAlgorithm end

"""
    AbstractLinearSolver <: AbstractAlgorithm

Solver for linear algebraic systems Ax = b.
"""
abstract type AbstractLinearSolver <: AbstractAlgorithm end

# --- Callbacks and Diagnostics ---

"""
    AbstractCallback

Abstract base for simulation callbacks (executed at steps).
"""
abstract type AbstractCallback end

"""
    AbstractDiagnostic

Abstract base for physics-based diagnostics.
"""
abstract type AbstractDiagnostic end

# --- Control and Events ---

"""
    AbstractTimeGrid

Abstract representation of a sequence of simulation times.
"""
abstract type AbstractTimeGrid end

"""
    AbstractController

Abstract base for simulation execution controllers.
"""
abstract type AbstractController end

"""
    AbstractEvent

Abstract base for discrete events in simulation flow.
"""
abstract type AbstractEvent end

# --- Post-processing and I/O ---

"""
    AbstractOutputManager

Abstract manager for result output and persistence.
"""
abstract type AbstractOutputManager end

"""
    AbstractConfig

Abstract configuration container.
"""
abstract type AbstractConfig end

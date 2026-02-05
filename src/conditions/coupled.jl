@doc raw"""
    Coupled Multi-Field Boundary Conditions Module

This module provides support for boundary conditions that couple multiple
fields in an `FVMSystem`. This allows boundary conditions where the value
of one field depends on other fields in the system.

## Mathematical Formulation

A general coupled BC takes the form:

```math
\sum_{i=1}^{N} a_i(\vb x, t) \, u_i = c(\vb x, t)
```

where $u_i$ are the different field variables and $a_i$, $c$ are coefficients
that may depend on position and time.

## Common Applications

1. **Electrochemical systems**: Butler-Volmer kinetics coupling concentration
   and potential at electrode surfaces
2. **Reactive flows**: Surface reactions coupling multiple species concentrations
3. **Thermal-mechanical**: Heat flux depending on displacement or stress
4. **Multi-phase flows**: Interface conditions coupling phase fields

## Usage with FVMSystem

Coupled BCs integrate with the existing `FVMSystem` infrastructure. Each
field can have BCs that reference values from other fields in the system.
"""

"""
    CoupledBC{T<:Real}

A boundary condition that couples multiple fields.

# Fields
- `field_indices`: Indices of the fields involved in the coupling
- `coefficients`: Coefficients for each field in the linear combination
- `rhs_value`: Right-hand side value of the constraint

# Mathematical Form
The constraint is: `Σᵢ coefficients[i] * u[field_indices[i]] = rhs_value`

# Example

```julia
# Constraint: 2*u₁ + 3*u₂ = 5 at boundary
coupled_bc = CoupledBC([1, 2], [2.0, 3.0], 5.0)
```
"""
struct CoupledBC{T<:Real}
    field_indices::Vector{Int}
    coefficients::Vector{T}
    rhs_value::T

    function CoupledBC(field_indices::Vector{Int}, coefficients::Vector{T}, rhs_value::T) where {T<:Real}
        @assert length(field_indices) == length(coefficients) "Number of fields must match number of coefficients"
        @assert length(field_indices) > 0 "Must specify at least one field"
        return new{T}(field_indices, coefficients, rhs_value)
    end
end

# Convenience constructors
function CoupledBC(field_indices, coefficients, rhs_value)
    T = promote_type(eltype(coefficients), typeof(rhs_value))
    return CoupledBC(collect(Int, field_indices), collect(T, coefficients), T(rhs_value))
end

"""
    CoupledDirichlet{F,P}

A coupled Dirichlet BC where the boundary value depends on other fields.

# Fields
- `target_field`: Index of the field this BC applies to
- `coupling_function`: Function `f(x, y, t, u_tuple, p) -> value`
- `parameters`: Parameters passed to the coupling function

The `u_tuple` contains values of all fields at the boundary point.

# Example

```julia
# u₁ at boundary depends on u₂: u₁ = exp(-u₂)
coupled_dirichlet = CoupledDirichlet(
    1,  # applies to field 1
    (x, y, t, u, p) -> exp(-u[2]),  # depends on field 2
    nothing
)
```
"""
struct CoupledDirichlet{F<:Function, P}
    target_field::Int
    coupling_function::F
    parameters::P
end

CoupledDirichlet(target_field::Int, f::Function; parameters=nothing) =
    CoupledDirichlet(target_field, f, parameters)

"""
    CoupledNeumann{F,P}

A coupled Neumann BC where the flux depends on other fields.

# Fields
- `target_field`: Index of the field this BC applies to
- `coupling_function`: Function `f(x, y, t, u_tuple, grad_tuple, p) -> flux`
- `parameters`: Parameters passed to the coupling function

# Example

```julia
# Flux of species 1 depends on concentration of species 2 (cross-diffusion)
coupled_neumann = CoupledNeumann(
    1,
    (x, y, t, u, grad, p) -> -p.D12 * grad[2][1],  # flux depends on ∇u₂
    (D12 = 0.1,)
)
```
"""
struct CoupledNeumann{F<:Function, P}
    target_field::Int
    coupling_function::F
    parameters::P
end

CoupledNeumann(target_field::Int, f::Function; parameters=nothing) =
    CoupledNeumann(target_field, f, parameters)

"""
    CoupledRobin{F,P}

A coupled Robin BC where the coefficients depend on other fields.

# Fields
- `target_field`: Index of the field this BC applies to
- `coupling_function`: Function `f(x, y, t, u_tuple, grad_tuple, p) -> (a, b, c)`
- `parameters`: Parameters passed to the coupling function

# Example

```julia
# Robin BC for temperature with convection coefficient depending on velocity field
coupled_robin = CoupledRobin(
    1,  # temperature field
    (x, y, t, u, grad, p) -> begin
        v_mag = sqrt(u[2]^2 + u[3]^2)  # velocity magnitude from fields 2,3
        h = p.h0 * (1 + p.C * v_mag)    # velocity-dependent h
        (h, 1.0, h * p.T_inf)
    end,
    (h0 = 10.0, C = 0.5, T_inf = 300.0)
)
```
"""
struct CoupledRobin{F<:Function, P}
    target_field::Int
    coupling_function::F
    parameters::P
end

CoupledRobin(target_field::Int, f::Function; parameters=nothing) =
    CoupledRobin(target_field, f, parameters)

"""
    CoupledConditionType

Union type for all coupled boundary condition types.
"""
const CoupledConditionType = Union{CoupledBC, CoupledDirichlet, CoupledNeumann, CoupledRobin}

@doc raw"""
    evaluate_coupled_bc(bc::CoupledDirichlet, x, y, t, u_values, grad_values)

Evaluate a coupled Dirichlet BC given the current field values.

# Arguments
- `bc`: The coupled Dirichlet BC
- `x, y`: Boundary point coordinates
- `t`: Current time
- `u_values`: Tuple of solution values for all fields at the boundary
- `grad_values`: Tuple of gradients for all fields (may be unused)

# Returns
The Dirichlet boundary value for the target field.
"""
function evaluate_coupled_bc(bc::CoupledDirichlet, x, y, t, u_values, grad_values=nothing)
    return bc.coupling_function(x, y, t, u_values, bc.parameters)
end

"""
    evaluate_coupled_bc(bc::CoupledNeumann, x, y, t, u_values, grad_values)

Evaluate a coupled Neumann BC given the current field values and gradients.

# Returns
The Neumann flux value for the target field.
"""
function evaluate_coupled_bc(bc::CoupledNeumann, x, y, t, u_values, grad_values)
    return bc.coupling_function(x, y, t, u_values, grad_values, bc.parameters)
end

"""
    evaluate_coupled_bc(bc::CoupledRobin, x, y, t, u_values, grad_values)

Evaluate a coupled Robin BC given the current field values and gradients.

# Returns
Tuple `(a, b, c)` for the Robin condition on the target field.
"""
function evaluate_coupled_bc(bc::CoupledRobin, x, y, t, u_values, grad_values)
    return bc.coupling_function(x, y, t, u_values, grad_values, bc.parameters)
end

"""
    evaluate_coupled_bc(bc::CoupledBC, u_values)

Evaluate a linear coupled constraint.

For the constraint `Σᵢ aᵢ*uᵢ = c`, returns the residual.

# Returns
The constraint residual: `Σᵢ aᵢ*uᵢ - c`
"""
function evaluate_coupled_bc(bc::CoupledBC, u_values)
    result = -bc.rhs_value
    for (idx, coef) in zip(bc.field_indices, bc.coefficients)
        result += coef * u_values[idx]
    end
    return result
end

"""
    get_target_field(bc::CoupledDirichlet) = bc.target_field
    get_target_field(bc::CoupledNeumann) = bc.target_field
    get_target_field(bc::CoupledRobin) = bc.target_field

Get the field index that a coupled BC applies to.
"""
get_target_field(bc::CoupledDirichlet) = bc.target_field
get_target_field(bc::CoupledNeumann) = bc.target_field
get_target_field(bc::CoupledRobin) = bc.target_field

"""
    get_coupled_fields(bc::CoupledBC)

Get the field indices involved in a coupled constraint.
"""
get_coupled_fields(bc::CoupledBC) = bc.field_indices

@doc raw"""
    extract_field_values_at_boundary(prob::FVMSystem, u, boundary_node::Int)

Extract the values of all fields at a boundary node from the system solution.

# Arguments
- `prob`: The FVMSystem
- `u`: Solution matrix (n_fields × n_nodes)
- `boundary_node`: Node index on the boundary

# Returns
Tuple of field values at the boundary node.
"""
function extract_field_values_at_boundary(prob, u, boundary_node::Int)
    n_fields = _neqs(prob)
    return ntuple(i -> u[i, boundary_node], n_fields)
end

@doc raw"""
    extract_field_gradients_at_boundary(prob::FVMSystem, mesh::FVMGeometry, u, i::Int, j::Int)

Extract the gradients of all fields at a boundary edge.

# Arguments
- `prob`: The FVMSystem
- `mesh`: The FVM geometry
- `u`: Solution matrix (n_fields × n_nodes)
- `i, j`: Boundary edge vertex indices

# Returns
Tuple of gradient tuples, one for each field.
"""
function extract_field_gradients_at_boundary(prob, mesh::FVMGeometry, u, i::Int, j::Int)
    n_fields = _neqs(prob)
    return ntuple(n_fields) do field
        # Extract single field solution
        u_field = view(u, field, :)
        compute_boundary_gradient(mesh, u_field, i, j)
    end
end

"""
    CoupledBoundaryConditions{C}

Container for coupled boundary conditions in an FVMSystem.

# Fields
- `conditions`: Dictionary mapping (segment_index, field_index) to coupled BC
"""
struct CoupledBoundaryConditions{C}
    conditions::Dict{Tuple{Int, Int}, C}
end

CoupledBoundaryConditions() = CoupledBoundaryConditions(Dict{Tuple{Int, Int}, CoupledConditionType}())

"""
    add_coupled_bc!(cbc::CoupledBoundaryConditions, segment::Int, bc::CoupledConditionType)

Add a coupled BC to a specific boundary segment for its target field.
"""
function add_coupled_bc!(cbc::CoupledBoundaryConditions, segment::Int, bc)
    field = get_target_field(bc)
    cbc.conditions[(segment, field)] = bc
    return cbc
end

"""
    get_coupled_bc(cbc::CoupledBoundaryConditions, segment::Int, field::Int)

Get the coupled BC for a specific segment and field, or nothing if not defined.
"""
function get_coupled_bc(cbc::CoupledBoundaryConditions, segment::Int, field::Int)
    return get(cbc.conditions, (segment, field), nothing)
end

"""
    has_coupled_bc(cbc::CoupledBoundaryConditions, segment::Int, field::Int)

Check if a coupled BC is defined for the given segment and field.
"""
function has_coupled_bc(cbc::CoupledBoundaryConditions, segment::Int, field::Int)
    return haskey(cbc.conditions, (segment, field))
end

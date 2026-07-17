@doc raw"""
    Nonlinear Boundary Conditions Module

This module provides support for nonlinear boundary conditions in finite volume methods.
Nonlinear BCs are conditions where the boundary value depends nonlinearly on the solution
or its gradient.

The implementation uses linearization at each timestep/iteration, converting the nonlinear
BC to an equivalent linear BC around the current solution state.

## Supported Types

- `NonlinearDirichlet`: Nonlinear constraint on the solution value
- `NonlinearNeumann`: Nonlinear flux condition
- `NonlinearRobin`: Nonlinear mixed condition

## Function Signatures

All nonlinear BC functions follow the FVM.jl convention:

    f(x, y, t, u, ∇u, p) -> value

where:
- `(x, y)` is the boundary point
- `t` is the current time
- `u` is the solution value at the boundary
- `∇u` is a tuple `(∂u/∂x, ∂u/∂y)` - the gradient at the boundary
- `p` contains additional parameters

For `NonlinearRobin`, the function returns a tuple `(a, b, c)` for the condition
`a*u + b*(q·n) = c`.
"""

"""
    NonlinearDirichlet

A nonlinear Dirichlet boundary condition of the form:

    u = f(x, y, t, u, ∇u, p)

where the boundary value depends on the solution itself. This creates an implicit
condition that is solved via linearization.

# Example

```julia
# Boundary condition: u = u^2 + sin(x)  (solved implicitly)
nonlinear_dirichlet = (x, y, t, u, grad_u, p) -> u^2 + sin(x)
```

Note: The condition `u = f(u)` is linearized to find the fixed point.
"""
struct NonlinearDirichlet end

"""
    NonlinearNeumann

A nonlinear Neumann boundary condition of the form:

    q·n = f(x, y, t, u, ∇u, p)

where the flux depends nonlinearly on the solution or its gradient.

# Example

```julia
# Nonlinear radiation BC: q·n = σ*(u^4 - T_amb^4)
radiation_bc = (x, y, t, u, grad_u, p) -> p.σ * (u^4 - p.T_amb^4)
```
"""
struct NonlinearNeumann end

"""
    NonlinearRobin

A nonlinear Robin boundary condition of the form:

    a(x,y,t,u,∇u)*u + b(x,y,t,u,∇u)*(q·n) = c(x,y,t,u,∇u)

where the coefficients can depend nonlinearly on the solution.

The function must return a tuple `(a, b, c)`.

# Example

```julia
# Nonlinear convective BC with temperature-dependent heat transfer
nonlinear_robin = (x, y, t, u, grad_u, p) -> begin
    h = p.h0 * (1 + p.β * abs(u - p.T_ref))  # Temperature-dependent h
    (h, 1.0, h * p.T_inf)
end
```
"""
struct NonlinearRobin end

"""
    NonlinearConditionType

Union type for all nonlinear boundary condition markers.
"""
const NonlinearConditionType = Union{Type{NonlinearDirichlet}, Type{NonlinearNeumann}, Type{NonlinearRobin}}

@doc raw"""
    linearize_bc(::Type{NonlinearDirichlet}, f, x, y, t, u, grad_u, p; ε=1e-6)

Linearize a nonlinear Dirichlet BC around the current solution.

For the condition `u = f(x, y, t, u, ∇u, p)`, we seek a fixed point.
Using Newton's method:

```math
u_{n+1} = u_n - \frac{f(u_n) - u_n}{\partial f/\partial u - 1}
```

For a single linearization step, we return the linearized Dirichlet value.

# Arguments
- `f`: The nonlinear BC function `f(x, y, t, u, ∇u, p)`
- `x, y`: Boundary point coordinates
- `t`: Current time
- `u`: Current solution value at the boundary
- `grad_u`: Tuple `(∂u/∂x, ∂u/∂y)` at the boundary
- `p`: Parameters
- `ε`: Perturbation size for numerical derivative (default: 1e-6)

# Returns
The linearized Dirichlet boundary value.
"""
function linearize_bc(::Type{NonlinearDirichlet}, f, x, y, t, u, grad_u, p; ε = 1.0e-6)
    # Evaluate f at current state
    f_val = f(x, y, t, u, grad_u, p)

    # Compute numerical derivative ∂f/∂u
    f_pert = f(x, y, t, u + ε, grad_u, p)
    df_du = (f_pert - f_val) / ε

    # Newton step for fixed point u = f(u)
    # u_new = u - (f(u) - u) / (df/du - 1)
    denom = df_du - one(df_du)

    if abs(denom) < 1.0e-12
        # Near fixed point or singular - return f(u) as estimate
        return f_val
    end

    u_new = u - (f_val - u) / denom
    return u_new
end

@doc raw"""
    linearize_bc(::Type{NonlinearNeumann}, f, x, y, t, u, grad_u, p; kwargs...)

Linearize a nonlinear Neumann BC by direct evaluation.

For the condition `q·n = f(x, y, t, u, ∇u, p)`, we simply evaluate the function
at the current state to get the linearized flux value.

# Returns
The linearized Neumann flux value `q·n`.
"""
function linearize_bc(::Type{NonlinearNeumann}, f, x, y, t, u, grad_u, p; kwargs...)
    return f(x, y, t, u, grad_u, p)
end

@doc raw"""
    linearize_bc(::Type{NonlinearRobin}, f, x, y, t, u, grad_u, p; kwargs...)

Linearize a nonlinear Robin BC by direct evaluation.

For the condition `a*u + b*(q·n) = c`, we evaluate the function to get
the current coefficients `(a, b, c)`.

# Returns
Tuple `(a, b, c)` for the linearized Robin condition.
"""
function linearize_bc(::Type{NonlinearRobin}, f, x, y, t, u, grad_u, p; kwargs...)
    return f(x, y, t, u, grad_u, p)
end

"""
    NonlinearBoundaryConditions{F,C,P}

Container for nonlinear boundary conditions that wraps the standard BoundaryConditions
with additional metadata for nonlinear handling.

# Fields
- `base_conditions`: The underlying `BoundaryConditions` object
- `nonlinear_types`: Tuple indicating which boundaries have nonlinear conditions
- `gradient_cache`: Pre-allocated storage for gradient computations
"""
struct NonlinearBoundaryConditions{F, C, P, N}
    base_conditions::BoundaryConditions{F, C}
    nonlinear_types::N
    parameters::P
end

"""
    is_nonlinear_bc(bc_type)

Check if a boundary condition type is nonlinear.
"""
is_nonlinear_bc(::Type{NonlinearDirichlet}) = true
is_nonlinear_bc(::Type{NonlinearNeumann}) = true
is_nonlinear_bc(::Type{NonlinearRobin}) = true
is_nonlinear_bc(::Any) = false

"""
    get_base_type(::Type{NonlinearDirichlet}) = Dirichlet
    get_base_type(::Type{NonlinearNeumann}) = Neumann
    get_base_type(::Type{NonlinearRobin}) = Robin

Get the linear base type for a nonlinear BC type.
"""
get_base_type(::Type{NonlinearDirichlet}) = Dirichlet
get_base_type(::Type{NonlinearNeumann}) = Neumann
get_base_type(::Type{NonlinearRobin}) = Robin

@doc raw"""
    compute_boundary_gradient(mesh::FVMGeometry, u, i::Int, j::Int)

Compute the gradient at a boundary edge midpoint using shape functions.

For a boundary edge (i, j), finds the adjacent triangle and uses its
shape function coefficients to compute the gradient.

# Arguments
- `mesh`: The FVM geometry
- `u`: Solution vector
- `i, j`: Boundary edge vertex indices

# Returns
Tuple `(∂u/∂x, ∂u/∂y)` at the edge midpoint.
"""
function compute_boundary_gradient(mesh::FVMGeometry, u, i::Int, j::Int)
    tri = mesh.triangulation

    # Find triangle adjacent to boundary edge
    k = DelaunayTriangulation.get_adjacent(tri, i, j)

    if DelaunayTriangulation.is_ghost_vertex(k)
        # Try other orientation
        k = DelaunayTriangulation.get_adjacent(tri, j, i)
        if DelaunayTriangulation.is_ghost_vertex(k)
            # Fallback: use finite difference along edge
            pᵢ = get_point(mesh, i)
            pⱼ = get_point(mesh, j)
            xᵢ, yᵢ = getxy(pᵢ)
            xⱼ, yⱼ = getxy(pⱼ)
            dx = xⱼ - xᵢ
            dy = yⱼ - yᵢ
            ds = sqrt(dx^2 + dy^2)
            du = u[j] - u[i]
            # Gradient along edge direction
            grad_s = du / ds
            return (grad_s * dx / ds, grad_s * dy / ds)
        end
        i, j = j, i
    end

    # Get triangle properties for gradient computation
    T_ordered, props = _safe_get_triangle_props(mesh, (i, j, k))
    i_T, j_T, k_T = T_ordered

    s = props.shape_function_coefficients
    s₁₁, s₁₂, s₁₃ = s[1], s[2], s[3]
    s₂₁, s₂₂, s₂₃ = s[4], s[5], s[6]

    # Gradient is constant in triangle
    grad_x = s₁₁ * u[i_T] + s₁₂ * u[j_T] + s₁₃ * u[k_T]
    grad_y = s₂₁ * u[i_T] + s₂₂ * u[j_T] + s₂₃ * u[k_T]

    return (grad_x, grad_y)
end

@doc raw"""
    evaluate_nonlinear_bc(bc_type, f, mesh, u, i, j, t, p)

Evaluate a nonlinear boundary condition at a boundary edge.

This function:
1. Computes the solution value at the edge midpoint
2. Computes the gradient at the edge
3. Linearizes the nonlinear BC around the current state

# Arguments
- `bc_type`: The nonlinear BC type (NonlinearDirichlet, etc.)
- `f`: The BC function
- `mesh`: The FVM geometry
- `u`: Current solution vector
- `i, j`: Boundary edge vertex indices
- `t`: Current time
- `p`: BC parameters

# Returns
The linearized BC value (scalar for Dirichlet/Neumann, tuple for Robin).
"""
function evaluate_nonlinear_bc(bc_type::Type{<:NonlinearConditionType}, f, mesh::FVMGeometry, u, i::Int, j::Int, t, p)
    # Get edge midpoint position
    pᵢ = get_point(mesh, i)
    pⱼ = get_point(mesh, j)
    xᵢ, yᵢ = getxy(pᵢ)
    xⱼ, yⱼ = getxy(pⱼ)
    x_mid = (xᵢ + xⱼ) / 2
    y_mid = (yᵢ + yⱼ) / 2

    # Interpolate solution at midpoint
    u_mid = (u[i] + u[j]) / 2

    # Compute gradient at edge
    grad_u = compute_boundary_gradient(mesh, u, i, j)

    # Linearize the BC
    return linearize_bc(bc_type, f, x_mid, y_mid, t, u_mid, grad_u, p)
end

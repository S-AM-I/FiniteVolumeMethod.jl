@doc raw"""
    AdvectionDiffusionEquation <: AbstractFVMTemplate

A struct for defining a problem representing an advection-diffusion equation:
```math
\pdv{u}{t} + \div\left[\vb v(\vb x, t) u\right] = \div\left[D(\vb x, t)\grad u\right] + S(\vb x, t, u)
```
inside a domain $\Omega$, where $\vb v = (v_x, v_y)$ is the velocity field,
$D$ is the diffusion coefficient, and $S$ is an optional source term.

This can also be written in the flux form:
```math
\pdv{u}{t} = -\div\vb q + S(\vb x, t, u)
```
where the total flux is:
```math
\vb q = \vb v u - D \grad u
```

You can solve this problem using [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)).

# Constructor

    AdvectionDiffusionEquation(mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions=InternalConditions();
        diffusion_function,
        diffusion_parameters=nothing,
        velocity_function,
        velocity_parameters=nothing,
        source_function=(x, y, t, u, p) -> zero(u),
        source_parameters=nothing,
        initial_condition,
        initial_time=0.0,
        final_time,
        kwargs...)

## Arguments
- `mesh::FVMGeometry`: The [`FVMGeometry`](@ref).
- `BCs::BoundaryConditions`: The [`BoundaryConditions`](@ref). For these boundary conditions, functions should be of the form `(x, y, t, u, p) -> Number`.
- `ICs::InternalConditions=InternalConditions()`: The [`InternalConditions`](@ref).

## Keyword Arguments
- `diffusion_function`: The diffusion function. Should be of the form `(x, y, t, u, p) -> Number`, where `p = diffusion_parameters`.
- `diffusion_parameters=nothing`: The argument `p` in `diffusion_function`.
- `velocity_function`: The velocity function. Should be of the form `(x, y, t, u, p) -> (vx, vy)`, where `p = velocity_parameters`. Returns a tuple of velocity components.
- `velocity_parameters=nothing`: The argument `p` in `velocity_function`.
- `source_function=(x, y, t, u, p) -> zero(u)`: The source function (optional). Should be of the form `(x, y, t, u, p) -> Number`.
- `source_parameters=nothing`: The argument `p` in `source_function`.
- `initial_condition`: The initial condition.
- `initial_time=0.0`: The initial time.
- `final_time`: The final time.
- `kwargs...`: Any other keyword arguments are passed to the `ODEProblem` that represents the problem.

# Fields
The struct has the following fields:
- `mesh`: The [`FVMGeometry`](@ref).
- `conditions`: The [`Conditions`](@ref).
- `diffusion_function`: The diffusion function.
- `diffusion_parameters`: The diffusion parameters.
- `velocity_function`: The velocity function.
- `velocity_parameters`: The velocity parameters.
- `source_function`: The source function.
- `source_parameters`: The source parameters.
- `initial_condition`: The initial condition.
- `initial_time`: The initial time.
- `final_time`: The final time.
- `problem`: The `ODEProblem` that represents the problem.

# Example

```julia
using FiniteVolumeMethod, DelaunayTriangulation

# Create a simple rectangular mesh
a, b, c, d = 0.0, 1.0, 0.0, 1.0
nx, ny = 20, 20
tri = triangulate_rectangle(a, b, c, d, nx, ny)
mesh = FVMGeometry(tri)

# Define functions
D = (x, y, t, u, p) -> p  # Constant diffusion
v = (x, y, t, u, p) -> (p[1], p[2])  # Constant velocity

# Boundary conditions (Dirichlet on all sides)
bc_func = (x, y, t, u, p) -> 0.0
BCs = BoundaryConditions(mesh, bc_func, Dirichlet)

# Initial condition (Gaussian pulse)
ic = [exp(-50*((x-0.3)^2 + (y-0.5)^2)) for (x,y) in DelaunayTriangulation.each_point(tri)]

# Create and solve
prob = AdvectionDiffusionEquation(mesh, BCs;
    diffusion_function=D, diffusion_parameters=0.01,
    velocity_function=v, velocity_parameters=(1.0, 0.0),
    initial_condition=ic, final_time=0.5)
sol = solve(prob, Tsit5())
```
"""
struct AdvectionDiffusionEquation{M, C, D, DP, V, VP, S, SP, IC, FT, ODE} <: AbstractFVMTemplate
    mesh::M
    conditions::C
    diffusion_function::D
    diffusion_parameters::DP
    velocity_function::V
    velocity_parameters::VP
    source_function::S
    source_parameters::SP
    initial_condition::IC
    initial_time::FT
    final_time::FT
    problem::ODE
end

function Base.show(io::IO, ::MIME"text/plain", prob::AdvectionDiffusionEquation)
    nv = DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation)
    t0 = prob.initial_time
    tf = prob.final_time
    return print(io, "AdvectionDiffusionEquation with $(nv) nodes and time span ($t0, $tf)")
end

# Internal struct for passing parameters to the ODE RHS function
struct AdvectionDiffusionParams{M, C, D, DP, V, VP, S, SP}
    mesh::M
    conditions::C
    diffusion_function::D
    diffusion_parameters::DP
    velocity_function::V
    velocity_parameters::VP
    source_function::S
    source_parameters::SP
end

function AdvectionDiffusionEquation(
        mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions=InternalConditions();
        diffusion_function,
        diffusion_parameters=nothing,
        velocity_function,
        velocity_parameters=nothing,
        source_function=(x, y, t, u, p) -> zero(typeof(x)),
        source_parameters=nothing,
        initial_condition,
        initial_time=0.0,
        final_time,
        kwargs...
    )
    conditions = Conditions(mesh, BCs, ICs)
    _ic = copy(initial_condition)
    apply_dirichlet_conditions!(_ic, mesh, conditions)

    params = AdvectionDiffusionParams(
        mesh, conditions,
        diffusion_function, diffusion_parameters,
        velocity_function, velocity_parameters,
        source_function, source_parameters
    )

    ode_prob = ODEProblem(advection_diffusion_rhs!, _ic, (initial_time, final_time), params; kwargs...)

    return AdvectionDiffusionEquation(
        mesh, conditions,
        diffusion_function, diffusion_parameters,
        velocity_function, velocity_parameters,
        source_function, source_parameters,
        initial_condition, initial_time, final_time,
        ode_prob
    )
end

@doc raw"""
    advection_diffusion_rhs!(du, u, p, t)

Compute the right-hand side of the advection-diffusion equation.

The equation is:
```math
\pdv{u_i}{t} = \frac{1}{V_i} \sum_{\sigma \in \mathcal{E}_i} \left[-(\vb v \cdot \vu n)_\sigma u_\sigma + D_\sigma (\grad u \cdot \vu n)_\sigma\right] L_\sigma + S_i
```
"""
function advection_diffusion_rhs!(du, u, p::AdvectionDiffusionParams, t)
    mesh = p.mesh
    conditions = p.conditions
    D_fn = p.diffusion_function
    D_p = p.diffusion_parameters
    v_fn = p.velocity_function
    v_p = p.velocity_parameters
    S_fn = p.source_function
    S_p = p.source_parameters

    fill!(du, zero(eltype(du)))

    # Interior triangle contributions
    _advection_diffusion_triangle_contributions!(du, u, mesh, conditions, D_fn, D_p, v_fn, v_p, t)

    # Boundary edge contributions
    _advection_diffusion_boundary_contributions!(du, u, mesh, conditions, D_fn, D_p, v_fn, v_p, t)

    # Source term contributions
    _advection_diffusion_source_contributions!(du, u, mesh, conditions, S_fn, S_p, t)

    return du
end

function _advection_diffusion_triangle_contributions!(
        du, u, mesh, conditions, D_fn, D_p, v_fn, v_p, t
    )
    for T in each_solid_triangle(mesh.triangulation)
        ijk = triangle_vertices(T)
        i, j, k = ijk
        props = get_triangle_props(mesh, i, j, k)
        s₁₁, s₁₂, s₁₃, s₂₁, s₂₂, s₂₃, _, _, _ = props.shape_function_coefficients

        for (edge_index, (e1, e2)) in enumerate(((i, j), (j, k), (k, i)))
            x, y, nx, ny, ℓ = get_cv_components(props, edge_index)

            # Interpolate u at edge midpoint using shape functions
            u_edge = s₁₁ * x + s₂₁ * y  # This is wrong - need actual interpolation
            # Actually compute shape function interpolation:
            # u(x,y) = α*x + β*y + γ where α = s₁₁*u[i] + s₁₂*u[j] + s₁₃*u[k], etc.
            α = s₁₁ * u[i] + s₁₂ * u[j] + s₁₃ * u[k]
            β = s₂₁ * u[i] + s₂₂ * u[j] + s₂₃ * u[k]
            # Need γ coefficients for constant term - get them from props
            s₃₁, s₃₂, s₃₃ = props.shape_function_coefficients[7:9]
            γ = s₃₁ * u[i] + s₃₂ * u[j] + s₃₃ * u[k]
            u_edge = α * x + β * y + γ

            # Gradient of u: ∇u = (α, β)
            grad_u_x = α
            grad_u_y = β

            # Get diffusion coefficient and velocity at edge midpoint
            D = D_fn(x, y, t, u_edge, D_p)
            vx, vy = v_fn(x, y, t, u_edge, v_p)

            # Compute flux: q = v*u - D*∇u
            # Normal component: q·n = (vx*u - D*∂u/∂x)*nx + (vy*u - D*∂u/∂y)*ny
            advective_flux = (vx * nx + vy * ny) * u_edge
            diffusive_flux = D * (grad_u_x * nx + grad_u_y * ny)
            total_flux = (advective_flux - diffusive_flux) * ℓ

            # Update du for control volumes
            e1_hascond = has_condition(conditions, e1)
            e2_hascond = has_condition(conditions, e2)
            e1_hascond || (du[e1] -= total_flux / get_volume(mesh, e1))
            e2_hascond || (du[e2] += total_flux / get_volume(mesh, e2))
        end
    end
    return nothing
end

function _advection_diffusion_boundary_contributions!(
        du, u, mesh, conditions, D_fn, D_p, v_fn, v_p, t
    )
    for e in keys(get_boundary_edge_map(mesh.triangulation))
        i, j = DelaunayTriangulation.edge_vertices(e)
        nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓ, T, props = get_boundary_cv_components(mesh, i, j)

        # Get shape function coefficients for the boundary triangle
        s₁₁, s₁₂, s₁₃, s₂₁, s₂₂, s₂₃, s₃₁, s₃₂, s₃₃ = props.shape_function_coefficients
        i_T, j_T, k_T = triangle_vertices(T)

        # Compute shape function coefficients
        α = s₁₁ * u[i_T] + s₁₂ * u[j_T] + s₁₃ * u[k_T]
        β = s₂₁ * u[i_T] + s₂₂ * u[j_T] + s₂₃ * u[k_T]
        γ = s₃₁ * u[i_T] + s₃₂ * u[j_T] + s₃₃ * u[k_T]

        # Check boundary condition type
        is_neumann = is_neumann_edge(conditions, i, j)
        is_robin = is_robin_edge(conditions, i, j)

        i_hascond = has_condition(conditions, i)
        j_hascond = has_condition(conditions, j)

        # Contribution at midpoint near node i
        uᵢ = α * mᵢx + β * mᵢy + γ
        if is_neumann
            fidx = get_neumann_fidx(conditions, i, j)
            qn_i = eval_condition_fnc(conditions, fidx, mᵢx, mᵢy, t, uᵢ)
        elseif is_robin
            fidx = get_robin_fidx(conditions, i, j)
            a, b, c = eval_condition_fnc(conditions, fidx, mᵢx, mᵢy, t, uᵢ)
            qn_i = iszero(b) ? zero(uᵢ) : (c - a * uᵢ) / b
        else
            # Use interior flux
            Dᵢ = D_fn(mᵢx, mᵢy, t, uᵢ, D_p)
            vx_i, vy_i = v_fn(mᵢx, mᵢy, t, uᵢ, v_p)
            advective_i = (vx_i * nx + vy_i * ny) * uᵢ
            diffusive_i = Dᵢ * (α * nx + β * ny)
            qn_i = advective_i - diffusive_i
        end

        # Contribution at midpoint near node j
        uⱼ = α * mⱼx + β * mⱼy + γ
        if is_neumann
            fidx = get_neumann_fidx(conditions, i, j)
            qn_j = eval_condition_fnc(conditions, fidx, mⱼx, mⱼy, t, uⱼ)
        elseif is_robin
            fidx = get_robin_fidx(conditions, i, j)
            a, b, c = eval_condition_fnc(conditions, fidx, mⱼx, mⱼy, t, uⱼ)
            qn_j = iszero(b) ? zero(uⱼ) : (c - a * uⱼ) / b
        else
            Dⱼ = D_fn(mⱼx, mⱼy, t, uⱼ, D_p)
            vx_j, vy_j = v_fn(mⱼx, mⱼy, t, uⱼ, v_p)
            advective_j = (vx_j * nx + vy_j * ny) * uⱼ
            diffusive_j = Dⱼ * (α * nx + β * ny)
            qn_j = advective_j - diffusive_j
        end

        # Update du (note: sign convention - flux leaving the domain is positive)
        i_hascond || (du[i] -= qn_i * ℓ / get_volume(mesh, i))
        j_hascond || (du[j] -= qn_j * ℓ / get_volume(mesh, j))
    end
    return nothing
end

function _advection_diffusion_source_contributions!(du, u, mesh, conditions, S_fn, S_p, t)
    for i in each_solid_vertex(mesh.triangulation)
        if !has_condition(conditions, i)
            p = get_point(mesh, i)
            x, y = getxy(p)
            du[i] += S_fn(x, y, t, u[i], S_p)
        end
    end
    return nothing
end

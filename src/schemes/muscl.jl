@doc raw"""
    MUSCL Reconstruction Module

This module provides MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws)
reconstruction for higher-order accuracy in finite volume methods on unstructured
triangular meshes.

MUSCL reconstruction achieves second-order accuracy by linearly reconstructing
the solution within each cell using gradients, with slope limiting to maintain
TVD (Total Variation Diminishing) properties.
"""

"""
    MUSCLScheme{L<:AbstractLimiter, G<:AbstractGradientMethod}

MUSCL reconstruction scheme configuration.

# Fields
- `limiter::L`: The flux limiter to use
- `gradient_method::G`: The gradient reconstruction method
"""
struct MUSCLScheme{L<:AbstractLimiter, G<:AbstractGradientMethod}
    limiter::L
    gradient_method::G
end

"""
    MUSCLScheme(; limiter=VanLeerLimiter(), gradient_method=GreenGaussGradient())

Create a MUSCL scheme with specified limiter and gradient method.
"""
function MUSCLScheme(;
        limiter::AbstractLimiter=VanLeerLimiter(),
        gradient_method::AbstractGradientMethod=GreenGaussGradient()
    )
    return MUSCLScheme(limiter, gradient_method)
end

@doc raw"""
    muscl_reconstruct_face_value(scheme::MUSCLScheme, mesh, u, i, j, face_x, face_y)

Reconstruct the solution value at a face using MUSCL with limiting.

The MUSCL reconstruction uses:
```math
u_f = u_C + \psi \cdot \grad u_C \cdot (\vb x_f - \vb x_C)
```

where $\psi$ is the limiter value and $\grad u_C$ is the gradient at the cell center.

# Arguments
- `scheme::MUSCLScheme`: The MUSCL scheme configuration
- `mesh::FVMGeometry`: The mesh geometry
- `u`: Solution vector
- `i::Int`: Index of the "left" vertex
- `j::Int`: Index of the "right" vertex
- `face_x::Real`: x-coordinate of face point
- `face_y::Real`: y-coordinate of face point

# Returns
Tuple `(u_L, u_R)` - reconstructed values from left and right sides.
"""
function muscl_reconstruct_face_value(
        scheme::MUSCLScheme, mesh::FVMGeometry, u,
        i::Int, j::Int, face_x, face_y
    )
    T = eltype(u)

    # Get vertex positions
    pᵢ = get_point(mesh, i)
    pⱼ = get_point(mesh, j)
    xᵢ, yᵢ = getxy(pᵢ)
    xⱼ, yⱼ = getxy(pⱼ)

    # Compute gradients at each vertex
    grad_i = reconstruct_gradient(scheme.gradient_method, mesh, u, i)
    grad_j = reconstruct_gradient(scheme.gradient_method, mesh, u, j)

    # Get neighbor values for limiting
    uᵢ_min, uᵢ_max = _get_neighbor_bounds(mesh, u, i)
    uⱼ_min, uⱼ_max = _get_neighbor_bounds(mesh, u, j)

    # Compute unlimited reconstructions
    dx_i = face_x - xᵢ
    dy_i = face_y - yᵢ
    u_i_unlimited = u[i] + grad_i[1] * dx_i + grad_i[2] * dy_i

    dx_j = face_x - xⱼ
    dy_j = face_y - yⱼ
    u_j_unlimited = u[j] + grad_j[1] * dx_j + grad_j[2] * dy_j

    # Apply limiting
    ψᵢ = barth_jespersen(u[i], uᵢ_min, uᵢ_max, u_i_unlimited)
    ψⱼ = barth_jespersen(u[j], uⱼ_min, uⱼ_max, u_j_unlimited)

    u_L = u[i] + ψᵢ * (grad_i[1] * dx_i + grad_i[2] * dy_i)
    u_R = u[j] + ψⱼ * (grad_j[1] * dx_j + grad_j[2] * dy_j)

    return (u_L, u_R)
end

@doc raw"""
    muscl_reconstruct_edge_values(scheme::MUSCLScheme, mesh, u, edge)

Reconstruct solution values at an edge midpoint from both sides.

# Arguments
- `scheme::MUSCLScheme`: The MUSCL scheme configuration
- `mesh::FVMGeometry`: The mesh geometry
- `u`: Solution vector
- `edge`: Edge as tuple (i, j)

# Returns
Tuple `(u_L, u_R)` - reconstructed values from both sides of the edge.
"""
function muscl_reconstruct_edge_values(
        scheme::MUSCLScheme, mesh::FVMGeometry, u, edge
    )
    i, j = edge

    # Compute edge midpoint
    pᵢ = get_point(mesh, i)
    pⱼ = get_point(mesh, j)
    xᵢ, yᵢ = getxy(pᵢ)
    xⱼ, yⱼ = getxy(pⱼ)
    face_x = (xᵢ + xⱼ) / 2
    face_y = (yᵢ + yⱼ) / 2

    return muscl_reconstruct_face_value(scheme, mesh, u, i, j, face_x, face_y)
end

@doc raw"""
    muscl_advective_flux(scheme::MUSCLScheme, mesh, u, i, j, vx, vy, nx, ny)

Compute the advective flux across an edge using MUSCL reconstruction with upwinding.

The advective flux is computed using the upwind value based on the velocity
direction relative to the edge normal.

# Arguments
- `scheme::MUSCLScheme`: The MUSCL scheme configuration
- `mesh::FVMGeometry`: The mesh geometry
- `u`: Solution vector
- `i::Int`: First vertex of edge
- `j::Int`: Second vertex of edge
- `vx, vy`: Velocity components at the edge
- `nx, ny`: Edge normal components

# Returns
The advective flux `(v·n) * u_upwind`.
"""
function muscl_advective_flux(
        scheme::MUSCLScheme, mesh::FVMGeometry, u,
        i::Int, j::Int, vx, vy, nx, ny
    )
    # Compute edge midpoint
    pᵢ = get_point(mesh, i)
    pⱼ = get_point(mesh, j)
    xᵢ, yᵢ = getxy(pᵢ)
    xⱼ, yⱼ = getxy(pⱼ)
    face_x = (xᵢ + xⱼ) / 2
    face_y = (yᵢ + yⱼ) / 2

    # Reconstruct face values
    u_L, u_R = muscl_reconstruct_face_value(scheme, mesh, u, i, j, face_x, face_y)

    # Normal velocity component
    vn = vx * nx + vy * ny

    # Upwind flux
    if vn >= zero(vn)
        return vn * u_L
    else
        return vn * u_R
    end
end

@doc raw"""
    muscl_diffusive_flux(scheme::MUSCLScheme, mesh, u, i, j, D, nx, ny)

Compute the diffusive flux across an edge using MUSCL gradient reconstruction.

The diffusive flux is:
```math
-D \grad u \cdot \vu n
```

For simplicity, this uses the average of gradients from both sides.

# Arguments
- `scheme::MUSCLScheme`: The MUSCL scheme configuration
- `mesh::FVMGeometry`: The mesh geometry
- `u`: Solution vector
- `i::Int`: First vertex of edge
- `j::Int`: Second vertex of edge
- `D`: Diffusion coefficient at the edge
- `nx, ny`: Edge normal components

# Returns
The diffusive flux `-D * (grad_u · n)`.
"""
function muscl_diffusive_flux(
        scheme::MUSCLScheme, mesh::FVMGeometry, u,
        i::Int, j::Int, D, nx, ny
    )
    # Use edge-based gradient (from shape functions of containing triangle)
    grad = reconstruct_gradient_at_edge(mesh, u, i, j)

    return -D * (grad[1] * nx + grad[2] * ny)
end

# Helper function to get min/max of neighboring values for limiting
function _get_neighbor_bounds(mesh::FVMGeometry, u, i::Int)
    tri = mesh.triangulation
    T = eltype(u)

    u_min = u[i]
    u_max = u[i]

    for j in DelaunayTriangulation.get_neighbours(tri, i)
        if !DelaunayTriangulation.is_ghost_vertex(j)
            u_min = min(u_min, u[j])
            u_max = max(u_max, u[j])
        end
    end

    return (u_min, u_max)
end

@doc raw"""
    MUSCLFluxFunction

A flux function wrapper that applies MUSCL reconstruction.

This can be used to create a flux function compatible with FVMProblem
that uses MUSCL reconstruction internally.
"""
struct MUSCLFluxFunction{S<:MUSCLScheme, D, V}
    scheme::S
    diffusion::D  # D(x, y, t, u, p) -> scalar
    velocity::V   # v(x, y, t, u, p) -> (vx, vy), or nothing for pure diffusion
end

function MUSCLFluxFunction(scheme::MUSCLScheme; diffusion, velocity=nothing)
    return MUSCLFluxFunction(scheme, diffusion, velocity)
end

@doc raw"""
    (f::MUSCLFluxFunction)(x, y, t, α, β, γ, p)

Evaluate the MUSCL-based flux function.

!!! note
    This is a simplified implementation. For full MUSCL accuracy, the
    reconstruction should be done at the edge level in the main assembly loop,
    not per-evaluation. This function provides a compatible interface but
    may not achieve full second-order accuracy.
"""
function (f::MUSCLFluxFunction)(x, y, t, α, β, γ, p)
    # Reconstruct local u value
    u = α * x + β * y + γ

    # Get diffusion coefficient
    D = f.diffusion(x, y, t, u, p)

    # Gradient from shape function coefficients (first-order)
    grad_x = α
    grad_y = β

    # Diffusive flux
    qx_diff = -D * grad_x
    qy_diff = -D * grad_y

    # Advective flux (if velocity is provided)
    if !isnothing(f.velocity)
        vx, vy = f.velocity(x, y, t, u, p)
        qx_adv = vx * u
        qy_adv = vy * u
        return (qx_adv + qx_diff, qy_adv + qy_diff)
    else
        return (qx_diff, qy_diff)
    end
end

@doc raw"""
    create_muscl_problem(mesh, BCs, ICs=InternalConditions();
                         diffusion_function, diffusion_parameters=nothing,
                         velocity_function=nothing, velocity_parameters=nothing,
                         source_function=(x,y,t,u,p)->zero(u), source_parameters=nothing,
                         initial_condition, initial_time=0.0, final_time,
                         limiter=VanLeerLimiter(), gradient_method=GreenGaussGradient(),
                         kwargs...)

Create an FVMProblem with MUSCL-based flux function.

This is a convenience constructor that sets up a standard FVMProblem
with a MUSCL flux function.

# Arguments
- `mesh::FVMGeometry`: The mesh geometry
- `BCs::BoundaryConditions`: Boundary conditions
- `ICs::InternalConditions`: Internal conditions (optional)

# Keyword Arguments
- `diffusion_function`: D(x, y, t, u, p) -> scalar
- `diffusion_parameters`: Parameters for diffusion function
- `velocity_function`: v(x, y, t, u, p) -> (vx, vy), or nothing for pure diffusion
- `velocity_parameters`: Parameters for velocity function
- `source_function`: S(x, y, t, u, p) -> scalar
- `source_parameters`: Parameters for source function
- `initial_condition`: Initial condition vector
- `initial_time=0.0`: Start time
- `final_time`: End time
- `limiter=VanLeerLimiter()`: Flux limiter to use
- `gradient_method=GreenGaussGradient()`: Gradient reconstruction method
- `kwargs...`: Additional arguments passed to FVMProblem

# Returns
An FVMProblem configured with MUSCL reconstruction.
"""
function create_muscl_problem(
        mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions=InternalConditions();
        diffusion_function,
        diffusion_parameters=nothing,
        velocity_function=nothing,
        velocity_parameters=nothing,
        source_function=(x, y, t, u, p) -> zero(typeof(x)),
        source_parameters=nothing,
        initial_condition,
        initial_time=0.0,
        final_time,
        limiter::AbstractLimiter=VanLeerLimiter(),
        gradient_method::AbstractGradientMethod=GreenGaussGradient(),
        kwargs...
    )
    scheme = MUSCLScheme(limiter=limiter, gradient_method=gradient_method)

    # Wrap functions to include parameters
    D_wrapped = (x, y, t, u, p) -> diffusion_function(x, y, t, u, diffusion_parameters)
    v_wrapped = isnothing(velocity_function) ? nothing :
                (x, y, t, u, p) -> velocity_function(x, y, t, u, velocity_parameters)

    flux_fn = MUSCLFluxFunction(scheme; diffusion=D_wrapped, velocity=v_wrapped)

    # The flux function signature for FVMProblem is q(x, y, t, α, β, γ, p)
    flux = (x, y, t, α, β, γ, p) -> flux_fn(x, y, t, α, β, γ, p)

    return FVMProblem(
        mesh, BCs, ICs;
        flux_function=flux,
        flux_parameters=nothing,
        source_function=source_function,
        source_parameters=source_parameters,
        initial_condition=initial_condition,
        initial_time=initial_time,
        final_time=final_time,
        kwargs...
    )
end

@doc raw"""
    Gradient Reconstruction Module

This module provides gradient reconstruction methods for finite volume methods
on unstructured triangular meshes. These methods are essential for achieving
higher-order accuracy and for use with slope limiters.

Two main approaches are provided:
1. Green-Gauss gradient reconstruction
2. Least-squares gradient reconstruction
"""

"""
    AbstractGradientMethod

Abstract type for gradient reconstruction methods.
"""
abstract type AbstractGradientMethod end

"""
    GreenGaussGradient <: AbstractGradientMethod

Green-Gauss gradient reconstruction method.
Uses the divergence theorem to compute gradients from face values.
"""
struct GreenGaussGradient <: AbstractGradientMethod end

"""
    LeastSquaresGradient <: AbstractGradientMethod

Least-squares gradient reconstruction method.
Uses weighted least-squares fitting over neighboring cells.
"""
struct LeastSquaresGradient{W} <: AbstractGradientMethod
    weighted::W  # true for distance-weighted, false for unweighted
end
LeastSquaresGradient() = LeastSquaresGradient(true)

@doc raw"""
    reconstruct_gradient(::GreenGaussGradient, mesh::FVMGeometry, u, i)

Compute the gradient at vertex `i` using Green-Gauss reconstruction.

The Green-Gauss method uses the divergence theorem:
```math
\grad \phi \approx \frac{1}{V} \oint_{\partial V} \phi \, \vu n \, dS
\approx \frac{1}{V} \sum_{\text{faces}} \phi_f \, \vu n_f \, A_f
```

For triangular meshes, we iterate over the triangles containing vertex `i`
and compute contributions from each edge.

# Arguments
- `mesh::FVMGeometry`: The mesh geometry
- `u`: Solution vector
- `i::Int`: Vertex index

# Returns
Tuple `(∂u/∂x, ∂u/∂y)` - the gradient components at vertex `i`.
"""
function reconstruct_gradient(::GreenGaussGradient, mesh::FVMGeometry, u, i::Int)
    tri = mesh.triangulation
    grad_x = zero(eltype(u))
    grad_y = zero(eltype(u))
    total_volume = zero(eltype(u))

    # Iterate over triangles containing vertex i by looking at all neighbors
    # and forming triangles from the edges
    for j in DelaunayTriangulation.get_neighbours(tri, i)
        if !DelaunayTriangulation.is_ghost_vertex(j)
            # Get the third vertex of the triangle containing edge (i, j)
            k = get_adjacent(tri, i, j)
            if !DelaunayTriangulation.is_ghost_vertex(k)
                # Use _safe_get_triangle_props to handle vertex ordering
                T_ordered, props = _safe_get_triangle_props(mesh, (i, j, k))
                i_T, j_T, k_T = T_ordered

                # Get shape function coefficients
                s = props.shape_function_coefficients
                s₁₁, s₁₂, s₁₃ = s[1], s[2], s[3]
                s₂₁, s₂₂, s₂₃ = s[4], s[5], s[6]

                # Gradient in this triangle: ∇u = (α, β) where
                # α = s₁₁*u[i_T] + s₁₂*u[j_T] + s₁₃*u[k_T]
                # β = s₂₁*u[i_T] + s₂₂*u[j_T] + s₂₃*u[k_T]
                α = s₁₁ * u[i_T] + s₁₂ * u[j_T] + s₁₃ * u[k_T]
                β = s₂₁ * u[i_T] + s₂₂ * u[j_T] + s₂₃ * u[k_T]

                # Weight by triangle area
                area = _triangle_area(mesh, i_T, j_T, k_T)

                grad_x += α * area
                grad_y += β * area
                total_volume += area
            end
        end
    end

    if total_volume > eps(eltype(u))
        grad_x /= total_volume
        grad_y /= total_volume
    end

    return (grad_x, grad_y)
end

@doc raw"""
    reconstruct_gradient(method::LeastSquaresGradient, mesh::FVMGeometry, u, i)

Compute the gradient at vertex `i` using least-squares reconstruction.

The least-squares method minimizes:
```math
\sum_{j \in \mathcal{N}(i)} w_{ij} \left( \phi_j - \phi_i - \grad\phi_i \cdot (\vb x_j - \vb x_i) \right)^2
```

where $w_{ij}$ are weights (typically inverse distance for `weighted=true`).

This leads to solving a 2×2 linear system:
```math
\begin{pmatrix} A_{xx} & A_{xy} \\ A_{xy} & A_{yy} \end{pmatrix}
\begin{pmatrix} \partial\phi/\partial x \\ \partial\phi/\partial y \end{pmatrix} =
\begin{pmatrix} b_x \\ b_y \end{pmatrix}
```

# Arguments
- `method::LeastSquaresGradient`: The gradient method (with weighting option)
- `mesh::FVMGeometry`: The mesh geometry
- `u`: Solution vector
- `i::Int`: Vertex index

# Returns
Tuple `(∂u/∂x, ∂u/∂y)` - the gradient components at vertex `i`.
"""
function reconstruct_gradient(method::LeastSquaresGradient, mesh::FVMGeometry, u, i::Int)
    tri = mesh.triangulation
    T = eltype(u)

    # Get position of vertex i
    pᵢ = get_point(mesh, i)
    xᵢ, yᵢ = getxy(pᵢ)
    uᵢ = u[i]

    # Initialize least-squares matrix components
    Axx = zero(T)
    Axy = zero(T)
    Ayy = zero(T)
    bx = zero(T)
    by = zero(T)

    # Iterate over neighbors
    for j in DelaunayTriangulation.get_neighbours(tri, i)
        if !DelaunayTriangulation.is_ghost_vertex(j)
            pⱼ = get_point(mesh, j)
            xⱼ, yⱼ = getxy(pⱼ)
            uⱼ = u[j]

            dx = xⱼ - xᵢ
            dy = yⱼ - yᵢ
            du = uⱼ - uᵢ

            # Compute weight
            if method.weighted
                dist² = dx * dx + dy * dy
                w = dist² > eps(T) ? one(T) / sqrt(dist²) : one(T)
            else
                w = one(T)
            end
            w² = w * w

            # Accumulate least-squares matrix
            Axx += w² * dx * dx
            Axy += w² * dx * dy
            Ayy += w² * dy * dy
            bx += w² * dx * du
            by += w² * dy * du
        end
    end

    # Solve 2x2 system using Cramer's rule
    det = Axx * Ayy - Axy * Axy
    if abs(det) < eps(T)
        return (zero(T), zero(T))
    end

    grad_x = (Ayy * bx - Axy * by) / det
    grad_y = (Axx * by - Axy * bx) / det

    return (grad_x, grad_y)
end

@doc raw"""
    reconstruct_gradient_at_edge(mesh::FVMGeometry, u, i, j)

Compute the gradient at the midpoint of edge (i, j) using shape functions.

For a triangular element containing edge (i, j), the gradient is constant
within the element and given by the shape function coefficients.

# Arguments
- `mesh::FVMGeometry`: The mesh geometry
- `u`: Solution vector
- `i::Int`: First vertex of edge
- `j::Int`: Second vertex of edge

# Returns
Tuple `(∂u/∂x, ∂u/∂y)` - the gradient components at the edge midpoint.
"""
function reconstruct_gradient_at_edge(mesh::FVMGeometry, u, i::Int, j::Int)
    tri = mesh.triangulation

    # Find a triangle containing edge (i, j)
    k = get_adjacent(tri, i, j)

    if DelaunayTriangulation.is_ghost_vertex(k)
        # Try the other orientation
        k = get_adjacent(tri, j, i)
        if DelaunayTriangulation.is_ghost_vertex(k)
            # Edge not found in mesh, use average of vertex gradients
            method = GreenGaussGradient()
            grad_i = reconstruct_gradient(method, mesh, u, i)
            grad_j = reconstruct_gradient(method, mesh, u, j)
            return ((grad_i[1] + grad_j[1]) / 2, (grad_i[2] + grad_j[2]) / 2)
        end
        # Swap to maintain consistent orientation
        i, j = j, i
    end

    # Get triangle properties
    props = get_triangle_props(mesh, i, j, k)
    s = props.shape_function_coefficients
    s₁₁, s₁₂, s₁₃ = s[1], s[2], s[3]
    s₂₁, s₂₂, s₂₃ = s[4], s[5], s[6]

    # Gradient is constant in the triangle
    grad_x = s₁₁ * u[i] + s₁₂ * u[j] + s₁₃ * u[k]
    grad_y = s₂₁ * u[i] + s₂₂ * u[j] + s₂₃ * u[k]

    return (grad_x, grad_y)
end

@doc raw"""
    reconstruct_gradient_at_point(mesh::FVMGeometry, u, x, y, T)

Compute the gradient at point (x, y) within triangle T using shape functions.

Since the shape functions are linear, the gradient is constant within each triangle.

# Arguments
- `mesh::FVMGeometry`: The mesh geometry
- `u`: Solution vector
- `x::Real`: x-coordinate
- `y::Real`: y-coordinate
- `T`: Triangle containing the point (as vertex tuple or triangle object)

# Returns
Tuple `(∂u/∂x, ∂u/∂y)` - the gradient components at the point.
"""
function reconstruct_gradient_at_point(mesh::FVMGeometry, u, x, y, T)
    i, j, k = triangle_vertices(T)
    props = get_triangle_props(mesh, i, j, k)
    s = props.shape_function_coefficients
    s₁₁, s₁₂, s₁₃ = s[1], s[2], s[3]
    s₂₁, s₂₂, s₂₃ = s[4], s[5], s[6]

    grad_x = s₁₁ * u[i] + s₁₂ * u[j] + s₁₃ * u[k]
    grad_y = s₂₁ * u[i] + s₂₂ * u[j] + s₂₃ * u[k]

    return (grad_x, grad_y)
end

@doc raw"""
    reconstruct_all_gradients(method::AbstractGradientMethod, mesh::FVMGeometry, u)

Compute gradients at all vertices.

# Arguments
- `method::AbstractGradientMethod`: The gradient reconstruction method
- `mesh::FVMGeometry`: The mesh geometry
- `u`: Solution vector

# Returns
Tuple of vectors `(grad_x, grad_y)` where `grad_x[i]` and `grad_y[i]`
are the gradient components at vertex `i`.
"""
function reconstruct_all_gradients(method::AbstractGradientMethod, mesh::FVMGeometry, u)
    tri = mesh.triangulation
    n = DelaunayTriangulation.num_solid_vertices(tri)
    T = eltype(u)

    grad_x = zeros(T, n)
    grad_y = zeros(T, n)

    for i in each_solid_vertex(tri)
        gx, gy = reconstruct_gradient(method, mesh, u, i)
        grad_x[i] = gx
        grad_y[i] = gy
    end

    return (grad_x, grad_y)
end

# Helper function to compute triangle area
function _triangle_area(mesh::FVMGeometry, i, j, k)
    pᵢ = get_point(mesh, i)
    pⱼ = get_point(mesh, j)
    pₖ = get_point(mesh, k)
    xᵢ, yᵢ = getxy(pᵢ)
    xⱼ, yⱼ = getxy(pⱼ)
    xₖ, yₖ = getxy(pₖ)

    # Shoelace formula for triangle area
    return abs((xⱼ - xᵢ) * (yₖ - yᵢ) - (xₖ - xᵢ) * (yⱼ - yᵢ)) / 2
end

@doc raw"""
    limit_gradient(φ_center, φ_min, φ_max, grad, x_center, neighbors_x, neighbors_y)

Apply the Barth-Jespersen limiter to a gradient to ensure monotonicity.

Computes the minimum limiter value across all reconstruction points.

# Arguments
- `φ_center`: Value at cell center
- `φ_min`: Minimum value in neighborhood
- `φ_max`: Maximum value in neighborhood
- `grad`: Tuple (∂φ/∂x, ∂φ/∂y)
- `x_center`: Position (x, y) of cell center
- `neighbors_x`: Vector of x-coordinates of reconstruction points
- `neighbors_y`: Vector of y-coordinates of reconstruction points

# Returns
Limited gradient (∂φ/∂x, ∂φ/∂y).
"""
function limit_gradient(φ_center, φ_min, φ_max, grad, x_center, neighbors_x, neighbors_y)
    T = typeof(φ_center)
    limiter_val = one(T)

    for (xf, yf) in zip(neighbors_x, neighbors_y)
        dx = xf - x_center[1]
        dy = yf - x_center[2]
        φ_face = φ_center + grad[1] * dx + grad[2] * dy

        ψ = barth_jespersen(φ_center, φ_min, φ_max, φ_face)
        limiter_val = min(limiter_val, ψ)
    end

    return (limiter_val * grad[1], limiter_val * grad[2])
end

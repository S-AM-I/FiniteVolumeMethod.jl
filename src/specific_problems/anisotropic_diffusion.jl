@doc raw"""
    AnisotropicDiffusionEquation <: AbstractFVMTemplate

A struct for defining a problem representing an anisotropic diffusion equation:
```math
\pdv{u}{t} = \div\left[\vb D(\vb x) \cdot \grad u\right]
```
inside a domain $\Omega$, where $\vb D$ is a 2×2 symmetric positive-definite
diffusion tensor:
```math
\vb D = \begin{pmatrix} D_{xx} & D_{xy} \\ D_{xy} & D_{yy} \end{pmatrix}
```

The flux is given by:
```math
\vb q = -\vb D \cdot \grad u = -\begin{pmatrix} D_{xx} \pdv{u}{x} + D_{xy} \pdv{u}{y} \\ D_{xy} \pdv{u}{x} + D_{yy} \pdv{u}{y} \end{pmatrix}
```

## Common Applications

1. **Heat conduction in crystals**: Different thermal conductivity along crystal axes
2. **Groundwater flow**: Anisotropic hydraulic conductivity in layered soils
3. **Fiber-reinforced materials**: Higher diffusion along fiber direction
4. **Image processing**: Perona-Malik style anisotropic smoothing

You can solve this problem using [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)).

!!! warning

    The solution to this problem will have an extra component added to it. The original solution will be inside
    `sol[begin:end-1, :]`, where `sol` is the solution returned by [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)).

# Constructor

    AnisotropicDiffusionEquation(mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions=InternalConditions();
        diffusion_tensor,
        diffusion_parameters=nothing,
        initial_condition,
        initial_time=0.0,
        final_time,
        kwargs...)

## Arguments
- `mesh::FVMGeometry`: The [`FVMGeometry`](@ref).
- `BCs::BoundaryConditions`: The [`BoundaryConditions`](@ref). For these boundary conditions, all functions should still be of the form `(x, y, t, u, p) -> Number`, but the `t` and `u` arguments should be unused as they will be replaced with `nothing`.
- `ICs::InternalConditions=InternalConditions()`: The [`InternalConditions`](@ref).

## Keyword Arguments
- `diffusion_tensor`: The diffusion tensor function. Should be of the form `(x, y, p) -> (Dxx, Dxy, Dyy)`, returning the upper-triangular components of the symmetric tensor (Dxy = Dyx).
- `diffusion_parameters=nothing`: The argument `p` in `diffusion_tensor`.
- `initial_condition`: The initial condition.
- `initial_time=0.0`: The initial time.
- `final_time`: The final time.
- `kwargs...`: Any other keyword arguments are passed to the `ODEProblem`.

# Fields
The struct has extra fields in addition to the arguments above:
- `A`: This is a sparse matrix `A` so that `du/dt = Au + b`.
- `b`: The `b` above.
- `Aop`: The `MatrixOperator` that represents the system.
- `problem`: The `ODEProblem` that represents the problem.

# Example

```julia
using FiniteVolumeMethod, DelaunayTriangulation

# Create mesh
tri = triangulate_rectangle(0, 1, 0, 1, 20, 20, single_boundary=true)
mesh = FVMGeometry(tri)

# Dirichlet boundary conditions
bc = (x, y, t, u, p) -> 0.0
BCs = BoundaryConditions(mesh, bc, Dirichlet)

# Anisotropic diffusion tensor: stronger diffusion in x-direction
# D = [[2, 0], [0, 0.5]]
diffusion_tensor = (x, y, p) -> (2.0, 0.0, 0.5)

# Initial condition: Gaussian bump
u0 = [exp(-((x-0.5)^2 + (y-0.5)^2)/0.01) for (x, y) in DelaunayTriangulation.each_point(tri)]

prob = AnisotropicDiffusionEquation(
    mesh, BCs;
    diffusion_tensor = diffusion_tensor,
    initial_condition = u0,
    final_time = 0.1
)

using OrdinaryDiffEq
sol = solve(prob, Tsit5(); saveat=0.01)
```
"""
struct AnisotropicDiffusionEquation{M, C, D, DP, IC, FT, A, B, OP, ODE} <: AbstractFVMTemplate
    mesh::M
    conditions::C
    diffusion_tensor::D
    diffusion_parameters::DP
    initial_condition::IC
    initial_time::FT
    final_time::FT
    A::A
    b::B
    Aop::OP
    problem::ODE
end

function Base.show(io::IO, ::MIME"text/plain", prob::AnisotropicDiffusionEquation)
    nv = DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation)
    t0 = prob.initial_time
    tf = prob.final_time
    return print(io, "AnisotropicDiffusionEquation with $(nv) nodes and time span ($t0, $tf)")
end

@doc raw"""
    anisotropic_triangle_contributions!(A, mesh, conditions, diffusion_tensor, diffusion_parameters)

Add the contributions from each triangle to the matrix `A` for anisotropic diffusion.

For the anisotropic flux $\vb q = -\vb D \cdot \grad u$, the contribution to node $i$ from
edge $\sigma$ is:
```math
\frac{1}{V_i} \left[ (D_{xx} n_x + D_{xy} n_y) \pdv{u}{x} + (D_{xy} n_x + D_{yy} n_y) \pdv{u}{y} \right] L_\sigma
```
"""
function anisotropic_triangle_contributions!(
        A, mesh, conditions, diffusion_tensor, diffusion_parameters
    )
    for T in each_solid_triangle(mesh.triangulation)
        ijk = triangle_vertices(T)
        i, j, k = ijk
        props = get_triangle_props(mesh, i, j, k)
        s₁₁, s₁₂, s₁₃, s₂₁, s₂₂, s₂₃, s₃₁, s₃₂, s₃₃ = props.shape_function_coefficients
        for (edge_index, (e1, e2)) in enumerate(((i, j), (j, k), (k, i)))
            x, y, nx, ny, ℓ = get_cv_components(props, edge_index)
            Dxx, Dxy, Dyy = diffusion_tensor(x, y, diffusion_parameters)

            # Flux normal components with tensor diffusion
            # q·n = -(Dxx*nx + Dxy*ny)*∂u/∂x - (Dxy*nx + Dyy*ny)*∂u/∂y
            coef_x = (Dxx * nx + Dxy * ny) * ℓ  # Coefficient for ∂u/∂x
            coef_y = (Dxy * nx + Dyy * ny) * ℓ  # Coefficient for ∂u/∂y

            a123 = (
                coef_x * s₁₁ + coef_y * s₂₁,
                coef_x * s₁₂ + coef_y * s₂₂,
                coef_x * s₁₃ + coef_y * s₂₃,
            )

            e1_hascond = has_condition(conditions, e1)
            e2_hascond = has_condition(conditions, e2)
            for vert in 1:3
                e1_hascond || (A[e1, ijk[vert]] += a123[vert] / get_volume(mesh, e1))
                e2_hascond || (A[e2, ijk[vert]] -= a123[vert] / get_volume(mesh, e2))
            end
        end
    end
    return
end

@doc raw"""
    anisotropic_boundary_edge_contributions!(A, b, mesh, conditions, diffusion_tensor, diffusion_parameters)

Add the contributions from each boundary edge to the matrix `A` and vector `b` for anisotropic diffusion.
"""
function anisotropic_boundary_edge_contributions!(
        A, b, mesh, conditions,
        diffusion_tensor, diffusion_parameters
    )
    anisotropic_non_neumann_boundary_edge_contributions!(
        A, mesh, conditions, diffusion_tensor, diffusion_parameters
    )
    anisotropic_neumann_boundary_edge_contributions!(
        b, mesh, conditions, diffusion_tensor, diffusion_parameters
    )
    return nothing
end

@doc raw"""
    anisotropic_neumann_boundary_edge_contributions!(b, mesh, conditions, diffusion_tensor, diffusion_parameters)

Add Neumann boundary contributions for anisotropic diffusion.

For anisotropic diffusion, the Neumann condition specifies $\vb q \cdot \vu n$ directly,
where $\vb q = -\vb D \cdot \grad u$.
"""
function anisotropic_neumann_boundary_edge_contributions!(
        b, mesh, conditions, diffusion_tensor, diffusion_parameters
    )
    for (e, fidx) in get_neumann_edges(conditions)
        i, j = DelaunayTriangulation.edge_vertices(e)
        _, _, mᵢx, mᵢy, mⱼx, mⱼy, ℓ, _, _ = get_boundary_cv_components(mesh, i, j)

        # Get tensor at midpoints (though for Neumann we use the flux directly)
        Dxx_i, Dxy_i, Dyy_i = diffusion_tensor(mᵢx, mᵢy, diffusion_parameters)
        Dxx_j, Dxy_j, Dyy_j = diffusion_tensor(mⱼx, mⱼy, diffusion_parameters)

        # Effective diffusion magnitude for scaling (use trace as a measure)
        D_eff_i = sqrt(Dxx_i * Dyy_i - Dxy_i^2)  # sqrt(det(D))
        D_eff_j = sqrt(Dxx_j * Dyy_j - Dxy_j^2)

        # Fallback to average of diagonal if det is small
        if D_eff_i < 1e-10
            D_eff_i = (Dxx_i + Dyy_i) / 2
        end
        if D_eff_j < 1e-10
            D_eff_j = (Dxx_j + Dyy_j) / 2
        end

        i_hascond = has_condition(conditions, i)
        j_hascond = has_condition(conditions, j)
        aᵢ = eval_condition_fnc(conditions, fidx, mᵢx, mᵢy, nothing, nothing)
        aⱼ = eval_condition_fnc(conditions, fidx, mⱼx, mⱼy, nothing, nothing)
        i_hascond || (b[i] += D_eff_i * aᵢ * ℓ / get_volume(mesh, i))
        j_hascond || (b[j] += D_eff_j * aⱼ * ℓ / get_volume(mesh, j))
    end
    return nothing
end

@doc raw"""
    anisotropic_non_neumann_boundary_edge_contributions!(A, mesh, conditions, diffusion_tensor, diffusion_parameters)

Add non-Neumann (Dirichlet) boundary edge contributions for anisotropic diffusion.
"""
function anisotropic_non_neumann_boundary_edge_contributions!(
        A, mesh, conditions, diffusion_tensor, diffusion_parameters
    )
    for e in keys(get_boundary_edge_map(mesh.triangulation))
        i, j = DelaunayTriangulation.edge_vertices(e)
        if !is_neumann_edge(conditions, i, j)
            nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓ, T, props = get_boundary_cv_components(mesh, i, j)
            ijk = triangle_vertices(T)
            s₁₁, s₁₂, s₁₃, s₂₁, s₂₂, s₂₃, s₃₁, s₃₂, s₃₃ = props.shape_function_coefficients

            # Tensor at midpoints
            Dxx_i, Dxy_i, Dyy_i = diffusion_tensor(mᵢx, mᵢy, diffusion_parameters)
            Dxx_j, Dxy_j, Dyy_j = diffusion_tensor(mⱼx, mⱼy, diffusion_parameters)

            # Coefficients for i midpoint
            coef_x_i = (Dxx_i * nx + Dxy_i * ny) * ℓ
            coef_y_i = (Dxy_i * nx + Dyy_i * ny) * ℓ

            # Coefficients for j midpoint
            coef_x_j = (Dxx_j * nx + Dxy_j * ny) * ℓ
            coef_y_j = (Dxy_j * nx + Dyy_j * ny) * ℓ

            i_hascond = has_condition(conditions, i)
            j_hascond = has_condition(conditions, j)

            aᵢ123 = (
                coef_x_i * s₁₁ + coef_y_i * s₂₁,
                coef_x_i * s₁₂ + coef_y_i * s₂₂,
                coef_x_i * s₁₃ + coef_y_i * s₂₃,
            )
            aⱼ123 = (
                coef_x_j * s₁₁ + coef_y_j * s₂₁,
                coef_x_j * s₁₂ + coef_y_j * s₂₂,
                coef_x_j * s₁₃ + coef_y_j * s₂₃,
            )

            for vert in 1:3
                i_hascond || (A[i, ijk[vert]] += aᵢ123[vert] / get_volume(mesh, i))
                j_hascond || (A[j, ijk[vert]] += aⱼ123[vert] / get_volume(mesh, i))
            end
        end
    end
    return nothing
end

function AnisotropicDiffusionEquation(
        mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions = InternalConditions();
        diffusion_tensor,
        diffusion_parameters = nothing,
        initial_condition,
        initial_time = 0.0,
        final_time,
        kwargs...
    )
    conditions = Conditions(mesh, BCs, ICs)
    n = DelaunayTriangulation.num_solid_vertices(mesh.triangulation)
    Afull = zeros(n + 1, n + 1)
    A = @views Afull[begin:(end - 1), begin:(end - 1)]
    b = @views Afull[begin:(end - 1), end]
    _ic = vcat(initial_condition, 1)

    anisotropic_triangle_contributions!(A, mesh, conditions, diffusion_tensor, diffusion_parameters)
    anisotropic_boundary_edge_contributions!(
        A, b, mesh, conditions, diffusion_tensor, diffusion_parameters
    )
    apply_dudt_conditions!(b, mesh, conditions)
    apply_dirichlet_conditions!(_ic, mesh, conditions)
    fix_missing_vertices!(A, b, mesh)

    A_op = MatrixOperator(sparse(Afull))
    prob = ODEProblem(A_op, _ic, (initial_time, final_time); kwargs...)
    return AnisotropicDiffusionEquation(
        mesh, conditions,
        diffusion_tensor, diffusion_parameters,
        initial_condition, initial_time, final_time,
        sparse(A), b, A_op, prob
    )
end

@doc raw"""
    make_rotation_tensor(θ, D_parallel, D_perp)

Create an anisotropic diffusion tensor rotated by angle θ from the x-axis.

The tensor has diffusion coefficient `D_parallel` along the direction θ and
`D_perp` perpendicular to it.

# Returns
A function `(x, y, p) -> (Dxx, Dxy, Dyy)` suitable for `AnisotropicDiffusionEquation`.

# Example
```julia
# Diffusion 10x stronger at 45° angle
diffusion_tensor = make_rotation_tensor(π/4, 1.0, 0.1)
```
"""
function make_rotation_tensor(θ::Real, D_parallel::Real, D_perp::Real)
    c = cos(θ)
    s = sin(θ)
    # D = R * diag(D_parallel, D_perp) * R^T
    Dxx = D_parallel * c^2 + D_perp * s^2
    Dxy = (D_parallel - D_perp) * c * s
    Dyy = D_parallel * s^2 + D_perp * c^2
    return (x, y, p) -> (Dxx, Dxy, Dyy)
end

@doc raw"""
    make_spatially_varying_tensor(Dxx_func, Dxy_func, Dyy_func)

Create a spatially-varying anisotropic diffusion tensor from component functions.

# Arguments
- `Dxx_func`: Function `(x, y) -> Dxx` for the xx-component
- `Dxy_func`: Function `(x, y) -> Dxy` for the xy-component (= yx-component)
- `Dyy_func`: Function `(x, y) -> Dyy` for the yy-component

# Returns
A function `(x, y, p) -> (Dxx, Dxy, Dyy)` suitable for `AnisotropicDiffusionEquation`.
"""
function make_spatially_varying_tensor(Dxx_func::Function, Dxy_func::Function, Dyy_func::Function)
    return (x, y, p) -> (Dxx_func(x, y), Dxy_func(x, y), Dyy_func(x, y))
end

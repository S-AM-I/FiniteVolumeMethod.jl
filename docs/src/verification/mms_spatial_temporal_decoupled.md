```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/mms_spatial_temporal_decoupled.jl"
```

````julia
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Decoupled Spatial–Temporal MMS (Parabolic)
This example separates spatial and temporal errors in a method of
manufactured solutions (MMS) study. Part A isolates spatial error using
a time-polynomial exact solution (so the ODE integrator is exact).
Part B isolates temporal error on a fixed mesh by sweeping solver tolerances.

## Mathematical Setup
**Part A (spatial):** The exact solution $u(x,y,t) = \sin(\pi x)\sin(\pi y)(1+t)$
is linear in time, so any standard ODE integrator reproduces it exactly.
The diffusion equation $\partial_t u = \nabla^2 u + S$ with $D = 1$ yields:
```math
S(x,y,t) = \sin(\pi x)\sin(\pi y) + 2\pi^2\sin(\pi x)\sin(\pi y)(1+t).
```

**Part B (temporal):** On a fixed fine mesh ($N = 80$), we use the
standard MMS solution $u = \sin(\pi x)\sin(\pi y)e^{-t}$ and sweep ODE
solver tolerances to verify that the temporal error decreases independently.

## Reference
- Roache, P.J. (1998). Verification and Validation in Computational
  Science and Engineering. Hermosa Publishers, Chapters 5 and 8.
- Oberkampf, W.L. & Roy, C.J. (2010). Verification and Validation in
  Scientific Computing. Cambridge University Press.

````julia
using FiniteVolumeMethod, DelaunayTriangulation
using OrdinaryDiffEq
using CairoMakie
````

## Part A — Spatial Convergence (Time-Polynomial Exact Solution)
The exact solution is linear in $t$, so Tsit5 reproduces it exactly.
Only the FVM spatial discretisation error remains.

````julia
u_exact_A(x, y, t) = sin(pi * x) * sin(pi * y) * (1 + t)

function source_A(x, y, t, u, p)
    return sin(pi * x) * sin(pi * y) + 2 * pi^2 * sin(pi * x) * sin(pi * y) * (1 + t)
end

D_const(x, y, t, u, p) = 1.0
bc_zero(x, y, t, u, p) = zero(u)

mesh_sizes_A = [10, 20, 40, 80]
t_final_A = 0.5
errors_A = Float64[]

for N in mesh_sizes_A
    tri = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, N, N; single_boundary = true)
    mesh = FVMGeometry(tri)
    BCs = BoundaryConditions(mesh, bc_zero, Dirichlet)

    f0 = (x, y) -> u_exact_A(x, y, 0.0)
    initial_condition = [f0(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]

    prob = FVMProblem(
        mesh, BCs;
        diffusion_function = D_const,
        source_function = source_A,
        initial_condition = initial_condition,
        final_time = t_final_A,
    )
    sol = solve(prob, Tsit5(); saveat = [t_final_A])

    err = 0.0
    n_pts = 0
    for (j, (x, y)) in enumerate(DelaunayTriangulation.each_point(tri))
        e = abs(sol.u[end][j] - u_exact_A(x, y, t_final_A))
        err = max(err, e)
        n_pts += 1
    end
    push!(errors_A, err)
end
````

## Part A — Convergence Rates

````julia
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_A = convergence_rates(errors_A)
````

## Part B — Temporal Convergence (Fixed Mesh)
Fix the mesh at $N = 80$ and sweep ODE solver tolerances.

````julia
u_exact_B(x, y, t) = sin(pi * x) * sin(pi * y) * exp(-t)

function source_B(x, y, t, u, p)
    return (2 * pi^2 - 1) * sin(pi * x) * sin(pi * y) * exp(-t)
end

N_fixed = 80
t_final_B = 1.0
tol_values = [1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9]
errors_B = Float64[]

tri_B = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, N_fixed, N_fixed; single_boundary = true)
mesh_B = FVMGeometry(tri_B)
BCs_B = BoundaryConditions(mesh_B, bc_zero, Dirichlet)
f0_B = (x, y) -> u_exact_B(x, y, 0.0)
ic_B = [f0_B(x, y) for (x, y) in DelaunayTriangulation.each_point(tri_B)]

for tol in tol_values
    prob = FVMProblem(
        mesh_B, BCs_B;
        diffusion_function = D_const,
        source_function = source_B,
        initial_condition = ic_B,
        final_time = t_final_B,
    )
    sol = solve(prob, Tsit5(); saveat = [t_final_B], abstol = tol, reltol = tol)

    err = 0.0
    for (j, (x, y)) in enumerate(DelaunayTriangulation.each_point(tri_B))
        e = abs(sol.u[end][j] - u_exact_B(x, y, t_final_B))
        err = max(err, e)
    end
    push!(errors_B, err)
end
````

## Visualisation — Spatial Convergence

````julia
h_vals = 1.0 ./ mesh_sizes_A
fig1 = Figure(fontsize = 24, size = (1100, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "h = 1/N", ylabel = L"L^\infty \text{ error}",
    xscale = log10, yscale = log10,
    title = "Part A: Spatial Convergence (linear-in-t)",
)
scatterlines!(
    ax1, h_vals, errors_A, color = :blue, marker = :circle,
    linewidth = 2, markersize = 12, label = "FVM",
)
lines!(
    ax1, h_vals, errors_A[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5, label = "O(h²)",
)
axislegend(ax1, position = :rb)

# Part B plot
ax2 = Axis(
    fig1[1, 2], xlabel = "Solver tolerance", ylabel = L"L^\infty \text{ error}",
    xscale = log10, yscale = log10,
    title = "Part B: Temporal Convergence (fixed mesh)",
)
scatterlines!(
    ax2, tol_values, errors_B, color = :red, marker = :diamond,
    linewidth = 2, markersize = 12, label = "Tsit5",
)
axislegend(ax2, position = :rb)
resize_to_layout!(fig1)
fig1
````

## Test Assertions
Part A: Spatial rates should be approximately second order.
Part B: Errors should decrease monotonically with tighter tolerances.

````julia
@assert minimum(rates_A) > 1.5 #hide
@assert all(errors_B[i] >= errors_B[i + 1] for i in 1:(length(errors_B) - 1)) #hide
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/mms_spatial_temporal_decoupled.jl).

```julia
using FiniteVolumeMethod, DelaunayTriangulation
using OrdinaryDiffEq
using CairoMakie

u_exact_A(x, y, t) = sin(pi * x) * sin(pi * y) * (1 + t)

function source_A(x, y, t, u, p)
    return sin(pi * x) * sin(pi * y) + 2 * pi^2 * sin(pi * x) * sin(pi * y) * (1 + t)
end

D_const(x, y, t, u, p) = 1.0
bc_zero(x, y, t, u, p) = zero(u)

mesh_sizes_A = [10, 20, 40, 80]
t_final_A = 0.5
errors_A = Float64[]

for N in mesh_sizes_A
    tri = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, N, N; single_boundary = true)
    mesh = FVMGeometry(tri)
    BCs = BoundaryConditions(mesh, bc_zero, Dirichlet)

    f0 = (x, y) -> u_exact_A(x, y, 0.0)
    initial_condition = [f0(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]

    prob = FVMProblem(
        mesh, BCs;
        diffusion_function = D_const,
        source_function = source_A,
        initial_condition = initial_condition,
        final_time = t_final_A,
    )
    sol = solve(prob, Tsit5(); saveat = [t_final_A])

    err = 0.0
    n_pts = 0
    for (j, (x, y)) in enumerate(DelaunayTriangulation.each_point(tri))
        e = abs(sol.u[end][j] - u_exact_A(x, y, t_final_A))
        err = max(err, e)
        n_pts += 1
    end
    push!(errors_A, err)
end

function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_A = convergence_rates(errors_A)

u_exact_B(x, y, t) = sin(pi * x) * sin(pi * y) * exp(-t)

function source_B(x, y, t, u, p)
    return (2 * pi^2 - 1) * sin(pi * x) * sin(pi * y) * exp(-t)
end

N_fixed = 80
t_final_B = 1.0
tol_values = [1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9]
errors_B = Float64[]

tri_B = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, N_fixed, N_fixed; single_boundary = true)
mesh_B = FVMGeometry(tri_B)
BCs_B = BoundaryConditions(mesh_B, bc_zero, Dirichlet)
f0_B = (x, y) -> u_exact_B(x, y, 0.0)
ic_B = [f0_B(x, y) for (x, y) in DelaunayTriangulation.each_point(tri_B)]

for tol in tol_values
    prob = FVMProblem(
        mesh_B, BCs_B;
        diffusion_function = D_const,
        source_function = source_B,
        initial_condition = ic_B,
        final_time = t_final_B,
    )
    sol = solve(prob, Tsit5(); saveat = [t_final_B], abstol = tol, reltol = tol)

    err = 0.0
    for (j, (x, y)) in enumerate(DelaunayTriangulation.each_point(tri_B))
        e = abs(sol.u[end][j] - u_exact_B(x, y, t_final_B))
        err = max(err, e)
    end
    push!(errors_B, err)
end

h_vals = 1.0 ./ mesh_sizes_A
fig1 = Figure(fontsize = 24, size = (1100, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "h = 1/N", ylabel = L"L^\infty \text{ error}",
    xscale = log10, yscale = log10,
    title = "Part A: Spatial Convergence (linear-in-t)",
)
scatterlines!(
    ax1, h_vals, errors_A, color = :blue, marker = :circle,
    linewidth = 2, markersize = 12, label = "FVM",
)
lines!(
    ax1, h_vals, errors_A[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5, label = "O(h²)",
)
axislegend(ax1, position = :rb)

# Part B plot
ax2 = Axis(
    fig1[1, 2], xlabel = "Solver tolerance", ylabel = L"L^\infty \text{ error}",
    xscale = log10, yscale = log10,
    title = "Part B: Temporal Convergence (fixed mesh)",
)
scatterlines!(
    ax2, tol_values, errors_B, color = :red, marker = :diamond,
    linewidth = 2, markersize = 12, label = "Tsit5",
)
axislegend(ax2, position = :rb)
resize_to_layout!(fig1)
fig1
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


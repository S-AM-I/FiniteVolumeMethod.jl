```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/porous_medium_barenblatt.jl"
```

````@example porous_medium_barenblatt
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Barenblatt-Pattle Self-Similar Solution (Porous Medium Equation)
This example verifies the parabolic solver on the porous medium equation
(PME) using the Barenblatt-Pattle self-similar solution. The PME has a
nonlinear diffusion coefficient $D(u) = m u^{m-1}$ and admits an exact
compactly-supported self-similar solution.

## Mathematical Setup
The porous medium equation in 2D:
```math
\pdv{u}{t} = \nabla \cdot (m\,u^{m-1}\,\nabla u)
```
with $m = 2$. The Barenblatt-Pattle solution at time $t > 0$ is:
```math
u(r,t) = t^{-\alpha}\left(C - \frac{\alpha(m-1)}{2md}\,\frac{r^2}{t^{2\alpha/d}}\right)_+^{1/(m-1)}
```
where $d = 2$ is the spatial dimension, $\alpha = d/(d(m-1)+2)$, and
$C$ is determined by the initial mass.

## Reference
- Barenblatt, G.I. (1952). On some unsteady motions of a liquid or a
  gas in a porous medium. Prikl. Mat. Mekh., 16, 67-78.
- Vázquez, J.L. (2007). The Porous Medium Equation. Oxford University Press.

````@example porous_medium_barenblatt
using FiniteVolumeMethod, DelaunayTriangulation
using OrdinaryDiffEq
using CairoMakie
````

## Barenblatt-Pattle Exact Solution

````@example porous_medium_barenblatt
m_pme = 2
d_dim = 2
alpha_bp = d_dim / (d_dim * (m_pme - 1) + 2)  # = 1/2 for m=2, d=2

# Choose C so the support is comfortably inside the domain at t_start
C_bp = 0.5
t_start = 1.0
t_final = 2.0

function barenblatt(x, y, t)
    r2 = x^2 + y^2
    coeff = alpha_bp * (m_pme - 1) / (2 * m_pme * d_dim)
    inner = C_bp - coeff * r2 / t^(2 * alpha_bp / d_dim)
    return inner > 0 ? t^(-alpha_bp) * inner^(1 / (m_pme - 1)) : 0.0
end
````

## Nonlinear Diffusion Coefficient
For PME: flux = D(u)∇u where D(u) = m·u^(m-1)

````@example porous_medium_barenblatt
function D_pme(x, y, t, u, p)
    return m_pme * max(u, 0.0)^(m_pme - 1)
end

bc_pme(x, y, t, u, p) = zero(u)
````

## Grid Refinement Study

````@example porous_medium_barenblatt
mesh_sizes = [10, 20, 40]
errors_L2 = Float64[]
errors_Linf = Float64[]

for N in mesh_sizes
    R = 3.0
    tri = triangulate_rectangle(-R, R, -R, R, N, N; single_boundary = true)
    mesh = FVMGeometry(tri)
    BCs = BoundaryConditions(mesh, bc_pme, Dirichlet)

    initial_condition = [barenblatt(x, y, t_start) for (x, y) in DelaunayTriangulation.each_point(tri)]

    prob = FVMProblem(
        mesh, BCs;
        diffusion_function = D_pme,
        initial_condition = initial_condition,
        initial_time = t_start,
        final_time = t_final,
    )
    sol = solve(prob, Tsit5(); saveat = [t_final])

    err_inf = 0.0
    err_l2 = 0.0
    n_pts = 0
    for (j, (x, y)) in enumerate(DelaunayTriangulation.each_point(tri))
        e = abs(sol.u[end][j] - barenblatt(x, y, t_final))
        err_inf = max(err_inf, e)
        err_l2 += e^2
        n_pts += 1
    end
    push!(errors_Linf, err_inf)
    push!(errors_L2, sqrt(err_l2 / n_pts))
end
````

## Convergence Rates

````@example porous_medium_barenblatt
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_L2 = convergence_rates(errors_L2)
````

## Visualisation — Solution at Finest Mesh

````@example porous_medium_barenblatt
R = 3.0
N_vis = mesh_sizes[end]
tri_vis = triangulate_rectangle(-R, R, -R, R, N_vis, N_vis; single_boundary = true)
mesh_vis = FVMGeometry(tri_vis)
BCs_vis = BoundaryConditions(mesh_vis, bc_pme, Dirichlet)
ic_vis = [barenblatt(x, y, t_start) for (x, y) in DelaunayTriangulation.each_point(tri_vis)]

prob_vis = FVMProblem(
    mesh_vis, BCs_vis;
    diffusion_function = D_pme,
    initial_condition = ic_vis,
    initial_time = t_start,
    final_time = t_final,
)
sol_vis = solve(prob_vis, Tsit5(); saveat = [t_final])

u_num = sol_vis.u[end]
u_exact = [barenblatt(x, y, t_final) for (x, y) in DelaunayTriangulation.each_point(tri_vis)]

fig = Figure(fontsize = 24, size = (1500, 450))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Numerical", aspect = DataAspect())
tricontourf!(ax1, tri_vis, u_num, colormap = :viridis)
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = "y", title = "Exact (Barenblatt)", aspect = DataAspect())
tricontourf!(ax2, tri_vis, u_exact, colormap = :viridis)
ax3 = Axis(fig[1, 3], xlabel = "x", ylabel = "y", title = "Error", aspect = DataAspect())
tricontourf!(ax3, tri_vis, abs.(u_num .- u_exact), colormap = :hot)
resize_to_layout!(fig)
fig
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("barenblatt_pattle_solution.png"), fig)
end
````

## Test Assertions
L2 errors should decrease monotonically with mesh refinement.
Convergence rate should be positive. The degenerate front limits order.

````@example porous_medium_barenblatt
@assert all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)) #hide
@assert minimum(rates_L2) > 0.3 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "linf_errors" => errors_Linf,
            "l2_errors" => errors_L2,
            "min_l2_rate" => minimum(rates_L2),
            "finest_l2_error" => errors_L2[end],
        ),
        artifacts = ["barenblatt_pattle_solution.png"],
        notes = [
            "Canonical parabolic solve path via solve(prob, Tsit5()).",
            "This evidence entry is the parabolic benchmark-stage self-similar exact-solution case.",
        ],
        summary = Dict(
            "mesh_sizes" => mesh_sizes,
            "l2_rates" => rates_L2,
            "monotone_l2_decrease" => all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)),
        ),
    )
end
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/porous_medium_barenblatt.jl).

```julia
using FiniteVolumeMethod, DelaunayTriangulation
using OrdinaryDiffEq
using CairoMakie

m_pme = 2
d_dim = 2
alpha_bp = d_dim / (d_dim * (m_pme - 1) + 2)  # = 1/2 for m=2, d=2

# Choose C so the support is comfortably inside the domain at t_start
C_bp = 0.5
t_start = 1.0
t_final = 2.0

function barenblatt(x, y, t)
    r2 = x^2 + y^2
    coeff = alpha_bp * (m_pme - 1) / (2 * m_pme * d_dim)
    inner = C_bp - coeff * r2 / t^(2 * alpha_bp / d_dim)
    return inner > 0 ? t^(-alpha_bp) * inner^(1 / (m_pme - 1)) : 0.0
end

function D_pme(x, y, t, u, p)
    return m_pme * max(u, 0.0)^(m_pme - 1)
end

bc_pme(x, y, t, u, p) = zero(u)

mesh_sizes = [10, 20, 40]
errors_L2 = Float64[]
errors_Linf = Float64[]

for N in mesh_sizes
    R = 3.0
    tri = triangulate_rectangle(-R, R, -R, R, N, N; single_boundary = true)
    mesh = FVMGeometry(tri)
    BCs = BoundaryConditions(mesh, bc_pme, Dirichlet)

    initial_condition = [barenblatt(x, y, t_start) for (x, y) in DelaunayTriangulation.each_point(tri)]

    prob = FVMProblem(
        mesh, BCs;
        diffusion_function = D_pme,
        initial_condition = initial_condition,
        initial_time = t_start,
        final_time = t_final,
    )
    sol = solve(prob, Tsit5(); saveat = [t_final])

    err_inf = 0.0
    err_l2 = 0.0
    n_pts = 0
    for (j, (x, y)) in enumerate(DelaunayTriangulation.each_point(tri))
        e = abs(sol.u[end][j] - barenblatt(x, y, t_final))
        err_inf = max(err_inf, e)
        err_l2 += e^2
        n_pts += 1
    end
    push!(errors_Linf, err_inf)
    push!(errors_L2, sqrt(err_l2 / n_pts))
end

function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_L2 = convergence_rates(errors_L2)

R = 3.0
N_vis = mesh_sizes[end]
tri_vis = triangulate_rectangle(-R, R, -R, R, N_vis, N_vis; single_boundary = true)
mesh_vis = FVMGeometry(tri_vis)
BCs_vis = BoundaryConditions(mesh_vis, bc_pme, Dirichlet)
ic_vis = [barenblatt(x, y, t_start) for (x, y) in DelaunayTriangulation.each_point(tri_vis)]

prob_vis = FVMProblem(
    mesh_vis, BCs_vis;
    diffusion_function = D_pme,
    initial_condition = ic_vis,
    initial_time = t_start,
    final_time = t_final,
)
sol_vis = solve(prob_vis, Tsit5(); saveat = [t_final])

u_num = sol_vis.u[end]
u_exact = [barenblatt(x, y, t_final) for (x, y) in DelaunayTriangulation.each_point(tri_vis)]

fig = Figure(fontsize = 24, size = (1500, 450))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Numerical", aspect = DataAspect())
tricontourf!(ax1, tri_vis, u_num, colormap = :viridis)
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = "y", title = "Exact (Barenblatt)", aspect = DataAspect())
tricontourf!(ax2, tri_vis, u_exact, colormap = :viridis)
ax3 = Axis(fig[1, 3], xlabel = "x", ylabel = "y", title = "Error", aspect = DataAspect())
tricontourf!(ax3, tri_vis, abs.(u_num .- u_exact), colormap = :hot)
resize_to_layout!(fig)
fig
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("barenblatt_pattle_solution.png"), fig)
end


if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "linf_errors" => errors_Linf,
            "l2_errors" => errors_L2,
            "min_l2_rate" => minimum(rates_L2),
            "finest_l2_error" => errors_L2[end],
        ),
        artifacts = ["barenblatt_pattle_solution.png"],
        notes = [
            "Canonical parabolic solve path via solve(prob, Tsit5()).",
            "This evidence entry is the parabolic benchmark-stage self-similar exact-solution case.",
        ],
        summary = Dict(
            "mesh_sizes" => mesh_sizes,
            "l2_rates" => rates_L2,
            "monotone_l2_decrease" => all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)),
        ),
    )
end
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


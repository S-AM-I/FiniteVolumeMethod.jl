```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/sod_shock_tube.jl"
```

````@example sod_shock_tube
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Sod Shock Tube
The Sod shock tube is the classic test problem for compressible flow solvers.
It consists of a 1D Riemann problem with a high-pressure region on the left
and a low-pressure region on the right, separated by a diaphragm at $x = 0.5$.
The resulting wave structure includes a left-moving rarefaction fan,
a right-moving contact discontinuity, and a right-moving shock wave.

## Problem Setup
The 1D Euler equations are:
```math
\pdv{}{t}\begin{pmatrix}\rho \\ \rho v \\ E\end{pmatrix} + \pdv{}{x}\begin{pmatrix}\rho v \\ \rho v^2 + P \\ (E+P)v\end{pmatrix} = 0,
```
with initial conditions:
```math
(\rho, v, P) = \begin{cases}(1, 0, 1) & x < 0.5,\\(0.125, 0, 0.1) & x \geq 0.5.\end{cases}
```

We begin by loading the package and defining the problem.

````@example sod_shock_tube
using FiniteVolumeMethod
using StaticArrays

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)
````

Define the left and right primitive states $(\rho, v, P)$:

````@example sod_shock_tube
wL = SVector(1.0, 0.0, 1.0)
wR = SVector(0.125, 0.0, 0.1)
````

Set up the mesh, boundary conditions, and initial condition:

````@example sod_shock_tube
N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)
bc_left = DirichletHyperbolicBC(wL)
bc_right = DirichletHyperbolicBC(wR)
````

The initial condition places the diaphragm at $x = 0.5$:

````@example sod_shock_tube
ic(x) = x < 0.5 ? wL : wR
````

## Solving with HLLC + MUSCL
We use the HLLC Riemann solver (which resolves the contact discontinuity)
with MUSCL reconstruction using the minmod limiter for second-order accuracy.

````@example sod_shock_tube
prob_hllc = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    bc_left, bc_right, ic;
    final_time = 0.2, cfl = 0.5
)
x_hllc, U_hllc, t_hllc = solve_hyperbolic(prob_hllc)
x_hllc |> tc #hide
````

## Solving with HLL + MUSCL
For comparison, we also solve with the HLL solver, which does not
resolve the contact discontinuity and is therefore more diffusive.

````@example sod_shock_tube
prob_hll = HyperbolicProblem(
    law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    bc_left, bc_right, ic;
    final_time = 0.2, cfl = 0.5
)
x_hll, U_hll, t_hll = solve_hyperbolic(prob_hll)
````

## Exact Solution
The exact Riemann solution for the Sod problem consists of five constant
states separated by a rarefaction, contact, and shock. The pre-computed
star-region values are:

````@example sod_shock_tube
function sod_exact(x, t; x0 = 0.5, gamma = 1.4)
    rhoL, vL, PL = 1.0, 0.0, 1.0
    rhoR, vR, PR = 0.125, 0.0, 0.1
    cL = sqrt(gamma * PL / rhoL)
    # Pre-computed star-region values
    P_star = 0.30313017805064707
    v_star = 0.92745262004895057
    rho_star_L = 0.42631942817849544
    rho_star_R = 0.26557371170530708
    c_star_L = sqrt(gamma * P_star / rho_star_L)
    # Shock speed
    S_shock = v_star + 1.0 / (gamma * rho_star_R) *
        (P_star - PR) / (v_star - vR + 1.0e-30)
    xi = (x - x0) / t
    if xi < -cL
        return rhoL, vL, PL
    elseif xi < v_star - c_star_L
        # Inside rarefaction fan
        v_fan = 2.0 / (gamma + 1) * (cL + xi)
        c_fan = cL - 0.5 * (gamma - 1) * v_fan
        rho_fan = rhoL * (c_fan / cL)^(2.0 / (gamma - 1))
        P_fan = PL * (rho_fan / rhoL)^gamma
        return rho_fan, v_fan, P_fan
    elseif xi < v_star
        return rho_star_L, v_star, P_star
    elseif xi < S_shock
        return rho_star_R, v_star, P_star
    else
        return rhoR, vR, PR
    end
end
````

## Visualisation
We now compare the numerical solutions against the exact solution.

````@example sod_shock_tube
using CairoMakie

x_exact = range(0.0, 1.0, length = 1000)
rho_exact = [sod_exact(xi, 0.2)[1] for xi in x_exact]
v_exact = [sod_exact(xi, 0.2)[2] for xi in x_exact]
P_exact = [sod_exact(xi, 0.2)[3] for xi in x_exact]

# Extract primitive variables from the numerical solutions
rho_hllc = [conserved_to_primitive(law, U_hllc[i])[1] for i in eachindex(U_hllc)]
v_hllc = [conserved_to_primitive(law, U_hllc[i])[2] for i in eachindex(U_hllc)]
P_hllc = [conserved_to_primitive(law, U_hllc[i])[3] for i in eachindex(U_hllc)]

rho_hll = [conserved_to_primitive(law, U_hll[i])[1] for i in eachindex(U_hll)]
v_hll = [conserved_to_primitive(law, U_hll[i])[2] for i in eachindex(U_hll)]
P_hll = [conserved_to_primitive(law, U_hll[i])[3] for i in eachindex(U_hll)]

fig = Figure(fontsize = 24, size = (1200, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho", title = "Density")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = "v", title = "Velocity")
ax3 = Axis(fig[1, 3], xlabel = "x", ylabel = "P", title = "Pressure")
lines!(ax1, x_exact, rho_exact, color = :black, linewidth = 2, label = "Exact")
scatter!(ax1, x_hllc, rho_hllc, color = :blue, markersize = 4, label = "HLLC")
scatter!(ax1, x_hll, rho_hll, color = :red, markersize = 4, label = "HLL")
lines!(ax2, x_exact, v_exact, color = :black, linewidth = 2)
scatter!(ax2, x_hllc, v_hllc, color = :blue, markersize = 4)
scatter!(ax2, x_hll, v_hll, color = :red, markersize = 4)
lines!(ax3, x_exact, P_exact, color = :black, linewidth = 2)
scatter!(ax3, x_hllc, P_hllc, color = :blue, markersize = 4)
scatter!(ax3, x_hll, P_hll, color = :red, markersize = 4)
axislegend(ax1, position = :cb)
resize_to_layout!(fig)
fig
````

Notice that the HLLC solver (blue) resolves the contact discontinuity
much more sharply than HLL (red), which smears it out. Both solvers
capture the shock and rarefaction well at this resolution ($N = 200$).

## Convergence
We can verify that the error decreases with resolution:

````@example sod_shock_tube
function compute_l1_error(N)
    m = StructuredMesh1D(0.0, 1.0, N)
    p = HyperbolicProblem(
        law, m, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        bc_left, bc_right, ic; final_time = 0.2, cfl = 0.5
    )
    xx, UU, _ = solve_hyperbolic(p)
    err = 0.0
    for i in eachindex(xx)
        rho_num = conserved_to_primitive(law, UU[i])[1]
        rho_ex = sod_exact(xx[i], 0.2)[1]
        err += abs(rho_num - rho_ex)
    end
    return err / N
end

err_100 = compute_l1_error(100)
err_200 = compute_l1_error(200)
err_400 = compute_l1_error(400)
@assert err_400 < err_100 #hide
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/sod_shock_tube.jl).

```julia
using FiniteVolumeMethod
using StaticArrays

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

wL = SVector(1.0, 0.0, 1.0)
wR = SVector(0.125, 0.0, 0.1)

N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)
bc_left = DirichletHyperbolicBC(wL)
bc_right = DirichletHyperbolicBC(wR)

ic(x) = x < 0.5 ? wL : wR

prob_hllc = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    bc_left, bc_right, ic;
    final_time = 0.2, cfl = 0.5
)
x_hllc, U_hllc, t_hllc = solve_hyperbolic(prob_hllc)

prob_hll = HyperbolicProblem(
    law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    bc_left, bc_right, ic;
    final_time = 0.2, cfl = 0.5
)
x_hll, U_hll, t_hll = solve_hyperbolic(prob_hll)

function sod_exact(x, t; x0 = 0.5, gamma = 1.4)
    rhoL, vL, PL = 1.0, 0.0, 1.0
    rhoR, vR, PR = 0.125, 0.0, 0.1
    cL = sqrt(gamma * PL / rhoL)
    # Pre-computed star-region values
    P_star = 0.30313017805064707
    v_star = 0.92745262004895057
    rho_star_L = 0.42631942817849544
    rho_star_R = 0.26557371170530708
    c_star_L = sqrt(gamma * P_star / rho_star_L)
    # Shock speed
    S_shock = v_star + 1.0 / (gamma * rho_star_R) *
        (P_star - PR) / (v_star - vR + 1.0e-30)
    xi = (x - x0) / t
    if xi < -cL
        return rhoL, vL, PL
    elseif xi < v_star - c_star_L
        # Inside rarefaction fan
        v_fan = 2.0 / (gamma + 1) * (cL + xi)
        c_fan = cL - 0.5 * (gamma - 1) * v_fan
        rho_fan = rhoL * (c_fan / cL)^(2.0 / (gamma - 1))
        P_fan = PL * (rho_fan / rhoL)^gamma
        return rho_fan, v_fan, P_fan
    elseif xi < v_star
        return rho_star_L, v_star, P_star
    elseif xi < S_shock
        return rho_star_R, v_star, P_star
    else
        return rhoR, vR, PR
    end
end

using CairoMakie

x_exact = range(0.0, 1.0, length = 1000)
rho_exact = [sod_exact(xi, 0.2)[1] for xi in x_exact]
v_exact = [sod_exact(xi, 0.2)[2] for xi in x_exact]
P_exact = [sod_exact(xi, 0.2)[3] for xi in x_exact]

# Extract primitive variables from the numerical solutions
rho_hllc = [conserved_to_primitive(law, U_hllc[i])[1] for i in eachindex(U_hllc)]
v_hllc = [conserved_to_primitive(law, U_hllc[i])[2] for i in eachindex(U_hllc)]
P_hllc = [conserved_to_primitive(law, U_hllc[i])[3] for i in eachindex(U_hllc)]

rho_hll = [conserved_to_primitive(law, U_hll[i])[1] for i in eachindex(U_hll)]
v_hll = [conserved_to_primitive(law, U_hll[i])[2] for i in eachindex(U_hll)]
P_hll = [conserved_to_primitive(law, U_hll[i])[3] for i in eachindex(U_hll)]

fig = Figure(fontsize = 24, size = (1200, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho", title = "Density")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = "v", title = "Velocity")
ax3 = Axis(fig[1, 3], xlabel = "x", ylabel = "P", title = "Pressure")
lines!(ax1, x_exact, rho_exact, color = :black, linewidth = 2, label = "Exact")
scatter!(ax1, x_hllc, rho_hllc, color = :blue, markersize = 4, label = "HLLC")
scatter!(ax1, x_hll, rho_hll, color = :red, markersize = 4, label = "HLL")
lines!(ax2, x_exact, v_exact, color = :black, linewidth = 2)
scatter!(ax2, x_hllc, v_hllc, color = :blue, markersize = 4)
scatter!(ax2, x_hll, v_hll, color = :red, markersize = 4)
lines!(ax3, x_exact, P_exact, color = :black, linewidth = 2)
scatter!(ax3, x_hllc, P_hllc, color = :blue, markersize = 4)
scatter!(ax3, x_hll, P_hll, color = :red, markersize = 4)
axislegend(ax1, position = :cb)
resize_to_layout!(fig)
fig

function compute_l1_error(N)
    m = StructuredMesh1D(0.0, 1.0, N)
    p = HyperbolicProblem(
        law, m, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        bc_left, bc_right, ic; final_time = 0.2, cfl = 0.5
    )
    xx, UU, _ = solve_hyperbolic(p)
    err = 0.0
    for i in eachindex(xx)
        rho_num = conserved_to_primitive(law, UU[i])[1]
        rho_ex = sod_exact(xx[i], 0.2)[1]
        err += abs(rho_num - rho_ex)
    end
    return err / N
end

err_100 = compute_l1_error(100)
err_200 = compute_l1_error(200)
err_400 = compute_l1_error(400)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


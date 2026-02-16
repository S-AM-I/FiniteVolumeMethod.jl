```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/imex_stiff_relaxation.jl"
```

````@example imex_stiff_relaxation
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# IMEX Stiff Relaxation
This tutorial demonstrates the implicit-explicit (IMEX) time integration
scheme for problems with stiff source terms. We consider the 1D Euler
equations with a stiff radiative cooling source that drives the gas
toward an equilibrium temperature on a time scale much shorter than
the CFL time step.

## Problem Setup
The governing equations are:
```math
\pdv{\vb U}{t} + \pdv{\vb F}{x} = \vb S_{\mathrm{stiff}}(\vb U),
```
where the source term represents optically thin radiative cooling:
$S_E = -\rho^2 \Lambda(T)$ with $\Lambda(T) = \lambda(T - T_{\mathrm{target}})$.
When $\lambda \gg 1$, the cooling is stiff and an explicit time integrator
would require impractically small time steps.

````@example imex_stiff_relaxation
using FiniteVolumeMethod
using StaticArrays

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

# Physical parameters
rho_init = 1.0
v_init = 0.0
P_target = 1.0     ## equilibrium pressure
P_init = 3.0       ## initial pressure (above equilibrium)
mu_mol = 1.0       ## mean molecular weight
lambda_rate = 50.0  ## stiff cooling rate
````

## Cooling Source
The `CoolingSource` takes a function $\Lambda(T)$ and the mean
molecular weight $\mu$. The temperature is computed as $T = P\mu/\rho$.

````@example imex_stiff_relaxation
T_target = P_target * mu_mol
cooling_func = T -> lambda_rate * (T - T_target)
source = CoolingSource(cooling_func; mu_mol = mu_mol)

w_init = SVector(rho_init, v_init, P_init)
````

## Solving with Different IMEX Schemes
We compare three IMEX Runge-Kutta schemes:

````@example imex_stiff_relaxation
N = 32
mesh = StructuredMesh1D(0.0, 1.0, N)
t_final = 0.05

prob = HyperbolicProblem(
    law, mesh, HLLSolver(), NoReconstruction(),
    TransmissiveBC(), TransmissiveBC(),
    x -> w_init;
    final_time = t_final, cfl = 0.4
)
````

**SSP3(4,3,3)** — 4-stage, 3rd-order SSP scheme:

````@example imex_stiff_relaxation
x_ssp, U_ssp, t_ssp = solve_hyperbolic_imex(
    prob, source; scheme = IMEX_SSP3_433(),
    newton_tol = 1.0e-12, newton_maxiter = 10
)
````

**ARS(2,2,2)** — 3-stage, 2nd-order L-stable scheme:

````@example imex_stiff_relaxation
x_ars, U_ars, t_ars = solve_hyperbolic_imex(
    prob, source; scheme = IMEX_ARS222(),
    newton_tol = 1.0e-12, newton_maxiter = 10
)
x_ars |> tc #hide
````

## Checking Relaxation
The pressure should relax toward $P_{\mathrm{target}} = 1.0$ from
the initial $P_{\mathrm{init}} = 3.0$:

````@example imex_stiff_relaxation
P_ssp = [conserved_to_primitive(law, U_ssp[i])[3] for i in eachindex(U_ssp)]
P_ars = [conserved_to_primitive(law, U_ars[i])[3] for i in eachindex(U_ars)]

# The pressure should be closer to P_target than P_init
P_avg_ssp = sum(P_ssp) / length(P_ssp)
P_avg_ars = sum(P_ars) / length(P_ars)
@assert abs(P_avg_ssp - P_target) < abs(P_init - P_target) #hide
@assert abs(P_avg_ars - P_target) < abs(P_init - P_target) #hide
````

## Visualisation

````@example imex_stiff_relaxation
using CairoMakie

fig = Figure(fontsize = 24, size = (900, 400))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "P",
    title = "Pressure relaxation"
)
scatter!(ax1, x_ssp, P_ssp, color = :blue, markersize = 6, label = "SSP3(4,3,3)")
scatter!(ax1, x_ars, P_ars, color = :red, markersize = 6, label = "ARS(2,2,2)")
hlines!(ax1, [P_target], color = :black, linestyle = :dash, label = L"P_{\mathrm{target}}")
hlines!(ax1, [P_init], color = :gray, linestyle = :dot, label = L"P_{\mathrm{init}}")
axislegend(ax1, position = :rt)

# Also check that density is preserved
rho_ssp = [conserved_to_primitive(law, U_ssp[i])[1] for i in eachindex(U_ssp)]
ax2 = Axis(
    fig[1, 2], xlabel = "x", ylabel = L"\rho",
    title = "Density (should be constant)"
)
scatter!(ax2, x_ssp, rho_ssp, color = :blue, markersize = 6)
hlines!(ax2, [rho_init], color = :black, linestyle = :dash)

resize_to_layout!(fig)
fig
````

The pressure relaxes from $P = 3$ toward the equilibrium $P = 1$,
while the density remains constant (the cooling source only affects
energy). The IMEX scheme handles the stiff source implicitly,
allowing stable time steps determined by the CFL condition rather
than the fast cooling time scale.

````@example imex_stiff_relaxation
rho_variation = maximum(rho_ssp) - minimum(rho_ssp) #hide
@assert rho_variation < 0.05 * rho_init #hide
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/imex_stiff_relaxation.jl).

```julia
using FiniteVolumeMethod
using StaticArrays

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

# Physical parameters
rho_init = 1.0
v_init = 0.0
P_target = 1.0     ## equilibrium pressure
P_init = 3.0       ## initial pressure (above equilibrium)
mu_mol = 1.0       ## mean molecular weight
lambda_rate = 50.0  ## stiff cooling rate

T_target = P_target * mu_mol
cooling_func = T -> lambda_rate * (T - T_target)
source = CoolingSource(cooling_func; mu_mol = mu_mol)

w_init = SVector(rho_init, v_init, P_init)

N = 32
mesh = StructuredMesh1D(0.0, 1.0, N)
t_final = 0.05

prob = HyperbolicProblem(
    law, mesh, HLLSolver(), NoReconstruction(),
    TransmissiveBC(), TransmissiveBC(),
    x -> w_init;
    final_time = t_final, cfl = 0.4
)

x_ssp, U_ssp, t_ssp = solve_hyperbolic_imex(
    prob, source; scheme = IMEX_SSP3_433(),
    newton_tol = 1.0e-12, newton_maxiter = 10
)

x_ars, U_ars, t_ars = solve_hyperbolic_imex(
    prob, source; scheme = IMEX_ARS222(),
    newton_tol = 1.0e-12, newton_maxiter = 10
)

P_ssp = [conserved_to_primitive(law, U_ssp[i])[3] for i in eachindex(U_ssp)]
P_ars = [conserved_to_primitive(law, U_ars[i])[3] for i in eachindex(U_ars)]

# The pressure should be closer to P_target than P_init
P_avg_ssp = sum(P_ssp) / length(P_ssp)
P_avg_ars = sum(P_ars) / length(P_ars)

using CairoMakie

fig = Figure(fontsize = 24, size = (900, 400))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "P",
    title = "Pressure relaxation"
)
scatter!(ax1, x_ssp, P_ssp, color = :blue, markersize = 6, label = "SSP3(4,3,3)")
scatter!(ax1, x_ars, P_ars, color = :red, markersize = 6, label = "ARS(2,2,2)")
hlines!(ax1, [P_target], color = :black, linestyle = :dash, label = L"P_{\mathrm{target}}")
hlines!(ax1, [P_init], color = :gray, linestyle = :dot, label = L"P_{\mathrm{init}}")
axislegend(ax1, position = :rt)

# Also check that density is preserved
rho_ssp = [conserved_to_primitive(law, U_ssp[i])[1] for i in eachindex(U_ssp)]
ax2 = Axis(
    fig[1, 2], xlabel = "x", ylabel = L"\rho",
    title = "Density (should be constant)"
)
scatter!(ax2, x_ssp, rho_ssp, color = :blue, markersize = 6)
hlines!(ax2, [rho_init], color = :black, linestyle = :dash)

resize_to_layout!(fig)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


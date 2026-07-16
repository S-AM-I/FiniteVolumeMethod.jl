```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/imex_stiff_relaxation.jl"
```

````julia
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

````julia
using FiniteVolumeMethod
using ADTypes: AutoFiniteDiff
using OrdinaryDiffEqSDIRK: KenCarp3, KenCarp47
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

````julia
T_target = P_target * mu_mol
cooling_func = T -> lambda_rate * (T - T_target)
source = CoolingSource(cooling_func; mu_mol = mu_mol)

w_init = SVector(rho_init, v_init, P_init)
````

## Solving with Different IMEX Schemes
The stiff source enters through a `SplitODEProblem` (explicit hyperbolic
fluxes plus implicit source), built by `sciml_problem(prob, source)` and
integrated with additive Runge-Kutta (IMEX) schemes. We compare two such
schemes:

````julia
N = 32
mesh = StructuredMesh1D(0.0, 1.0, N)
t_final = 0.05

prob = HyperbolicProblem(
    law, mesh, HLLSolver(), NoReconstruction(),
    TransmissiveBC(), TransmissiveBC(),
    x -> w_init;
    final_time = t_final, cfl = 0.4
)

split_prob = sciml_problem(prob, source)
dt0 = compute_initial_dt(split_prob.p, split_prob.u0)
acc = solution_accessor(prob)
````

**KenCarp47** — 7-stage, 4th-order ARK scheme:

````julia
sol_kc47 = solve(
    split_prob, KenCarp47(autodiff = AutoFiniteDiff());
    adaptive = false, dt = dt0
)
U_kc47 = get_conserved(acc, sol_kc47, length(sol_kc47.t))
x_kc47 = get_coordinates(acc)
t_kc47 = sol_kc47.t[end]
````

**KenCarp3** — 4-stage, 3rd-order L-stable ARK scheme:

````julia
sol_kc3 = solve(
    split_prob, KenCarp3(autodiff = AutoFiniteDiff());
    adaptive = false, dt = dt0
)
U_kc3 = get_conserved(acc, sol_kc3, length(sol_kc3.t))
x_kc3 = get_coordinates(acc)
t_kc3 = sol_kc3.t[end]
x_kc3 |> tc #hide
````

## Checking Relaxation
The pressure should relax toward $P_{\mathrm{target}} = 1.0$ from
the initial $P_{\mathrm{init}} = 3.0$:

````julia
P_kc47 = [conserved_to_primitive(law, U_kc47[i])[3] for i in eachindex(U_kc47)]
P_kc3 = [conserved_to_primitive(law, U_kc3[i])[3] for i in eachindex(U_kc3)]

# The pressure should be closer to P_target than P_init
P_avg_kc47 = sum(P_kc47) / length(P_kc47)
P_avg_kc3 = sum(P_kc3) / length(P_kc3)
abs(P_avg_kc47 - P_target) < abs(P_init - P_target) || @warn("KenCarp47 pressure did not relax toward target") #hide
abs(P_avg_kc3 - P_target) < abs(P_init - P_target) || @warn("KenCarp3 pressure did not relax toward target") #hide
````

## Visualisation

````julia
using CairoMakie

fig = Figure(fontsize = 24, size = (900, 400))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "P",
    title = "Pressure relaxation"
)
scatter!(ax1, x_kc47, P_kc47, color = :blue, markersize = 6, label = "KenCarp47")
scatter!(ax1, x_kc3, P_kc3, color = :red, markersize = 6, label = "KenCarp3")
hlines!(ax1, [P_target], color = :black, linestyle = :dash, label = L"P_{\mathrm{target}}")
hlines!(ax1, [P_init], color = :gray, linestyle = :dot, label = L"P_{\mathrm{init}}")
axislegend(ax1, position = :rt)

# Also check that density is preserved
rho_kc47 = [conserved_to_primitive(law, U_kc47[i])[1] for i in eachindex(U_kc47)]
ax2 = Axis(
    fig[1, 2], xlabel = "x", ylabel = L"\rho",
    title = "Density (should be constant)"
)
scatter!(ax2, x_kc47, rho_kc47, color = :blue, markersize = 6)
hlines!(ax2, [rho_init], color = :black, linestyle = :dash)

resize_to_layout!(fig)
fig
````

The pressure relaxes from $P = 3$ toward the equilibrium $P = 1$,
while the density remains constant (the cooling source only affects
energy). The IMEX scheme handles the stiff source implicitly,
allowing stable time steps determined by the CFL condition rather
than the fast cooling time scale.

````julia
rho_variation = maximum(rho_kc47) - minimum(rho_kc47) #hide
rho_variation < 0.05 * rho_init || @warn("Density variation exceeds tolerance: $rho_variation") #hide
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/imex_stiff_relaxation.jl).

```julia
using FiniteVolumeMethod
using ADTypes: AutoFiniteDiff
using OrdinaryDiffEqSDIRK: KenCarp3, KenCarp47
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

split_prob = sciml_problem(prob, source)
dt0 = compute_initial_dt(split_prob.p, split_prob.u0)
acc = solution_accessor(prob)

sol_kc47 = solve(
    split_prob, KenCarp47(autodiff = AutoFiniteDiff());
    adaptive = false, dt = dt0
)
U_kc47 = get_conserved(acc, sol_kc47, length(sol_kc47.t))
x_kc47 = get_coordinates(acc)
t_kc47 = sol_kc47.t[end]

sol_kc3 = solve(
    split_prob, KenCarp3(autodiff = AutoFiniteDiff());
    adaptive = false, dt = dt0
)
U_kc3 = get_conserved(acc, sol_kc3, length(sol_kc3.t))
x_kc3 = get_coordinates(acc)
t_kc3 = sol_kc3.t[end]

P_kc47 = [conserved_to_primitive(law, U_kc47[i])[3] for i in eachindex(U_kc47)]
P_kc3 = [conserved_to_primitive(law, U_kc3[i])[3] for i in eachindex(U_kc3)]

# The pressure should be closer to P_target than P_init
P_avg_kc47 = sum(P_kc47) / length(P_kc47)
P_avg_kc3 = sum(P_kc3) / length(P_kc3)

using CairoMakie

fig = Figure(fontsize = 24, size = (900, 400))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "P",
    title = "Pressure relaxation"
)
scatter!(ax1, x_kc47, P_kc47, color = :blue, markersize = 6, label = "KenCarp47")
scatter!(ax1, x_kc3, P_kc3, color = :red, markersize = 6, label = "KenCarp3")
hlines!(ax1, [P_target], color = :black, linestyle = :dash, label = L"P_{\mathrm{target}}")
hlines!(ax1, [P_init], color = :gray, linestyle = :dot, label = L"P_{\mathrm{init}}")
axislegend(ax1, position = :rt)

# Also check that density is preserved
rho_kc47 = [conserved_to_primitive(law, U_kc47[i])[1] for i in eachindex(U_kc47)]
ax2 = Axis(
    fig[1, 2], xlabel = "x", ylabel = L"\rho",
    title = "Density (should be constant)"
)
scatter!(ax2, x_kc47, rho_kc47, color = :blue, markersize = 6)
hlines!(ax2, [rho_init], color = :black, linestyle = :dash)

resize_to_layout!(fig)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


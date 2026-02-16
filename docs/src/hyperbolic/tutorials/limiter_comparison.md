```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/limiter_comparison.jl"
```

````@example limiter_comparison
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Limiter Comparison
MUSCL reconstruction achieves second-order accuracy by extrapolating
cell-centred values to face centres. The **slope limiter** controls how
aggressively this extrapolation is performed near discontinuities.
A more aggressive limiter resolves smooth features better but can
introduce oscillations near shocks; a more dissipative limiter is
robust but smears sharp features.

This tutorial compares all five available limiters on the Sod shock
tube to visualise these trade-offs.

## Problem Setup
We use the standard 1D Sod shock tube with $N = 200$ cells:

````@example limiter_comparison
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
````

## Running with Each Limiter
We solve the same problem with all five limiters: Minmod (most
dissipative), Superbee (least dissipative), Van Leer, Koren, and
Ospre.

````@example limiter_comparison
limiters = [
    ("Minmod", MinmodLimiter()),
    ("Superbee", SuperbeeLimiter()),
    ("Van Leer", VanLeerLimiter()),
    ("Koren", KorenLimiter()),
    ("Ospre", OspreLimiter()),
]

results = map(limiters) do (name, lim)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(lim),
        bc_left, bc_right, ic;
        final_time = 0.2, cfl = 0.5
    )
    x, U, t = solve_hyperbolic(prob)
    rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
    (name = name, x = x, rho = rho)
end
````

We also include a first-order (no reconstruction) baseline:

````@example limiter_comparison
prob_first = HyperbolicProblem(
    law, mesh, HLLCSolver(), NoReconstruction(),
    bc_left, bc_right, ic;
    final_time = 0.2, cfl = 0.5
)
x_first, U_first, _ = solve_hyperbolic(prob_first)
rho_first = [conserved_to_primitive(law, U_first[i])[1] for i in eachindex(U_first)]
````

## Exact Solution

````@example limiter_comparison
function sod_exact(x, t; x0 = 0.5, gamma = 1.4)
    cL = sqrt(gamma * 1.0 / 1.0)
    P_star = 0.30313017805064707
    v_star = 0.92745262004895057
    rho_star_L = 0.42631942817849544
    rho_star_R = 0.26557371170530708
    c_star_L = sqrt(gamma * P_star / rho_star_L)
    S_shock = v_star + 1.0 / (gamma * rho_star_R) *
        (P_star - 0.1) / (v_star + 1.0e-30)
    xi = (x - x0) / t
    if xi < -cL
        return 1.0
    elseif xi < v_star - c_star_L
        v_fan = 2.0 / (gamma + 1) * (cL + xi)
        c_fan = cL - 0.5 * (gamma - 1) * v_fan
        return 1.0 * (c_fan / cL)^(2.0 / (gamma - 1))
    elseif xi < v_star
        return rho_star_L
    elseif xi < S_shock
        return rho_star_R
    else
        return 0.125
    end
end
````

## Visualisation

````@example limiter_comparison
using CairoMakie

x_exact = range(0.0, 1.0, length = 1000)
rho_exact = [sod_exact(xi, 0.2) for xi in x_exact]

colors = [:blue, :red, :green, :orange, :purple]

fig = Figure(fontsize = 20, size = (1200, 800))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho", title = "Full Domain")
lines!(ax1, x_exact, rho_exact, color = :black, linewidth = 2, label = "Exact")
scatter!(ax1, x_first, rho_first, color = :gray, markersize = 3, label = "1st order")
for (i, r) in enumerate(results)
    scatter!(ax1, r.x, r.rho, color = colors[i], markersize = 3, label = r.name)
end
axislegend(ax1, position = :cb)
````

Zoom into the contact discontinuity region:

````@example limiter_comparison
ax2 = Axis(
    fig[2, 1], xlabel = "x", ylabel = L"\rho",
    title = "Contact Discontinuity (zoom)"
)
lines!(ax2, x_exact, rho_exact, color = :black, linewidth = 2)
scatter!(ax2, x_first, rho_first, color = :gray, markersize = 5, label = "1st order")
for (i, r) in enumerate(results)
    scatter!(ax2, r.x, r.rho, color = colors[i], markersize = 5, label = r.name)
end
xlims!(ax2, 0.55, 0.85)
ylims!(ax2, 0.15, 0.5)
axislegend(ax2, position = :rt)

resize_to_layout!(fig)
fig
````

## Observations
- **Superbee** resolves the contact discontinuity most sharply, but can
  produce slight overshoots near the shock.
- **Minmod** is the most dissipative, producing the smoothest profiles.
- **Van Leer**, **Koren**, and **Ospre** offer a middle ground between
  sharpness and robustness.
- The **first-order** solution (no reconstruction) is very diffusive,
  illustrating why MUSCL reconstruction is important.

````@example limiter_comparison
@assert all(r -> all(isfinite, r.rho), results) #hide
@assert all(r -> all(>(0), r.rho), results) #hide
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/limiter_comparison.jl).

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

limiters = [
    ("Minmod", MinmodLimiter()),
    ("Superbee", SuperbeeLimiter()),
    ("Van Leer", VanLeerLimiter()),
    ("Koren", KorenLimiter()),
    ("Ospre", OspreLimiter()),
]

results = map(limiters) do (name, lim)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(lim),
        bc_left, bc_right, ic;
        final_time = 0.2, cfl = 0.5
    )
    x, U, t = solve_hyperbolic(prob)
    rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
    (name = name, x = x, rho = rho)
end

prob_first = HyperbolicProblem(
    law, mesh, HLLCSolver(), NoReconstruction(),
    bc_left, bc_right, ic;
    final_time = 0.2, cfl = 0.5
)
x_first, U_first, _ = solve_hyperbolic(prob_first)
rho_first = [conserved_to_primitive(law, U_first[i])[1] for i in eachindex(U_first)]

function sod_exact(x, t; x0 = 0.5, gamma = 1.4)
    cL = sqrt(gamma * 1.0 / 1.0)
    P_star = 0.30313017805064707
    v_star = 0.92745262004895057
    rho_star_L = 0.42631942817849544
    rho_star_R = 0.26557371170530708
    c_star_L = sqrt(gamma * P_star / rho_star_L)
    S_shock = v_star + 1.0 / (gamma * rho_star_R) *
        (P_star - 0.1) / (v_star + 1.0e-30)
    xi = (x - x0) / t
    if xi < -cL
        return 1.0
    elseif xi < v_star - c_star_L
        v_fan = 2.0 / (gamma + 1) * (cL + xi)
        c_fan = cL - 0.5 * (gamma - 1) * v_fan
        return 1.0 * (c_fan / cL)^(2.0 / (gamma - 1))
    elseif xi < v_star
        return rho_star_L
    elseif xi < S_shock
        return rho_star_R
    else
        return 0.125
    end
end

using CairoMakie

x_exact = range(0.0, 1.0, length = 1000)
rho_exact = [sod_exact(xi, 0.2) for xi in x_exact]

colors = [:blue, :red, :green, :orange, :purple]

fig = Figure(fontsize = 20, size = (1200, 800))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho", title = "Full Domain")
lines!(ax1, x_exact, rho_exact, color = :black, linewidth = 2, label = "Exact")
scatter!(ax1, x_first, rho_first, color = :gray, markersize = 3, label = "1st order")
for (i, r) in enumerate(results)
    scatter!(ax1, r.x, r.rho, color = colors[i], markersize = 3, label = r.name)
end
axislegend(ax1, position = :cb)

ax2 = Axis(
    fig[2, 1], xlabel = "x", ylabel = L"\rho",
    title = "Contact Discontinuity (zoom)"
)
lines!(ax2, x_exact, rho_exact, color = :black, linewidth = 2)
scatter!(ax2, x_first, rho_first, color = :gray, markersize = 5, label = "1st order")
for (i, r) in enumerate(results)
    scatter!(ax2, r.x, r.rho, color = colors[i], markersize = 5, label = r.name)
end
xlims!(ax2, 0.55, 0.85)
ylims!(ax2, 0.15, 0.5)
axislegend(ax2, position = :rt)

resize_to_layout!(fig)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


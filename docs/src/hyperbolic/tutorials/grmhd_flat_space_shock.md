```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/grmhd_flat_space_shock.jl"
```

````@example grmhd_flat_space_shock
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# GRMHD in Flat Spacetime
The general relativistic MHD (GRMHD) solver in
`FiniteVolumeMethod.jl` implements the Valencia formulation of the
relativistic MHD equations on a curved spacetime. In flat (Minkowski)
spacetime the geometric source terms vanish and the GRMHD equations
reduce exactly to the special relativistic MHD (SRMHD) equations.

This tutorial verifies that equivalence by solving the Balsara 1
shock tube with both SRMHD and GRMHD solvers and comparing the
density profiles.

## Problem Setup
The Balsara 1 initial data are:
```math
(\rho, v_x, v_y, v_z, P, B_x, B_y, B_z) = \begin{cases}
(1,\, 0,\, 0,\, 0,\, 1,\, 0.5,\, 1,\, 0) & x < 0.5,\\
(0.125,\, 0,\, 0,\, 0,\, 0.1,\, 0.5,\, -1,\, 0) & x \geq 0.5.
\end{cases}
```

````@example grmhd_flat_space_shock
using FiniteVolumeMethod
using StaticArrays

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
````

Create both the SRMHD and GRMHD conservation laws:

````@example grmhd_flat_space_shock
law_sr = SRMHDEquations{1}(eos)
law_gr = GRMHDEquations{1}(eos, MinkowskiMetric{1}())

wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)

N = 400
mesh = StructuredMesh1D(0.0, 1.0, N)
ic(x) = x < 0.5 ? wL : wR
````

## Solving with SRMHD

````@example grmhd_flat_space_shock
prob_sr = HyperbolicProblem(
    law_sr, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.4, cfl = 0.4
)
x_sr, U_sr, t_sr = solve_hyperbolic(prob_sr)
x_sr |> tc #hide
````

## Solving with GRMHD (Minkowski)

````@example grmhd_flat_space_shock
prob_gr = HyperbolicProblem(
    law_gr, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.4, cfl = 0.4
)
x_gr, U_gr, t_gr = solve_hyperbolic(prob_gr)
````

## Comparing Results

````@example grmhd_flat_space_shock
rho_sr = [conserved_to_primitive(law_sr, U_sr[i])[1] for i in eachindex(U_sr)]
vx_sr = [conserved_to_primitive(law_sr, U_sr[i])[2] for i in eachindex(U_sr)]
By_sr = [conserved_to_primitive(law_sr, U_sr[i])[7] for i in eachindex(U_sr)]

rho_gr = [conserved_to_primitive(law_gr, U_gr[i])[1] for i in eachindex(U_gr)]
vx_gr = [conserved_to_primitive(law_gr, U_gr[i])[2] for i in eachindex(U_gr)]
By_gr = [conserved_to_primitive(law_gr, U_gr[i])[7] for i in eachindex(U_gr)]
````

The two solutions should match to high accuracy:

````@example grmhd_flat_space_shock
max_rho_diff = maximum(abs.(rho_sr .- rho_gr))
@assert max_rho_diff < 1.0e-10 #hide
````

## Visualisation

````@example grmhd_flat_space_shock
using CairoMakie

fig = Figure(fontsize = 20, size = (1200, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho", title = "Density")
lines!(ax1, x_sr, rho_sr, color = :blue, linewidth = 2, label = "SRMHD")
scatter!(ax1, x_gr, rho_gr, color = :red, markersize = 3, label = "GRMHD (Minkowski)")
axislegend(ax1, position = :ct)

ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = L"v_x", title = "Velocity")
lines!(ax2, x_sr, vx_sr, color = :blue, linewidth = 2)
scatter!(ax2, x_gr, vx_gr, color = :red, markersize = 3)

ax3 = Axis(fig[1, 3], xlabel = "x", ylabel = L"B_y", title = "Magnetic Field (y)")
lines!(ax3, x_sr, By_sr, color = :blue, linewidth = 2)
scatter!(ax3, x_gr, By_gr, color = :red, markersize = 3)

resize_to_layout!(fig)
fig
````

The SRMHD (blue lines) and GRMHD (red dots) solutions overlap
perfectly, confirming that the GRMHD solver correctly reduces to
SRMHD in flat spacetime. The maximum density difference is
$(round(max_rho_diff, sigdigits = 2)).

````@example grmhd_flat_space_shock
@assert all(rho_sr .> 0) #hide
@assert all(rho_gr .> 0) #hide
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/grmhd_flat_space_shock.jl).

```julia
using FiniteVolumeMethod
using StaticArrays

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)

law_sr = SRMHDEquations{1}(eos)
law_gr = GRMHDEquations{1}(eos, MinkowskiMetric{1}())

wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)

N = 400
mesh = StructuredMesh1D(0.0, 1.0, N)
ic(x) = x < 0.5 ? wL : wR

prob_sr = HyperbolicProblem(
    law_sr, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.4, cfl = 0.4
)
x_sr, U_sr, t_sr = solve_hyperbolic(prob_sr)

prob_gr = HyperbolicProblem(
    law_gr, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.4, cfl = 0.4
)
x_gr, U_gr, t_gr = solve_hyperbolic(prob_gr)

rho_sr = [conserved_to_primitive(law_sr, U_sr[i])[1] for i in eachindex(U_sr)]
vx_sr = [conserved_to_primitive(law_sr, U_sr[i])[2] for i in eachindex(U_sr)]
By_sr = [conserved_to_primitive(law_sr, U_sr[i])[7] for i in eachindex(U_sr)]

rho_gr = [conserved_to_primitive(law_gr, U_gr[i])[1] for i in eachindex(U_gr)]
vx_gr = [conserved_to_primitive(law_gr, U_gr[i])[2] for i in eachindex(U_gr)]
By_gr = [conserved_to_primitive(law_gr, U_gr[i])[7] for i in eachindex(U_gr)]

max_rho_diff = maximum(abs.(rho_sr .- rho_gr))

using CairoMakie

fig = Figure(fontsize = 20, size = (1200, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho", title = "Density")
lines!(ax1, x_sr, rho_sr, color = :blue, linewidth = 2, label = "SRMHD")
scatter!(ax1, x_gr, rho_gr, color = :red, markersize = 3, label = "GRMHD (Minkowski)")
axislegend(ax1, position = :ct)

ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = L"v_x", title = "Velocity")
lines!(ax2, x_sr, vx_sr, color = :blue, linewidth = 2)
scatter!(ax2, x_gr, vx_gr, color = :red, markersize = 3)

ax3 = Axis(fig[1, 3], xlabel = "x", ylabel = L"B_y", title = "Magnetic Field (y)")
lines!(ax3, x_sr, By_sr, color = :blue, linewidth = 2)
scatter!(ax3, x_gr, By_gr, color = :red, markersize = 3)

resize_to_layout!(fig)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


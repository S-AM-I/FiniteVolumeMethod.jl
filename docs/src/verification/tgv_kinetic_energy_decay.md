```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/tgv_kinetic_energy_decay.jl"
```

````@example tgv_kinetic_energy_decay
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Taylor-Green Vortex — Kinetic Energy Decay
This example extends the Navier-Stokes Taylor-Green vortex convergence
test by explicitly tracking kinetic energy decay over time and comparing
against the analytical exponential decay law.

## Mathematical Setup
The Taylor-Green vortex velocity decays as:
```math
\text{KE}(t) = \text{KE}_0 \exp(-4\nu k^2 t)
```
where $\nu = \mu/\rho_0$ and $k = 2\pi/L$. We measure the relative
error in the KE decay rate and perform a grid convergence study.

## Reference
- Taylor, G.I. & Green, A.E. (1937). Mechanism of the Production of
  Small Eddies from Large Ones. Proc. R. Soc. A, 158, 499-521.

````@example tgv_kinetic_energy_decay
using FiniteVolumeMethod
using StaticArrays
using CairoMakie

rho0 = 1.0
P0 = 100.0     # high pressure → low Mach
U0 = 0.01      # small velocity amplitude
L = 1.0
k = 2 * pi / L
mu = 0.01
nu = mu / rho0
decay_rate = 4 * nu * k^2

gamma = 1.4
eos = IdealGasEOS(gamma)

function tgv_ic(x, y)
    vx = -U0 * cos(k * x) * sin(k * y)
    vy = U0 * sin(k * x) * cos(k * y)
    P = P0 - rho0 * U0^2 / 4.0 * (cos(2 * k * x) + cos(2 * k * y))
    return SVector(rho0, vx, vy, P)
end
````

## KE Computation
Kinetic energy = 0.5 * sum(rho * (vx^2 + vy^2) * dx * dy)

````@example tgv_kinetic_energy_decay
function compute_ke(law, coords, U, N)
    dx = L / N
    ke = 0.0
    for iy in 1:N, ix in 1:N
        w = conserved_to_primitive(law, U[ix, iy])
        ke += 0.5 * w[1] * (w[2]^2 + w[3]^2) * dx * dx
    end
    return ke
end
````

## KE Decay at Multiple Resolutions
We run the TGV at each resolution and measure KE at final time.
The analytical KE_0 = 0.5 * rho0 * U0^2 * L^2 / 2 (integral of cos^2*sin^2)

````@example tgv_kinetic_energy_decay
KE_analytical_0 = 0.25 * rho0 * U0^2 * L^2

t_final = 0.1
KE_exact_final = KE_analytical_0 * exp(-decay_rate * t_final)

resolutions = [16, 32, 64]
ke_errors = Float64[]

for N in resolutions
    ns = NavierStokesEquations{2}(eos, mu = mu, Pr = 0.72)
    mesh = StructuredMesh2D(0.0, L, 0.0, L, N, N)
    prob = HyperbolicProblem2D(
        ns, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        tgv_ic; final_time = t_final, cfl = 0.3,
    )
    coords, U, t_end = solve_hyperbolic(prob)
    ke_num = compute_ke(ns, coords, U, N)
    ke_exact = KE_analytical_0 * exp(-decay_rate * t_end)
    push!(ke_errors, abs(ke_num - ke_exact) / ke_exact)
end
````

## Convergence Rates

````@example tgv_kinetic_energy_decay
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end
rates = convergence_rates(ke_errors)
````

## KE Time History at Finest Resolution
Run the finest mesh and collect KE at multiple time snapshots.

````@example tgv_kinetic_energy_decay
N_fine = 64
ns_fine = NavierStokesEquations{2}(eos, mu = mu, Pr = 0.72)
n_snapshots = 5
dt_snap = t_final / n_snapshots
ke_history = Float64[]
t_history = Float64[]

# Run a single simulation to final time and collect KE at snapshots.
# We run in intervals, restarting from the previous state each time.
let current_U = nothing, coords_f = nothing
    for snap in 1:n_snapshots
        t_start = (snap - 1) * dt_snap
        t_end_snap = snap * dt_snap

        if snap == 1
            mesh_f = StructuredMesh2D(0.0, L, 0.0, L, N_fine, N_fine)
            prob_f = HyperbolicProblem2D(
                ns_fine, mesh_f, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                tgv_ic; initial_time = t_start, final_time = t_end_snap, cfl = 0.3,
            )
            coords_f, U_f, t_f = solve_hyperbolic(prob_f)
        else
            prev_U = current_U
            mesh_f = StructuredMesh2D(0.0, L, 0.0, L, N_fine, N_fine)
            restart_ic = let U_p = prev_U, Nf = N_fine
                function (x, y)
                    ix = clamp(Int(floor(x / (L / Nf))) + 1, 1, Nf)
                    iy = clamp(Int(floor(y / (L / Nf))) + 1, 1, Nf)
                    return conserved_to_primitive(ns_fine, U_p[ix, iy])
                end
            end
            prob_f = HyperbolicProblem2D(
                ns_fine, mesh_f, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                restart_ic; initial_time = t_start, final_time = t_end_snap, cfl = 0.3,
            )
            coords_f, U_f, t_f = solve_hyperbolic(prob_f)
        end
        current_U = U_f
        push!(t_history, t_f)
        push!(ke_history, compute_ke(ns_fine, coords_f, U_f, N_fine))
    end
end

ke_exact_history = [KE_analytical_0 * exp(-decay_rate * t) for t in t_history]
````

## Visualisation

````@example tgv_kinetic_energy_decay
fig = Figure(fontsize = 22, size = (1100, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "t", ylabel = "KE(t)",
    title = "Kinetic Energy Decay (N=64)",
)
scatter!(ax1, t_history, ke_history, color = :blue, markersize = 10, label = "Numerical")
t_ref = range(0, t_final, length = 100)
lines!(ax1, collect(t_ref), KE_analytical_0 .* exp.(-decay_rate .* t_ref), color = :black, linewidth = 2, label = "Exact")
axislegend(ax1, position = :rt)

ax2 = Axis(
    fig[1, 2], xlabel = "N", ylabel = "Relative KE error",
    xscale = log2, yscale = log10,
    title = "KE Decay Convergence",
)
scatterlines!(ax2, resolutions, ke_errors, color = :blue, marker = :circle, linewidth = 2, markersize = 12)
e_ref = ke_errors[1]
N_ref = resolutions[1]
lines!(
    ax2, resolutions, e_ref .* (N_ref ./ resolutions) .^ 2,
    color = :black, linestyle = :dash, linewidth = 1, label = L"O(N^{-2})",
)
axislegend(ax2, position = :lb)
resize_to_layout!(fig)
fig
````

## Test Assertions
Relative KE error at N=64 should be < 5%.
KE errors should decrease with resolution.

````@example tgv_kinetic_energy_decay
@assert ke_errors[end] < 0.05 #hide
@assert all(ke_errors[i] > ke_errors[i + 1] for i in 1:(length(ke_errors) - 1)) #hide
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/tgv_kinetic_energy_decay.jl).

```julia
using FiniteVolumeMethod
using StaticArrays
using CairoMakie

rho0 = 1.0
P0 = 100.0     # high pressure → low Mach
U0 = 0.01      # small velocity amplitude
L = 1.0
k = 2 * pi / L
mu = 0.01
nu = mu / rho0
decay_rate = 4 * nu * k^2

gamma = 1.4
eos = IdealGasEOS(gamma)

function tgv_ic(x, y)
    vx = -U0 * cos(k * x) * sin(k * y)
    vy = U0 * sin(k * x) * cos(k * y)
    P = P0 - rho0 * U0^2 / 4.0 * (cos(2 * k * x) + cos(2 * k * y))
    return SVector(rho0, vx, vy, P)
end

function compute_ke(law, coords, U, N)
    dx = L / N
    ke = 0.0
    for iy in 1:N, ix in 1:N
        w = conserved_to_primitive(law, U[ix, iy])
        ke += 0.5 * w[1] * (w[2]^2 + w[3]^2) * dx * dx
    end
    return ke
end

KE_analytical_0 = 0.25 * rho0 * U0^2 * L^2

t_final = 0.1
KE_exact_final = KE_analytical_0 * exp(-decay_rate * t_final)

resolutions = [16, 32, 64]
ke_errors = Float64[]

for N in resolutions
    ns = NavierStokesEquations{2}(eos, mu = mu, Pr = 0.72)
    mesh = StructuredMesh2D(0.0, L, 0.0, L, N, N)
    prob = HyperbolicProblem2D(
        ns, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        tgv_ic; final_time = t_final, cfl = 0.3,
    )
    coords, U, t_end = solve_hyperbolic(prob)
    ke_num = compute_ke(ns, coords, U, N)
    ke_exact = KE_analytical_0 * exp(-decay_rate * t_end)
    push!(ke_errors, abs(ke_num - ke_exact) / ke_exact)
end

function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end
rates = convergence_rates(ke_errors)

N_fine = 64
ns_fine = NavierStokesEquations{2}(eos, mu = mu, Pr = 0.72)
n_snapshots = 5
dt_snap = t_final / n_snapshots
ke_history = Float64[]
t_history = Float64[]

# Run a single simulation to final time and collect KE at snapshots.
# We run in intervals, restarting from the previous state each time.
let current_U = nothing, coords_f = nothing
    for snap in 1:n_snapshots
        t_start = (snap - 1) * dt_snap
        t_end_snap = snap * dt_snap

        if snap == 1
            mesh_f = StructuredMesh2D(0.0, L, 0.0, L, N_fine, N_fine)
            prob_f = HyperbolicProblem2D(
                ns_fine, mesh_f, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                tgv_ic; initial_time = t_start, final_time = t_end_snap, cfl = 0.3,
            )
            coords_f, U_f, t_f = solve_hyperbolic(prob_f)
        else
            prev_U = current_U
            mesh_f = StructuredMesh2D(0.0, L, 0.0, L, N_fine, N_fine)
            restart_ic = let U_p = prev_U, Nf = N_fine
                function (x, y)
                    ix = clamp(Int(floor(x / (L / Nf))) + 1, 1, Nf)
                    iy = clamp(Int(floor(y / (L / Nf))) + 1, 1, Nf)
                    return conserved_to_primitive(ns_fine, U_p[ix, iy])
                end
            end
            prob_f = HyperbolicProblem2D(
                ns_fine, mesh_f, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                restart_ic; initial_time = t_start, final_time = t_end_snap, cfl = 0.3,
            )
            coords_f, U_f, t_f = solve_hyperbolic(prob_f)
        end
        current_U = U_f
        push!(t_history, t_f)
        push!(ke_history, compute_ke(ns_fine, coords_f, U_f, N_fine))
    end
end

ke_exact_history = [KE_analytical_0 * exp(-decay_rate * t) for t in t_history]

fig = Figure(fontsize = 22, size = (1100, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "t", ylabel = "KE(t)",
    title = "Kinetic Energy Decay (N=64)",
)
scatter!(ax1, t_history, ke_history, color = :blue, markersize = 10, label = "Numerical")
t_ref = range(0, t_final, length = 100)
lines!(ax1, collect(t_ref), KE_analytical_0 .* exp.(-decay_rate .* t_ref), color = :black, linewidth = 2, label = "Exact")
axislegend(ax1, position = :rt)

ax2 = Axis(
    fig[1, 2], xlabel = "N", ylabel = "Relative KE error",
    xscale = log2, yscale = log10,
    title = "KE Decay Convergence",
)
scatterlines!(ax2, resolutions, ke_errors, color = :blue, marker = :circle, linewidth = 2, markersize = 12)
e_ref = ke_errors[1]
N_ref = resolutions[1]
lines!(
    ax2, resolutions, e_ref .* (N_ref ./ resolutions) .^ 2,
    color = :black, linestyle = :dash, linewidth = 1, label = L"O(N^{-2})",
)
axislegend(ax2, position = :lb)
resize_to_layout!(fig)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


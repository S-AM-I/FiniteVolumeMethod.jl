using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Smooth Advection Order of Accuracy
# This example measures the convergence rate of each reconstruction scheme
# on a smooth problem — a small-amplitude density wave advected by uniform
# flow with periodic boundary conditions.
#
# ## Mathematical Setup
# We solve the 1D Euler equations with initial condition:
# ```math
# \rho(x,0) = 1 + 0.01\sin(2\pi x), \qquad v = 1, \qquad P = 1.
# ```
# The exact solution at time $t$ is:
# ```math
# \rho_{\text{exact}}(x,t) = 1 + 0.01\sin\!\bigl(2\pi(x - vt)\bigr).
# ```
#
# ## Inputs
# - **Resolutions**: $N \in \{32, 64, 128, 256\}$
# - **Schemes**: `NoReconstruction`, `MUSCL(Minmod)`, `MUSCL(VanLeer)`, `WENO3`
# - **Riemann solver**: `HLLCSolver`
# - **Final time**: $t_f = 0.05$

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

rho0 = 1.0
v0 = 1.0
P0 = 1.0
amp = 0.01
t_final = 0.05

advection_ic(x) = SVector(rho0 + amp * sin(2 * pi * x), v0, P0)

# ## Convergence Measurement
# The $L^1$ density error at each resolution and reconstruction scheme:
function compute_error(N, recon)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), recon,
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), advection_ic;
        final_time = t_final, cfl = 0.4
    )
    x, U, t_end = solve_hyperbolic(prob)
    err = 0.0
    for i in eachindex(x)
        x_shifted = mod(x[i] - v0 * t_end, 1.0)
        rho_exact = rho0 + amp * sin(2 * pi * x_shifted)
        rho_num = conserved_to_primitive(law, U[i])[1]
        err += abs(rho_num - rho_exact)
    end
    return err / N
end

resolutions = [32, 64, 128, 256]

schemes = [
    ("NoReconstruction", NoReconstruction()),
    ("MUSCL (Minmod)", CellCenteredMUSCL(MinmodLimiter())),
    ("MUSCL (VanLeer)", CellCenteredMUSCL(VanLeerLimiter())),
    ("WENO3", WENO3()),
]
colors = [:gray, :blue, :cyan, :red]
markers = [:circle, :utriangle, :diamond, :star5]

all_errors = Dict{String, Vector{Float64}}()
for (name, recon) in schemes
    errs = [compute_error(N, recon) for N in resolutions]
    all_errors[name] = errs
end

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

# ## Visualisation — Solutions at Low and High Resolution
mesh_lo = StructuredMesh1D(0.0, 1.0, 32)
mesh_hi = StructuredMesh1D(0.0, 1.0, 256)
prob_lo = HyperbolicProblem(
    law, mesh_lo, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), advection_ic;
    final_time = t_final, cfl = 0.4
)
prob_hi = HyperbolicProblem(
    law, mesh_hi, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), advection_ic;
    final_time = t_final, cfl = 0.4
)
x_lo, U_lo, t_lo = solve_hyperbolic(prob_lo)
x_hi, U_hi, t_hi = solve_hyperbolic(prob_hi)
rho_lo = [conserved_to_primitive(law, U_lo[i])[1] for i in eachindex(U_lo)]
rho_hi = [conserved_to_primitive(law, U_hi[i])[1] for i in eachindex(U_hi)]

x_exact = range(0.0, 1.0, length = 500)
rho_exact = [rho0 + amp * sin(2 * pi * mod(xi - v0 * t_lo, 1.0)) for xi in x_exact]

fig1 = Figure(fontsize = 24, size = (1100, 450))
ax1 = Axis(fig1[1, 1], xlabel = "x", ylabel = L"\rho", title = "N = 32")
scatter!(ax1, x_lo, rho_lo, color = :blue, markersize = 6, label = "Numerical")
lines!(ax1, x_exact, rho_exact, color = :black, linewidth = 2, label = "Exact")
axislegend(ax1, position = :rt)
ax2 = Axis(fig1[1, 2], xlabel = "x", ylabel = L"\rho", title = "N = 256")
scatter!(ax2, x_hi, rho_hi, color = :blue, markersize = 3, label = "Numerical")
lines!(ax2, x_exact, rho_exact, color = :black, linewidth = 2, label = "Exact")
axislegend(ax2, position = :rt)
resize_to_layout!(fig1)
fig1
@test_reference joinpath(@__DIR__, "../figures", "smooth_advection_solutions.png") fig1 #src

# ## Visualisation — Convergence Plot
fig2 = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig2[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error}",
    xscale = log2, yscale = log10,
    title = "Convergence: smooth density wave"
)
for (idx, (name, _)) in enumerate(schemes)
    errs = all_errors[name]
    scatterlines!(
        ax, resolutions, errs, label = name, marker = markers[idx],
        color = colors[idx], linewidth = 2, markersize = 12
    )
end

## Reference slopes
e_ref = all_errors["NoReconstruction"][1]
N_ref = resolutions[1]
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 1,
    color = :black, linestyle = :dash, linewidth = 1, label = L"O(N^{-1})"
)
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 2,
    color = :black, linestyle = :dashdot, linewidth = 1, label = L"O(N^{-2})"
)
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 3,
    color = :black, linestyle = :dot, linewidth = 1, label = L"O(N^{-3})"
)
axislegend(ax, position = :lb)
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "smooth_advection_convergence.png") fig2 #src

# ## Test Assertions
rates_norecon = convergence_rates(all_errors["NoReconstruction"])
rates_muscl_mm = convergence_rates(all_errors["MUSCL (Minmod)"])
rates_muscl_vl = convergence_rates(all_errors["MUSCL (VanLeer)"])
rates_weno3 = convergence_rates(all_errors["WENO3"])

@test all(r -> r > 0.8, rates_norecon) #src
@test all(r -> r > 1.7, rates_muscl_mm) #src
@test all(r -> r > 1.7, rates_muscl_vl) #src
@test all(r -> r > 2.5, rates_weno3) #src
@assert all(r -> r > 0.8, rates_norecon) #hide
@assert all(r -> r > 1.7, rates_muscl_mm) #hide
@assert all(r -> r > 1.7, rates_muscl_vl) #hide
@assert all(r -> r > 2.5, rates_weno3) #hide

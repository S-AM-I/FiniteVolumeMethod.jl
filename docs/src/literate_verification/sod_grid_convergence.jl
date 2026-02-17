using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Sod Shock Tube Grid Convergence
# This example verifies the convergence of Riemann problem solutions with
# discontinuities. We perform a grid refinement study with the HLLC solver
# and compare multiple solver/reconstruction combinations.
#
# ## Mathematical Setup
# Standard Sod problem with initial conditions:
# ```math
# (\rho, v, P) = \begin{cases}(1, 0, 1) & x < 0.5,\\(0.125, 0, 0.1) & x \geq 0.5.\end{cases}
# ```
# with $\gamma = 1.4$.
#
# ## Inputs
# - **Grid sizes**: $N \in \{50, 100, 200, 400\}$
# - **Riemann solvers**: Lax-Friedrichs, HLL, HLLC
# - **Reconstruction**: `NoReconstruction`, `MUSCL(Minmod)`, `WENO3`

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

wL = SVector(1.0, 0.0, 1.0)
wR = SVector(0.125, 0.0, 0.1)
sod_ic(x) = x < 0.5 ? wL : wR

# ## Exact Solution
function sod_exact(x, t; x0 = 0.5, gamma = 1.4)
    rhoL, vL, PL = 1.0, 0.0, 1.0
    rhoR, vR, PR = 0.125, 0.0, 0.1
    cL = sqrt(gamma * PL / rhoL)
    ## Pre-computed star-region values
    P_star = 0.30313017805064707
    v_star = 0.92745262004895057
    rho_star_L = 0.42631942817849544
    rho_star_R = 0.26557371170530708
    c_star_L = sqrt(gamma * P_star / rho_star_L)
    ## Shock speed
    S_shock = v_star + 1.0 / (gamma * rho_star_R) *
        (P_star - PR) / (v_star - vR + 1.0e-30)
    xi = (x - x0) / t
    if xi < -cL
        return rhoL, vL, PL
    elseif xi < v_star - c_star_L
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

# ## Part 1: Grid Convergence with HLLC + MUSCL
t_final = 0.2
grid_sizes = [50, 100, 200, 400]

function compute_l1_errors(N)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR), sod_ic;
        final_time = t_final, cfl = 0.5
    )
    x, U, _ = solve_hyperbolic(prob)
    err_rho = 0.0
    err_v = 0.0
    err_P = 0.0
    for i in eachindex(x)
        w = conserved_to_primitive(law, U[i])
        rho_ex, v_ex, P_ex = sod_exact(x[i], t_final)
        err_rho += abs(w[1] - rho_ex)
        err_v += abs(w[2] - v_ex)
        err_P += abs(w[3] - P_ex)
    end
    return err_rho / N, err_v / N, err_P / N
end

errs_rho = Float64[]
errs_v = Float64[]
errs_P = Float64[]
for N in grid_sizes
    e_rho, e_v, e_P = compute_l1_errors(N)
    push!(errs_rho, e_rho)
    push!(errs_v, e_v)
    push!(errs_P, e_P)
end

# ## Visualisation — Density at Multiple Resolutions
x_exact_plot = range(0.0, 1.0, length = 1000)
rho_exact_plot = [sod_exact(xi, t_final)[1] for xi in x_exact_plot]

fig1 = Figure(fontsize = 24, size = (1500, 420))
for (panel, N_idx) in enumerate([1, 3, 4])
    N = grid_sizes[N_idx]
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR), sod_ic;
        final_time = t_final, cfl = 0.5
    )
    x, U, _ = solve_hyperbolic(prob)
    rho_num = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
    local ax
    ax = Axis(fig1[1, panel], xlabel = "x", ylabel = L"\rho", title = "N = $N")
    lines!(ax, x_exact_plot, rho_exact_plot, color = :black, linewidth = 2, label = "Exact")
    scatter!(ax, x, rho_num, color = :blue, markersize = 4, label = "HLLC+MUSCL")
    panel == 1 && axislegend(ax, position = :cb)
end
resize_to_layout!(fig1)
fig1
@test_reference joinpath(@__DIR__, "../figures", "sod_grid_convergence_density.png") fig1 #src

# ## Visualisation — L1 Error Convergence
fig2 = Figure(fontsize = 24, size = (600, 500))
ax = Axis(
    fig2[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error}",
    xscale = log2, yscale = log10,
    title = "Sod Shock Tube: Grid Convergence"
)
scatterlines!(ax, grid_sizes, errs_rho, label = "Density", color = :blue, marker = :circle, linewidth = 2, markersize = 12)
scatterlines!(ax, grid_sizes, errs_v, label = "Velocity", color = :red, marker = :utriangle, linewidth = 2, markersize = 12)
scatterlines!(ax, grid_sizes, errs_P, label = "Pressure", color = :green, marker = :diamond, linewidth = 2, markersize = 12)
## O(N^{-1}) reference slope
lines!(
    ax, grid_sizes, errs_rho[1] .* (grid_sizes[1] ./ grid_sizes) .^ 1,
    color = :gray, linestyle = :dash, linewidth = 1.5, label = L"O(N^{-1})"
)
axislegend(ax, position = :lb)
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "sod_grid_convergence_errors.png") fig2 #src

# ## Part 2: Solver/Reconstruction Comparison at N = 200
N_compare = 200
combos = [
    ("LxF + NoRecon", LaxFriedrichsSolver(), NoReconstruction()),
    ("HLL + NoRecon", HLLSolver(), NoReconstruction()),
    ("HLL + MUSCL", HLLSolver(), CellCenteredMUSCL(MinmodLimiter())),
    ("HLLC + NoRecon", HLLCSolver(), NoReconstruction()),
    ("HLLC + MUSCL", HLLCSolver(), CellCenteredMUSCL(MinmodLimiter())),
    ("HLLC + WENO3", HLLCSolver(), WENO3()),
]

combo_errors = Float64[]
combo_names = String[]
for (name, solver, recon) in combos
    mesh = StructuredMesh1D(0.0, 1.0, N_compare)
    prob = HyperbolicProblem(
        law, mesh, solver, recon,
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR), sod_ic;
        final_time = t_final, cfl = 0.4
    )
    x, U, _ = solve_hyperbolic(prob)
    err = 0.0
    for i in eachindex(x)
        rho_num = conserved_to_primitive(law, U[i])[1]
        rho_ex = sod_exact(x[i], t_final)[1]
        err += abs(rho_num - rho_ex)
    end
    push!(combo_errors, err / N_compare)
    push!(combo_names, name)
end

# ## Visualisation — Solver Comparison
fig3 = Figure(fontsize = 20, size = (700, 450))
ax3 = Axis(
    fig3[1, 1], xlabel = "Solver + Reconstruction", ylabel = L"L^1 \text{ density error}",
    title = "Sod Shock Tube: Solver Comparison (N = $N_compare)",
    xticks = (1:length(combo_names), combo_names), xticklabelrotation = pi / 4
)
barplot!(ax3, 1:length(combo_errors), combo_errors, color = :steelblue)
resize_to_layout!(fig3)
fig3
@test_reference joinpath(@__DIR__, "../figures", "sod_grid_convergence_comparison.png") fig3 #src

# ## Test Assertions
## Monotone convergence
@test all(errs_rho[i] > errs_rho[i + 1] for i in 1:(length(errs_rho) - 1)) #src
## HLLC + MUSCL beats LxF + NoRecon
@test combo_errors[5] < combo_errors[1] #src
@assert all(errs_rho[i] > errs_rho[i + 1] for i in 1:(length(errs_rho) - 1)) #hide
@assert combo_errors[5] < combo_errors[1] #hide

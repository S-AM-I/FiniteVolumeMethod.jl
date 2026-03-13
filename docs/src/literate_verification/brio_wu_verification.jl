using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Brio-Wu MHD Shock Tube Verification
# This example performs a grid convergence study of the Brio-Wu MHD shock
# tube (Brio & Wu, 1988) and verifies key solution properties against the
# published reference.
#
# ## Mathematical Setup
# The Brio-Wu problem is the MHD analogue of the Sod shock tube with
# $\gamma = 2$ and a constant normal magnetic field $B_x = 0.75$:
# ```math
# (\rho,v_x,v_y,v_z,P,B_x,B_y,B_z) = \begin{cases}
# (1, 0, 0, 0, 1, 0.75, 1, 0) & x < 0.5 \\
# (0.125, 0, 0, 0, 0.1, 0.75, -1, 0) & x \ge 0.5
# \end{cases}
# ```
# The sign change in $B_y$ produces compound MHD waves.
#
# ## References
# - Brio, M. & Wu, C.C. (1988). An Upwind Differencing Scheme for the
#   Equations of Ideal Magnetohydrodynamics. J. Comput. Phys., 75, 400-422.
# - Balsara, D.S. (2001). Total Variation Diminishing Scheme for Adiabatic
#   and Isothermal MHD. J. Comput. Phys., 174, 614-648.

using FiniteVolumeMethod
using OrdinaryDiffEq
using SciMLBase: ReturnCode
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 2.0
eos = IdealGasEOS(gamma)
law = IdealMHDEquations{1}(eos)

Bx = 0.75
wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx, 1.0, 0.0)
wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx, -1.0, 0.0)
bw_ic(x) = x < 0.5 ? wL : wR
t_final = 0.1

# ## Grid Convergence Study
# We use a very fine grid (N=1600) as the reference solution and compute
# L1 errors of coarser grids against it.
function solve_bw(N)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), bw_ic;
        final_time = t_final, cfl = 0.5,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    @test sol.retcode == ReturnCode.Success #src
    accessor = solution_accessor(prob)
    x = solution_coordinates(accessor)
    W = get_primitive(accessor, sol, length(sol.t))
    rho = [W[i][1] for i in eachindex(W)]
    return x, W, rho, sol.t[end]
end

# Reference solution at N=1600
x_ref, _, rho_ref, _ = solve_bw(1600)

# Interpolate reference to coarse grid via nearest-neighbour
function interp_nearest(x_coarse, x_fine, vals_fine)
    return [vals_fine[argmin(abs.(x_fine .- xc))] for xc in x_coarse]
end

grid_sizes = [100, 200, 400, 800]
errors_rho = Float64[]
for N in grid_sizes
    x_c, _, rho_c, _ = solve_bw(N)
    rho_ref_interp = interp_nearest(x_c, x_ref, rho_ref)
    err = sum(abs(rho_c[i] - rho_ref_interp[i]) for i in eachindex(rho_c)) / N
    push!(errors_rho, err)
end

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end
rates = convergence_rates(errors_rho)

# ## Visualisation — Solutions at Multiple Resolutions
fig1 = Figure(fontsize = 20, size = (1400, 450))
for (panel, N_idx) in enumerate([1, 2, 4])
    local Nv = grid_sizes[N_idx]
    x_c, _, rho_c, _ = solve_bw(Nv)
    ax = Axis(
        fig1[1, panel], xlabel = "x", ylabel = L"\rho",
        title = "N = $(Nv)"
    )
    lines!(ax, x_ref, rho_ref, color = :black, linewidth = 1.5, label = "Ref (1600)")
    scatter!(ax, x_c, rho_c, color = :blue, markersize = 3, label = "HLLD+MUSCL")
    panel == 1 && axislegend(ax, position = :cb)
end
resize_to_layout!(fig1)
fig1
@test_reference joinpath(@__DIR__, "../figures", "brio_wu_verification_solutions.png") fig1 #src

# ## Visualisation — Convergence Plot
fig2 = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig2[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error } (\rho)",
    xscale = log2, yscale = log10,
    title = "Brio-Wu Grid Convergence",
)
scatterlines!(
    ax, grid_sizes, errors_rho, color = :blue, marker = :circle,
    linewidth = 2, markersize = 12, label = "HLLD+MUSCL"
)
e_ref = errors_rho[1]
N_ref = grid_sizes[1]
lines!(
    ax, grid_sizes, e_ref .* (N_ref ./ grid_sizes) .^ 1,
    color = :black, linestyle = :dash, linewidth = 1, label = L"O(N^{-1})",
)
axislegend(ax, position = :lb)
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "brio_wu_verification_convergence.png") fig2 #src

# ## Structural Verification
# Check that the high-resolution solution has expected physical properties.
_, W_hr, rho_hr, _ = solve_bw(800)
Bx_vals = [W_hr[i][6] for i in eachindex(W_hr)]
By_vals = [W_hr[i][7] for i in eachindex(W_hr)]

# ## Test Assertions
# 1. Monotone error convergence
@test all(errors_rho[i] > errors_rho[i + 1] for i in 1:(length(errors_rho) - 1)) #src
# 2. Bx remains constant at 0.75 (normal B is conserved in 1D MHD)
@test all(b -> abs(b - Bx) < 1.0e-10, Bx_vals) #src
# 3. By changes sign across the contact (key Brio-Wu feature)
@test minimum(By_vals) < -0.5 #src
@test maximum(By_vals) > 0.5 #src
# 4. Density stays positive
@test all(rho_hr .> 0) #src
@assert all(errors_rho[i] > errors_rho[i + 1] for i in 1:(length(errors_rho) - 1)) #hide
@assert all(b -> abs(b - Bx) < 1.0e-10, Bx_vals) #hide
@assert minimum(By_vals) < -0.5 #hide
@assert maximum(By_vals) > 0.5 #hide
@assert all(rho_hr .> 0) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "density_errors" => errors_rho,
            "density_rates" => rates,
            "min_density_rate" => minimum(rates),
            "finest_density_error" => errors_rho[end],
            "by_min" => minimum(By_vals),
            "by_max" => maximum(By_vals),
        ),
        artifacts = [
            "brio_wu_verification_solutions.png",
            "brio_wu_verification_convergence.png",
        ],
        notes = [
            "Canonical SciML execution path via sciml_problem(prob) and solve(prob, SSPRK33()).",
            "This evidence entry is the mhd_ct benchmark-stage Brio-Wu reference comparison case.",
        ],
        summary = Dict(
            "grid_sizes" => grid_sizes,
            "density_errors" => errors_rho,
            "density_rates" => rates,
            "bx_constant" => all(b -> abs(b - Bx) < 1.0e-10, Bx_vals),
            "density_positive" => all(rho_hr .> 0),
        ),
    )
end

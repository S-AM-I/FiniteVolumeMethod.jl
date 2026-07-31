# # Plane Poiseuille Grid Convergence (Collocated SIMPLE)
# This case verifies the collocated incompressible solver — SIMPLE
# pressure-velocity coupling on an unstructured Cartesian mesh — against the
# exact plane Poiseuille solution, and confirms second-order spatial
# convergence over three successive uniform refinements.
#
# ## Mathematical Setup
# Fully-developed pressure-driven flow between parallel plates at distance
# $H$ obeys
# ```math
# 0 = -\frac{\mathrm{d}p}{\mathrm{d}x} + \mu \frac{\mathrm{d}^2 u}{\mathrm{d}y^2},
# \qquad u(0) = u(H) = 0,
# ```
# with the exact parabolic profile
# ```math
# u_{\text{exact}}(y) = \frac{G}{2\mu}\, y\,(H - y), \qquad G = -\frac{\mathrm{d}p}{\mathrm{d}x}.
# ```
# The channel is driven by imposing the exact profile at the inlet and a
# fixed reference pressure at the outlet; both walls are no-slip.
#
# ## Inputs
# - **Meshes**: $N_x \times N_y \in \{25 \times 10,\; 50 \times 20,\; 100 \times 40\}$
# - **Viscosity**: $\mu = 1$, **pressure gradient**: $G = 2$, $H = 1$, $L = 5$
# - **Solver**: `SIMPLE(0.5, 0.2, 500, 1e-6)` via `SteadyIncompressibleProblem`
#
# The error metric is the volume-weighted $L^2$ velocity error on the
# mid-channel column over $0.1 < y < 0.9$, excluding the near-wall cells
# where the finite-volume boundary treatment contributes its natural
# $O(h)$ boundary-layer error.

using FiniteVolumeMethod
using FiniteVolumeMethod: SpatialVelocityBC
using LinearSolve
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

H = 1.0
L = 5.0
mu = 1.0
G = 2.0

u_exact(y) = G / (2 * mu) * y * (H - y)

# ## Refinement Study
# Each solve imposes the exact inlet profile and measures the mid-channel
# $L^2$ error against $u_{\text{exact}}$.
function poiseuille_solution(Nx, Ny)
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, L, H)
    u_inlet = x -> SVector(G / (2 * mu) * x[2] * (H - x[2]), 0.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => SpatialVelocityBC(u_inlet, Val(2), Float64),
        :right => FixedPressureBC(0.0),
        :bottom => NoSlipWallBC(),
        :top => NoSlipWallBC(),
    )
    algo = SIMPLE(0.5, 0.2, 500, 1.0e-6)
    prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = mu, density = 1.0)
    ## Pin the linear solver so the evidence is deterministic across
    ## platforms (the LinearSolve default is platform-dependent).
    sol = solve(prob, algo; linear_solver = LUFactorization())
    return mesh, sol
end

function midchannel_l2_error(mesh, sol, Nx, Ny)
    err_sq = 0.0
    vol = 0.0
    i_mid = round(Int, Nx / 2)
    for j in 1:Ny
        c = (j - 1) * Nx + i_mid
        y = mesh.cell_centers[2, c]
        if 0.1 < y < 0.9
            err_sq += mesh.cell_volumes[c] *
                (sol.result.state.U.internal[c][1] - u_exact(y))^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

function midchannel_profile(mesh, sol, Nx, Ny)
    i_mid = round(Int, Nx / 2)
    ys = [mesh.cell_centers[2, (j - 1) * Nx + i_mid] for j in 1:Ny]
    us = [sol.result.state.U.internal[(j - 1) * Nx + i_mid][1] for j in 1:Ny]
    return ys, us
end

mesh_pairs = [(25, 10), (50, 20), (100, 40)]
results = [poiseuille_solution(Nx, Ny) for (Nx, Ny) in mesh_pairs]
errors_L2 = [
    midchannel_l2_error(mesh, sol, Nx, Ny)
        for ((mesh, sol), (Nx, Ny)) in zip(results, mesh_pairs)
]

# ## Observed Convergence Orders
# The mesh spacing halves at each refinement, so the observed order at each
# transition is $p_i = \log_2(e_i / e_{i+1})$.
orders = [log2(errors_L2[i] / errors_L2[i + 1]) for i in 1:(length(errors_L2) - 1)]

# ## Visualisation — Profile Comparison
# Computed mid-channel profile on the finest mesh against the exact parabola.
ys_fine, us_fine = midchannel_profile(results[end]..., mesh_pairs[end]...)
y_dense = range(0.0, H; length = 200)

fig1 = Figure(fontsize = 24, size = (600, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "u(y)", ylabel = "y",
    title = "Poiseuille profile (100 × 40 mesh)"
)
lines!(
    ax1, u_exact.(y_dense), y_dense, color = :black, linewidth = 2,
    label = "Exact"
)
scatter!(
    ax1, us_fine, ys_fine, color = :red, markersize = 10,
    label = "SIMPLE (collocated)"
)
axislegend(ax1, position = :rc)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("poiseuille_profile.png"), fig1)
end

# ## Visualisation — Convergence Plot
h_vals = [H / Ny for (_, Ny) in mesh_pairs]
fig2 = Figure(fontsize = 24, size = (600, 500))
ax2 = Axis(
    fig2[1, 1], xlabel = "h", ylabel = "L² error",
    xscale = log10, yscale = log10,
    title = "Poiseuille convergence (collocated SIMPLE)"
)
scatterlines!(
    ax2, h_vals, errors_L2, label = "L² (mid-channel)", marker = :circle,
    color = :blue, linewidth = 2, markersize = 12
)
lines!(
    ax2, h_vals, errors_L2[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5, label = "O(h²)"
)
axislegend(ax2, position = :rb)
for i in eachindex(orders)
    x_mid = sqrt(h_vals[i] * h_vals[i + 1])
    text!(
        ax2, x_mid, errors_L2[i] * 0.6;
        text = "$(round(orders[i], digits = 2))", fontsize = 14, color = :blue
    )
end
resize_to_layout!(fig2)
fig2
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("poiseuille_convergence.png"), fig2)
end

# ## Acceptance
# Every refinement must reduce the error, each observed order must lie in
# $(1.7, 2.3)$ around the second-order expectation, and the finest-grid
# error must be below $10^{-4}$.
@test all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)) #src
for p in orders #src
    @test 1.7 < p < 2.3 #src
end #src
@test errors_L2[end] < 1.0e-4 #src
@assert all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)) #hide
@assert all(p -> 1.7 < p < 2.3, orders) #hide
@assert errors_L2[end] < 1.0e-4 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "l2_errors" => errors_L2,
            "observed_orders" => orders,
            "min_order" => minimum(orders),
            "finest_error" => errors_L2[end],
        ),
        artifacts = ["poiseuille_profile.png", "poiseuille_convergence.png"],
        notes = [
            "Canonical steady collocated path via SteadyIncompressibleProblem and solve(prob, SIMPLE(...)).",
            "Verification-stage exact-solution convergence evidence for incompressible_ns.",
        ],
        summary = Dict(
            "mesh_pairs" => [collect(pair) for pair in mesh_pairs],
            "h_values" => h_vals,
            "observed_orders" => orders,
        ),
    )
end

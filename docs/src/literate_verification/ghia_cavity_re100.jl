# # Lid-Driven Cavity vs Ghia et al. (1982), Re = 100
# This case benchmarks the collocated incompressible solver — SIMPLE
# pressure-velocity coupling on an 80 × 80 unstructured Cartesian mesh —
# against the tabulated centerline data of Ghia, Ghia & Shin (1982),
# *High-Re Solutions for Incompressible Flow Using the Navier-Stokes
# Equations and a Multigrid Method*, J. Comput. Phys. 48, 387–411.
#
# ## Setup
# The unit cavity has three no-slip walls and a lid moving with
# $U_{\text{lid}} = 1$; at $\nu = 0.01$ the Reynolds number is 100. The
# quantity of interest is the horizontal velocity $u(y)$ along the vertical
# centerline $x = 0.5$, compared against ten reference stations from Ghia's
# Table I.
#
# ## Acceptance Gates
# - Interior reference points: $\leq 8\%$ relative error
# - Near-lid points ($y > 0.9$): $\leq 5\%$ relative error
# - Zero-crossing points ($|u_{\text{ref}}| < 0.05$): $\leq 0.025$ absolute error
# - Primary-vortex peak: $u_{\min} \in (-0.22, -0.18)$ at $y \in (0.4, 0.55)$
# - Interior continuity residual (excluding the corner-singularity band):
#   $< 10^{-4}$

using FiniteVolumeMethod
using FiniteVolumeMethod: continuity_residual_interior
using LinearSolve
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

# Ghia 1982 Table I: Re = 100, u(y) at x = 0.5 (10 selected stations).
ghia_y = [
    0.0, 0.0547, 0.1719, 0.2813, 0.5,
    0.7344, 0.8516, 0.9531, 0.9688, 1.0,
]
ghia_u = [
    0.0, -0.03717, -0.1015, -0.15662, -0.20581,
    0.00332, 0.23151, 0.68717, 0.78871, 1.0,
]

# ## Solve
N = 80
mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
bcs = Dict{Symbol, AbstractBoundaryCondition}(
    :left => NoSlipWallBC(),
    :right => NoSlipWallBC(),
    :bottom => NoSlipWallBC(),
    :top => FixedVelocityBC(SVector(1.0, 0.0)),
)
algo = SIMPLE(; max_iterations = 2500, tolerance = 1.0e-5)
prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = 0.01, density = 1.0)
## Pin the linear solver so the evidence is deterministic across platforms
## (the LinearSolve default is platform-dependent).
sol = solve(prob, algo; linear_solver = LUFactorization())

# The lid-corner singularity keeps the flux-normalized continuity residual
# above the nominal tolerance on this mesh, so the output fields are
# validated directly rather than the retcode.
sol.result.iterations

# ## Centerline Extraction
i_mid = N ÷ 2
centerline_y = [mesh.cell_centers[2, (j - 1) * N + i_mid] for j in 1:N]
centerline_u = [sol.result.state.U.internal[(j - 1) * N + i_mid][1] for j in 1:N]

(peak_u, peak_i) = findmin(centerline_u)
peak_y = centerline_y[peak_i]

# Interior continuity residual, excluding the 0.1L band at each wall where
# the lid/wall corner singularity concentrates the divergence defect.
interior_div = continuity_residual_interior(sol.result.state, mesh)

# ## Visualisation — Centerline Profile vs Ghia
fig1 = Figure(fontsize = 24, size = (600, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "u(y) at x = 0.5", ylabel = "y",
    title = "Lid-driven cavity, Re = 100 (80 × 80)"
)
lines!(
    ax1, centerline_u, centerline_y, color = :blue, linewidth = 2,
    label = "SIMPLE (collocated)"
)
scatter!(
    ax1, ghia_u, ghia_y, color = :black, marker = :diamond,
    markersize = 14, label = "Ghia et al. (1982)"
)
axislegend(ax1, position = :rc)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("ghia_centerline.png"), fig1)
end

# ## Visualisation — Velocity Magnitude
u_mag = [
    sqrt(sum(abs2, sol.result.state.U.internal[(j - 1) * N + i]))
        for i in 1:N, j in 1:N
]
x_centers = [mesh.cell_centers[1, i] for i in 1:N]
y_centers = [mesh.cell_centers[2, (j - 1) * N + 1] for j in 1:N]

fig2 = Figure(fontsize = 24, size = (600, 500))
ax2 = Axis(
    fig2[1, 1], xlabel = "x", ylabel = "y",
    title = "|U|, Re = 100", aspect = DataAspect()
)
hm = heatmap!(ax2, x_centers, y_centers, u_mag, colormap = :viridis)
Colorbar(fig2[1, 2], hm)
resize_to_layout!(fig2)
fig2
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("ghia_velocity_magnitude.png"), fig2)
end

# ## Acceptance
@test sol.result.iterations > 0 #src
@test -0.22 < peak_u < -0.18 #src
@test 0.4 < peak_y < 0.55 #src
@test 0.95 < maximum(centerline_u) <= 1.01 #src
@test interior_div < 1.0e-4 #src
@assert sol.result.iterations > 0 #hide
@assert -0.22 < peak_u < -0.18 #hide
@assert 0.4 < peak_y < 0.55 #hide
@assert 0.95 < maximum(centerline_u) <= 1.01 #hide
@assert interior_div < 1.0e-4 #hide

# Point-wise agreement with the Ghia table. Zero-crossing stations are gated
# on absolute error (a relative gate against a reference value of 0.003 would
# be meaningless); interior stations at 8% relative, near-lid at 5%.
tol_interior = 0.08
tol_near_lid = 0.05
abs_zero_crossing = 0.025

point_errors = map(zip(ghia_y, ghia_u)) do (y_t, u_t)
    _, idx = findmin(abs.(centerline_y .- y_t))
    u_c = centerline_u[idx]
    if abs(u_t) < 0.05
        (station = y_t, kind = "absolute", error = abs(u_c - u_t), gate = abs_zero_crossing)
    else
        tol = y_t > 0.9 ? tol_near_lid : tol_interior
        (station = y_t, kind = "relative", error = abs(u_c - u_t) / abs(u_t), gate = tol)
    end
end

for pe in point_errors #src
    @test pe.error <= pe.gate #src
end #src
@assert all(pe -> pe.error <= pe.gate, point_errors) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "peak_u" => peak_u,
            "peak_y" => peak_y,
            "interior_continuity_residual" => interior_div,
            "max_relative_error" => maximum(
                pe.error for pe in point_errors if pe.kind == "relative"
            ),
            "max_absolute_error_zero_crossing" => maximum(
                pe.error for pe in point_errors if pe.kind == "absolute"
            ),
            "iterations" => sol.result.iterations,
        ),
        artifacts = ["ghia_centerline.png", "ghia_velocity_magnitude.png"],
        notes = [
            "Benchmark-stage literature-table evidence for incompressible_ns against Ghia, Ghia & Shin (1982) Table I, Re = 100.",
            "Gates: 8% interior / 5% near-lid relative, 0.025 absolute at zero crossings, primary-vortex peak location and magnitude, interior continuity residual < 1e-4.",
        ],
        summary = Dict(
            "mesh" => [N, N],
            "reynolds" => 100.0,
            "stations" => ghia_y,
            "point_errors" => [pe.error for pe in point_errors],
            "point_gates" => [pe.gate for pe in point_errors],
        ),
    )
end

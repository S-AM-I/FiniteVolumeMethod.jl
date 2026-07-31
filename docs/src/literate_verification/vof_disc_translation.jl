# # VOF Disc Translation
# This case verifies the alpha-transport solver (`assemble_alpha!` + linear
# solve + `clip_alpha!`) against three exact properties of kinematic
# advection under a divergence-free uniform velocity field:
# 1. **Mass conservation** — with $\nabla \cdot u = 0$ and $\alpha = 0$ at
#    the inflow, $\sum_c \alpha_c V_c$ is conserved exactly (upwind
#    convection + backward Euler are conservative by construction; the
#    only drift is LU round-off)
# 2. **Centre-of-mass translation** — a disc of $\alpha = 1$ translates
#    with the fluid: $x_{\text{COM}}(t) = x_{\text{COM}}(0) + U t$
# 3. **Boundedness** — $\alpha \in [0, 1]$ at every cell and step
#
# Interface compression is disabled ($C_\alpha = 0$): this is a
# pure-kinematic verification of the transport operator.
#
# ## Acceptance Gates
# - Relative mass drift $< 10^{-6}$ over the full run and $< 10^{-12}$
#   over the first half (round-off regime)
# - $\Delta x_{\text{COM}}$ within $2h$ of $U\,t_{\text{end}}$; lateral
#   drift $< 10^{-10}$ (symmetry)
# - $\alpha \in [0, 1]$ throughout

using FiniteVolumeMethod
using FiniteVolumeMethod: CollocatedEquation, assemble_alpha!, clip_alpha!,
    face_normal_area, to_linear_problem
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using LinearSolve
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

Lx = 2.0
Ly = 0.5
Ux = 1.0
disc_center = (0.3, 0.25)
disc_radius = 0.1
t_end = 0.5

# ## Setup and Time Loop
# The face flux of a uniform velocity is divergence-free by the closed-cell
# identity; a direct LU solve keeps mass conservation at round-off (an
# iterative tolerance of $10^{-8}$ would leak mass at that level each step).
function run_translation(Nx, Ny, n_steps)
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, Lx, Ly)
    n_cells = length(mesh.cell_volumes)
    n_faces = size(mesh.face_cells, 2)

    alpha = CollocatedScalarField(:alpha, mesh; value = 0.0)
    for c in 1:n_cells
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if (x - disc_center[1])^2 + (y - disc_center[2])^2 < disc_radius^2
            alpha.internal[c] = 1.0
        end
    end

    phi = FaceFluxField(:phi, mesh)
    for f in 1:n_faces
        phi.values[f] = Ux * face_normal_area(mesh, f)[1]
    end

    bcs_alpha = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => NeumannBC(0.0),
        :bottom => NeumannBC(0.0),
        :top => NeumannBC(0.0),
    )

    total_mass() = sum(alpha.internal[c] * mesh.cell_volumes[c] for c in 1:n_cells)
    function center_of_mass()
        mass = 0.0
        mx = 0.0
        my = 0.0
        for c in 1:n_cells
            w = alpha.internal[c] * mesh.cell_volumes[c]
            mass += w
            mx += w * mesh.cell_centers[1, c]
            my += w * mesh.cell_centers[2, c]
        end
        return (x = mx / mass, y = my / mass)
    end

    alpha0 = copy(alpha.internal)
    mass_hist = Float64[total_mass()]
    bound_min = minimum(alpha.internal)
    bound_max = maximum(alpha.internal)
    com0 = center_of_mass()

    dt = t_end / n_steps
    for _ in 1:n_steps
        eq = CollocatedEquation(mesh)
        assemble_alpha!(eq, alpha, phi, mesh, bcs_alpha; dt = dt, C_alpha = 0.0)
        sol = LinearSolve.solve(to_linear_problem(eq), LUFactorization())
        for c in 1:n_cells
            alpha.internal[c] = sol.u[c]
        end
        clip_alpha!(alpha, mesh)
        push!(mass_hist, total_mass())
        bound_min = min(bound_min, minimum(alpha.internal))
        bound_max = max(bound_max, maximum(alpha.internal))
    end

    return (
        mesh = mesh, alpha = alpha, alpha0 = alpha0,
        mass_hist = mass_hist, com0 = com0, com_final = center_of_mass(),
        bound_min = bound_min, bound_max = bound_max,
    )
end

Nx, Ny, n_steps = 80, 20, 25
res = run_translation(Nx, Ny, n_steps)

# ## Metrics
mass0 = res.mass_hist[1]
mass_drift = abs(res.mass_hist[end] - mass0) / mass0
mass_drift_half = abs(res.mass_hist[(n_steps ÷ 2) + 1] - mass0) / mass0
mass_range = (maximum(res.mass_hist) - minimum(res.mass_hist)) / mass0
dx_com = res.com_final.x - res.com0.x
dy_com = abs(res.com_final.y - res.com0.y)
h = Lx / Nx

# ## Visualisation — Initial and Final Fields
alpha0_mat = [res.alpha0[(j - 1) * Nx + i] for i in 1:Nx, j in 1:Ny]
alpha_mat = [res.alpha.internal[(j - 1) * Nx + i] for i in 1:Nx, j in 1:Ny]
x_centers = [res.mesh.cell_centers[1, i] for i in 1:Nx]
y_centers = [res.mesh.cell_centers[2, (j - 1) * Nx + 1] for j in 1:Ny]

fig1 = Figure(fontsize = 24, size = (900, 500))
ax1 = Axis(fig1[1, 1], xlabel = "x", ylabel = "y", title = "α at t = 0", aspect = DataAspect())
heatmap!(ax1, x_centers, y_centers, alpha0_mat, colormap = :blues, colorrange = (0, 1))
ax2 = Axis(fig1[2, 1], xlabel = "x", ylabel = "y", title = "α at t = 0.5", aspect = DataAspect())
hm = heatmap!(ax2, x_centers, y_centers, alpha_mat, colormap = :blues, colorrange = (0, 1))
Colorbar(fig1[1:2, 2], hm)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("vof_translation_fields.png"), fig1)
end

# ## Visualisation — Mass History
fig2 = Figure(fontsize = 24, size = (600, 450))
ax3 = Axis(
    fig2[1, 1], xlabel = "step", ylabel = "relative mass drift",
    title = "Mass conservation"
)
lines!(
    ax3, 0:n_steps, abs.(res.mass_hist .- mass0) ./ mass0 .+ 1.0e-18,
    color = :blue, linewidth = 2
)
resize_to_layout!(fig2)
fig2
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("vof_translation_mass.png"), fig2)
end

# ## Acceptance
@test mass_drift < 1.0e-6 #src
@test mass_drift_half < 1.0e-12 #src
@test mass_range < 1.0e-6 #src
@test res.bound_min >= 0.0 #src
@test res.bound_max <= 1.0 + 1.0e-14 #src
@test abs(dx_com - Ux * t_end) < 2 * h #src
@test dy_com < 1.0e-10 #src
@assert mass_drift < 1.0e-6 #hide
@assert mass_drift_half < 1.0e-12 #hide
@assert mass_range < 1.0e-6 #hide
@assert res.bound_min >= 0.0 #hide
@assert res.bound_max <= 1.0 + 1.0e-14 #hide
@assert abs(dx_com - Ux * t_end) < 2 * h #hide
@assert dy_com < 1.0e-10 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "mass_drift" => mass_drift,
            "mass_drift_first_half" => mass_drift_half,
            "com_dx_error" => abs(dx_com - Ux * t_end),
            "com_dy_drift" => dy_com,
            "alpha_bounds" => [res.bound_min, res.bound_max],
        ),
        artifacts = ["vof_translation_fields.png", "vof_translation_mass.png"],
        notes = [
            "Verification-stage evidence for multiphase_vof: pure-kinematic alpha transport under divergence-free uniform flow — exact mass conservation, first-moment translation, and boundedness.",
            "Interface compression disabled (C_alpha = 0); the transport operator alone is under test.",
        ],
        summary = Dict(
            "mesh" => [Nx, Ny],
            "steps" => n_steps,
            "t_end" => t_end,
        ),
    )
end

# # WALE Operator Invariants
# The WALE model (Nicoud & Ducros 1999) was designed so that the subgrid
# viscosity vanishes at a wall without explicit damping. Its defining
# analytic property is that the traceless symmetric part of the squared
# velocity-gradient tensor, $S^d_{ij}$, vanishes identically on 2D
# "shear-only" fields:
# - pure shear $U = (A y, 0)$ — the single off-diagonal gradient entry
#   squares to zero,
# - solid-body rotation $U = (-\Omega y, \Omega x)$ — $g^2 = -\Omega^2 I$
#   is isotropic, so its deviator vanishes.
#
# This is exactly where Smagorinsky produces its largest $\nu_t$, so the
# invariant discriminates the two closures. On a flow with non-trivial
# second gradients, $U = (xy, 0)$, WALE gives $\nu_t > 0$ obeying the
# $(C_w \Delta)^2$ scaling.
#
# ## Acceptance Gates
# - Zero field: $|\nu_t| < 10^{-14}$ everywhere
# - Pure shear: interior $|\nu_t| < 10^{-12}$ (Smagorinsky would give
#   $(C_s\Delta)^2 A > 0$ here)
# - Solid-body rotation: interior $|\nu_t| < 10^{-10}$
# - $U = (xy, 0)$: $\nu_t \geq 0$ everywhere, strictly positive somewhere;
#   interior-average ratio $1/4$ under 2× refinement (to 5%); $C_w$-doubling
#   ratio $= 4$ to relative $10^{-10}$

using FiniteVolumeMethod
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

function wale_nu_t(mesh, Cw, U)
    model = WALE(mesh; Cw = Cw)
    nu_t = zeros(Float64, length(mesh.cell_volumes))
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)
    return nu_t
end

function field_from(mesh, f)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:length(mesh.cell_volumes)
        U.internal[c] = f(mesh.cell_centers[1, c], mesh.cell_centers[2, c])
    end
    return U
end

interior_cells(mesh, lo, hi) = [
    c for c in 1:length(mesh.cell_volumes)
        if lo < mesh.cell_centers[1, c] < hi && lo < mesh.cell_centers[2, c] < hi
]

mesh16 = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
interior = interior_cells(mesh16, 0.2, 0.8)

# ## Vanishing Cases
nu_zero = wale_nu_t(mesh16, 0.325, field_from(mesh16, (x, y) -> SVector(0.0, 0.0)))
nu_shear = wale_nu_t(mesh16, 0.325, field_from(mesh16, (x, y) -> SVector(3.0 * y, 0.0)))
nu_rotation = wale_nu_t(
    mesh16, 0.325,
    field_from(mesh16, (x, y) -> SVector(-2.5 * (y - 0.5), 2.5 * (x - 0.5)))
)

zero_max = maximum(abs, nu_zero)
shear_max = maximum(abs, nu_shear[c] for c in interior)
rotation_max = maximum(abs, nu_rotation[c] for c in interior)

# ## Non-Trivial Flow with Δ² Scaling
xy_field(mesh) = field_from(mesh, (x, y) -> SVector(x * y, 0.0))

refine_means = map((8, 16)) do N
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    nu = wale_nu_t(mesh, 0.325, xy_field(mesh))
    cells = interior_cells(mesh, 0.3, 0.7)
    sum(nu[c] for c in cells) / length(cells)
end
refine_ratio = refine_means[2] / refine_means[1]

nu_xy = wale_nu_t(mesh16, 0.325, xy_field(mesh16))
xy_nonneg = all(>=(0.0), nu_xy)
xy_positive = maximum(nu_xy) > 0.0

# ## Invariant — ν_t ∝ C_w²
cw_values = [0.1, 0.2, 0.4]
cw_means = map(cw_values) do Cw
    nu = wale_nu_t(mesh16, Cw, xy_field(mesh16))
    cells = interior_cells(mesh16, 0.3, 0.7)
    sum(nu[c] for c in cells) / length(cells)
end
cw_ratios = [cw_means[2] / cw_means[1], cw_means[3] / cw_means[2]]

# ## Visualisation — Discrimination and Scaling
N = 16
nu_xy_mat = [nu_xy[(j - 1) * N + i] for i in 1:N, j in 1:N]
x_centers = [mesh16.cell_centers[1, i] for i in 1:N]
y_centers = [mesh16.cell_centers[2, (j - 1) * N + 1] for j in 1:N]

fig1 = Figure(fontsize = 24, size = (1000, 450))
ax1 = Axis(
    fig1[1, 1], xlabel = "x", ylabel = "y",
    title = "WALE ν_t under U = (xy, 0)", aspect = DataAspect()
)
hm = heatmap!(ax1, x_centers, y_centers, nu_xy_mat, colormap = :viridis)
Colorbar(fig1[1, 2], hm)
ax2 = Axis(
    fig1[1, 3], xlabel = "C_w", ylabel = "interior-average ν_t",
    xscale = log10, yscale = log10, title = "C_w² scaling"
)
scatterlines!(
    ax2, cw_values, cw_means, marker = :circle, color = :blue,
    linewidth = 2, markersize = 12
)
lines!(
    ax2, cw_values, cw_means[1] .* (cw_values ./ cw_values[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5
)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("wale_invariants.png"), fig1)
end

# ## Acceptance
@test zero_max < 1.0e-14 #src
@test shear_max < 1.0e-12 #src
@test rotation_max < 1.0e-10 #src
@test xy_nonneg #src
@test xy_positive #src
@test isapprox(refine_ratio, 0.25; rtol = 5.0e-2) #src
@test all(r -> isapprox(r, 4.0; rtol = 1.0e-10), cw_ratios) #src
@assert zero_max < 1.0e-14 #hide
@assert shear_max < 1.0e-12 #hide
@assert rotation_max < 1.0e-10 #hide
@assert xy_nonneg #hide
@assert xy_positive #hide
@assert isapprox(refine_ratio, 0.25; rtol = 5.0e-2) #hide
@assert all(r -> isapprox(r, 4.0; rtol = 1.0e-10), cw_ratios) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "zero_field_max" => zero_max,
            "pure_shear_max" => shear_max,
            "rotation_max" => rotation_max,
            "refinement_ratio" => refine_ratio,
            "cw_scaling_ratios" => cw_ratios,
        ),
        artifacts = ["wale_invariants.png"],
        notes = [
            "Benchmark-stage evidence for turbulence_les: the WALE operator invariants of Nicoud & Ducros (1999) — vanishing nu_t on pure shear and solid-body rotation (where Smagorinsky is maximal), positive nu_t with (Cw*Delta)^2 scaling on a flow with non-trivial second gradients.",
        ],
        summary = Dict(
            "Cw" => 0.325,
            "interior_cells_checked" => length(interior),
            "cw_values" => cw_values,
        ),
    )
end

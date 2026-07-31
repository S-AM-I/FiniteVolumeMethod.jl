# # Smagorinsky Eddy Viscosity on Prescribed Shear
# This case verifies the Smagorinsky subgrid-scale viscosity
# ```math
# \nu_t = (C_s \Delta)^2 |S|
# ```
# against its exact value on prescribed velocity fields, where the FVM
# gradient is exact:
# 1. Zero velocity → $|S| = 0$ → $\nu_t \equiv 0$
# 2. Linear shear $U = (A y, 0)$ → $|S| = A$ exactly →
#    $\nu_t = (C_s \Delta)^2 A$ uniform on the interior
#
# Two algebraic invariants complete the verification: $\nu_t \propto C_s^2$
# at fixed flow and mesh, and $\nu_t \propto \Delta^2$ at fixed flow and
# $C_s$ (filter width halves under 2× refinement, so the interior-average
# $\nu_t$ ratio is exactly $1/4$).
#
# ## Acceptance Gates
# - Zero field: $|\nu_t| < 10^{-14}$ everywhere
# - Linear shear: interior cells match $(C_s \Delta)^2 A$ to relative
#   $10^{-8}$; realizability $\nu_t \geq 0$ everywhere
# - $C_s$-doubling ratio $= 4$ to relative $10^{-10}$
# - Refinement ratio $= 1/4$ to relative $10^{-8}$

using FiniteVolumeMethod
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

function smagorinsky_nu_t(mesh, Cs, U)
    model = Smagorinsky(mesh; Cs = Cs)
    nu_t = zeros(Float64, length(mesh.cell_volumes))
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)
    return model, nu_t
end

shear_field(mesh, A) = begin
    U = CollocatedVectorField(:U, mesh)
    for c in 1:length(mesh.cell_volumes)
        U.internal[c] = SVector(A * mesh.cell_centers[2, c], 0.0)
    end
    U
end

interior_cells(mesh, lo, hi) = [
    c for c in 1:length(mesh.cell_volumes)
        if lo < mesh.cell_centers[1, c] < hi && lo < mesh.cell_centers[2, c] < hi
]

# ## Zero Field
mesh16 = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
U_zero = CollocatedVectorField(:U, mesh16; value = SVector(0.0, 0.0))
_, nu_zero = smagorinsky_nu_t(mesh16, 0.1, U_zero)
zero_max = maximum(abs, nu_zero)

# ## Linear Shear
A = 3.0
U_shear = shear_field(mesh16, A)
model_shear, nu_shear = smagorinsky_nu_t(mesh16, 0.1, U_shear)
interior = interior_cells(mesh16, 0.2, 0.8)
shear_errors = [
    abs(nu_shear[c] - (model_shear.Cs * model_shear.delta[c])^2 * A) /
        ((model_shear.Cs * model_shear.delta[c])^2 * A)
        for c in interior
]
realizable = all(>=(0.0), nu_shear)

# ## Invariant — ν_t ∝ C_s²
probe = first(interior_cells(mesh16, 0.4, 0.6))
cs_values = [0.05, 0.1, 0.2]
nu_probe = map(cs_values) do Cs
    _, nu = smagorinsky_nu_t(mesh16, Cs, shear_field(mesh16, 2.0))
    nu[probe]
end
cs_ratios = [nu_probe[2] / nu_probe[1], nu_probe[3] / nu_probe[2]]

# ## Invariant — ν_t ∝ Δ²
refine_means = map((8, 16)) do N
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    _, nu = smagorinsky_nu_t(mesh, 0.1, shear_field(mesh, 1.0))
    cells = interior_cells(mesh, 0.3, 0.7)
    sum(nu[c] for c in cells) / length(cells)
end
refine_ratio = refine_means[2] / refine_means[1]

# ## Visualisation — ν_t Field and Scaling
N = 16
nu_mat = [nu_shear[(j - 1) * N + i] for i in 1:N, j in 1:N]
x_centers = [mesh16.cell_centers[1, i] for i in 1:N]
y_centers = [mesh16.cell_centers[2, (j - 1) * N + 1] for j in 1:N]

fig1 = Figure(fontsize = 24, size = (1000, 450))
ax1 = Axis(
    fig1[1, 1], xlabel = "x", ylabel = "y",
    title = "ν_t under U = (3y, 0)", aspect = DataAspect()
)
hm = heatmap!(ax1, x_centers, y_centers, nu_mat, colormap = :viridis)
Colorbar(fig1[1, 2], hm)
ax2 = Axis(
    fig1[1, 3], xlabel = "C_s", ylabel = "ν_t (probe cell)",
    xscale = log10, yscale = log10, title = "C_s² scaling"
)
scatterlines!(
    ax2, cs_values, nu_probe, marker = :circle, color = :blue,
    linewidth = 2, markersize = 12
)
lines!(
    ax2, cs_values, nu_probe[1] .* (cs_values ./ cs_values[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5
)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("smagorinsky_shear.png"), fig1)
end

# ## Acceptance
@test zero_max < 1.0e-14 #src
@test all(e -> e < 1.0e-8, shear_errors) #src
@test realizable #src
@test all(r -> isapprox(r, 4.0; rtol = 1.0e-10), cs_ratios) #src
@test isapprox(refine_ratio, 0.25; rtol = 1.0e-8) #src
@assert zero_max < 1.0e-14 #hide
@assert all(e -> e < 1.0e-8, shear_errors) #hide
@assert realizable #hide
@assert all(r -> isapprox(r, 4.0; rtol = 1.0e-10), cs_ratios) #hide
@assert isapprox(refine_ratio, 0.25; rtol = 1.0e-8) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "zero_field_max" => zero_max,
            "max_shear_relative_error" => maximum(shear_errors),
            "cs_scaling_ratios" => cs_ratios,
            "refinement_ratio" => refine_ratio,
        ),
        artifacts = ["smagorinsky_shear.png"],
        notes = [
            "Verification-stage exact-solution evidence for turbulence_les: the Smagorinsky nu_t algebra on prescribed fields where the FVM gradient is exact, plus the Cs^2 and Delta^2 scaling invariants.",
        ],
        summary = Dict(
            "shear_rate" => A,
            "interior_cells_checked" => length(interior),
            "cs_values" => cs_values,
        ),
    )
end

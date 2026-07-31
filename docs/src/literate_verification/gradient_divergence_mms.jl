# # Gradient and Divergence Operator MMS
# This case verifies the two remaining Phase-0 collocated operators by the
# method of manufactured solutions (Versteeg & Malalasekera 2007; Ferziger
# & Perić 2020):
#
# **Green-Gauss gradient** — on
# $\varphi(x, y) = \sin(\pi x)\sin(\pi y)$ the interior gradient error
# converges at $O(h^2)$ (midpoint-rule face values close the volume
# integral at second order).
#
# **Divergence** — the manufactured field
# $U = (\sin\pi x \cos\pi y,\; -\cos\pi x \sin\pi y)$ is divergence-free
# analytically; evaluated at face centres on a uniform Cartesian mesh, the
# discrete divergence is a pure floating-point cancellation, so the
# operator is **exact** on this input — near machine zero independent of
# $h$, a stronger statement than $O(h^2)$.
#
# ## Acceptance Gates
# - Gradient: finest-transition order in $(1.8, 2.2)$, monotone decrease,
#   finest interior $L^2 < 0.05$
# - Divergence: interior RMS divergence $< 10^{-10}$ at every tested $N$

using FiniteVolumeMethod
using LinearAlgebra: norm
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

phi_exact(x, y) = sin(π * x) * sin(π * y)
grad_phi_exact(x, y) = SVector(
    π * cos(π * x) * sin(π * y),
    π * sin(π * x) * cos(π * y)
)
U_div_free(x, y) = SVector(sin(π * x) * cos(π * y), -cos(π * x) * sin(π * y))

# ## Green-Gauss Gradient MMS
function gradient_mms_error(N)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    phi = CollocatedScalarField(:phi, mesh)
    for c in 1:length(mesh.cell_volumes)
        phi.internal[c] = phi_exact(mesh.cell_centers[1, c], mesh.cell_centers[2, c])
    end
    for (i, f) in enumerate(phi.boundary_face_indices)
        phi.boundary[i] = phi_exact(mesh.face_centers[1, f], mesh.face_centers[2, f])
    end
    grad_num = gradient(phi, mesh)

    err_sq = 0.0
    vol = 0.0
    for c in 1:length(mesh.cell_volumes)
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.15 < x < 0.85 && 0.15 < y < 0.85
            err_sq += mesh.cell_volumes[c] * norm(grad_num[c] - grad_phi_exact(x, y))^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

mesh_sizes = [20, 40, 80]
grad_errors = [gradient_mms_error(N) for N in mesh_sizes]
grad_orders = [log2(grad_errors[i] / grad_errors[i + 1]) for i in 1:(length(mesh_sizes) - 1)]

# ## Divergence Exactness
function divergence_rms(N)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    n_cells = length(mesh.cell_volumes)
    n_faces = size(mesh.face_cells, 2)

    div_per_cell = zeros(n_cells)
    for f in 1:n_faces
        U_f = U_div_free(mesh.face_centers[1, f], mesh.face_centers[2, f])
        flux = mesh.face_areas[f] * (
            U_f[1] * mesh.face_normals[1, f] +
                U_f[2] * mesh.face_normals[2, f]
        )
        P = mesh.face_cells[1, f]
        Nb = mesh.face_cells[2, f]
        div_per_cell[P] += flux
        if Nb != 0
            div_per_cell[Nb] -= flux
        end
    end

    err_sq = 0.0
    vol = 0.0
    for c in 1:n_cells
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.15 < x < 0.85 && 0.15 < y < 0.85
            V = mesh.cell_volumes[c]
            err_sq += V * (div_per_cell[c] / V)^2
            vol += V
        end
    end
    return sqrt(err_sq / vol)
end

div_residuals = [divergence_rms(N) for N in mesh_sizes]

# ## Visualisation — Gradient Convergence and Divergence Exactness
h_vals = 1.0 ./ mesh_sizes
fig1 = Figure(fontsize = 24, size = (1000, 450))
ax1 = Axis(
    fig1[1, 1], xlabel = "h", ylabel = "L² gradient error",
    xscale = log10, yscale = log10, title = "Green-Gauss gradient MMS"
)
scatterlines!(
    ax1, h_vals, grad_errors, marker = :circle, color = :blue,
    linewidth = 2, markersize = 12
)
lines!(
    ax1, h_vals, grad_errors[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5
)
ax2 = Axis(
    fig1[1, 2], xlabel = "N", ylabel = "interior RMS divergence",
    yscale = log10, title = "Divergence exactness"
)
scatterlines!(
    ax2, mesh_sizes, div_residuals .+ 1.0e-18, marker = :utriangle,
    color = :red, linewidth = 2, markersize = 12
)
hlines!(ax2, [1.0e-10], color = :black, linestyle = :dash)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("gradient_divergence_mms.png"), fig1)
end

# ## Acceptance
@test 1.8 < grad_orders[end] < 2.2 #src
@test all(grad_errors[i] > grad_errors[i + 1] for i in 1:(length(grad_errors) - 1)) #src
@test grad_errors[end] < 0.05 #src
@test all(r -> r < 1.0e-10, div_residuals) #src
@assert 1.8 < grad_orders[end] < 2.2 #hide
@assert all(grad_errors[i] > grad_errors[i + 1] for i in 1:(length(grad_errors) - 1)) #hide
@assert grad_errors[end] < 0.05 #hide
@assert all(r -> r < 1.0e-10, div_residuals) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "gradient_l2_errors" => grad_errors,
            "gradient_orders" => grad_orders,
            "divergence_rms" => div_residuals,
        ),
        artifacts = ["gradient_divergence_mms.png"],
        notes = [
            "Benchmark-stage manufactured-solution evidence for collocated_operators: Green-Gauss gradient at interior O(h^2) and machine-exact discrete divergence of an analytically divergence-free face flux.",
        ],
        summary = Dict(
            "mesh_sizes" => mesh_sizes,
            "h_values" => h_vals,
        ),
    )
end

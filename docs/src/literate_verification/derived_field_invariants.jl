# # Derived-Field Invariants (Vorticity, Q-Criterion, Enstrophy)
# This case verifies `compute_vorticity`, `compute_q_criterion`, and
# `compute_enstrophy` against closed-form values on three canonical
# linear flows, where the FVM gradient is exact:
#
# | Flow | $\omega_z$ | $Q$ | Enstrophy $|\omega|^2$ |
# |---|---|---|---|
# | Uniform $U = (U_0, 0)$ | $0$ | $0$ | $0$ |
# | Shear $U = (Ay, 0)$ | $-A$ | $0$ (\|Ω\|² = \|S\|²) | $A^2$ |
# | Rotation $U = (-\Omega y, \Omega x)$ | $2\Omega$ | $\Omega^2$ ($S = 0$) | $4\Omega^2$ |
#
# The shear case discriminates $Q$ from vorticity (rotational but not
# vortical), and the rotation case is the pure-vortex limit.
#
# ## Acceptance Gates
# - Uniform: $|\omega|, |Q| < 10^{-10}$ on the interior
# - Shear: $\omega = -A$ and rotation: $\omega = 2\Omega$, $Q = \Omega^2$,
#   enstrophy $= 4\Omega^2$, all to relative $10^{-8}$; shear $|Q| < 10^{-8}$

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_enstrophy, compute_q_criterion, compute_vorticity
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
n_cells = length(mesh.cell_volumes)
interior = [
    c for c in 1:n_cells
        if 0.2 < mesh.cell_centers[1, c] < 0.8 && 0.2 < mesh.cell_centers[2, c] < 0.8
]

function field_from(f)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:n_cells
        U.internal[c] = f(mesh.cell_centers[1, c], mesh.cell_centers[2, c])
    end
    return U
end

# ## Uniform Flow
U_uniform = field_from((x, y) -> SVector(2.0, 0.0))
omega_u = compute_vorticity(U_uniform, mesh)
Q_u = compute_q_criterion(U_uniform, mesh)
uniform_max = max(
    maximum(abs, omega_u[c] for c in interior),
    maximum(abs, Q_u[c] for c in interior),
)

# ## Simple Shear
A = 4.0
U_shear = field_from((x, y) -> SVector(A * y, 0.0))
omega_s = compute_vorticity(U_shear, mesh)
Q_s = compute_q_criterion(U_shear, mesh)
shear_omega_err = maximum(abs(omega_s[c] + A) / A for c in interior)
shear_Q_max = maximum(abs, Q_s[c] for c in interior)

# ## Solid-Body Rotation
Omega = 3.0
U_rot = field_from((x, y) -> SVector(-Omega * (y - 0.5), Omega * (x - 0.5)))
omega_r = compute_vorticity(U_rot, mesh)
Q_r = compute_q_criterion(U_rot, mesh)
enst_r = compute_enstrophy(U_rot, mesh)
rot_omega_err = maximum(abs(omega_r[c] - 2 * Omega) / (2 * Omega) for c in interior)
rot_Q_err = maximum(abs(Q_r[c] - Omega^2) / Omega^2 for c in interior)
rot_enst_err = maximum(abs(enst_r[c] - 4 * Omega^2) / (4 * Omega^2) for c in interior)

# ## Visualisation — Rotation Fields
N = 16
omega_mat = [omega_r[(j - 1) * N + i] for i in 1:N, j in 1:N]
Q_mat = [Q_r[(j - 1) * N + i] for i in 1:N, j in 1:N]
x_centers = [mesh.cell_centers[1, i] for i in 1:N]
y_centers = [mesh.cell_centers[2, (j - 1) * N + 1] for j in 1:N]

fig1 = Figure(fontsize = 24, size = (1000, 450))
ax1 = Axis(
    fig1[1, 1], xlabel = "x", ylabel = "y",
    title = "ω under rotation (exact 2Ω = 6)", aspect = DataAspect()
)
hm1 = heatmap!(ax1, x_centers, y_centers, omega_mat, colormap = :viridis)
Colorbar(fig1[1, 2], hm1)
ax2 = Axis(
    fig1[1, 3], xlabel = "x", ylabel = "y",
    title = "Q under rotation (exact Ω² = 9)", aspect = DataAspect()
)
hm2 = heatmap!(ax2, x_centers, y_centers, Q_mat, colormap = :plasma)
Colorbar(fig1[1, 4], hm2)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("derived_field_invariants.png"), fig1)
end

# ## Acceptance
@test uniform_max < 1.0e-10 #src
@test shear_omega_err < 1.0e-8 #src
@test shear_Q_max < 1.0e-8 #src
@test rot_omega_err < 1.0e-8 #src
@test rot_Q_err < 1.0e-8 #src
@test rot_enst_err < 1.0e-8 #src
@assert uniform_max < 1.0e-10 #hide
@assert shear_omega_err < 1.0e-8 #hide
@assert shear_Q_max < 1.0e-8 #hide
@assert rot_omega_err < 1.0e-8 #hide
@assert rot_Q_err < 1.0e-8 #hide
@assert rot_enst_err < 1.0e-8 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "uniform_max" => uniform_max,
            "shear_omega_relative_error" => shear_omega_err,
            "shear_Q_max" => shear_Q_max,
            "rotation_omega_relative_error" => rot_omega_err,
            "rotation_Q_relative_error" => rot_Q_err,
            "rotation_enstrophy_relative_error" => rot_enst_err,
        ),
        artifacts = ["derived_field_invariants.png"],
        notes = [
            "Verification-stage exact-solution evidence for postprocessing: vorticity, Q-criterion, and enstrophy match closed-form values on linear flows where the FVM gradient is exact; the shear case discriminates Q from vorticity.",
        ],
        summary = Dict(
            "mesh" => [N, N],
            "interior_cells_checked" => length(interior),
        ),
    )
end

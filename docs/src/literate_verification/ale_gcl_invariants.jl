# # ALE Geometric Conservation Law Invariants
# The Geometric Conservation Law is the cornerstone invariance of ALE
# transport: the swept-face flux $\varphi_{\text{mesh}}$ must satisfy, for
# every cell,
# ```math
# \frac{V^{n+1}_c - V^n_c}{\Delta t} = \sum_f \varepsilon(c, f)\,\varphi_{\text{mesh}, f}.
# ```
# Its failure manifests as artificial mass or energy creation under mesh
# motion. This case establishes GCL exactness (to round-off) on three
# analytically tractable motion patterns:
# 1. **Zero motion** — geometry invariant, $\varphi_{\text{mesh}} \equiv 0$,
#    residual identically zero
# 2. **Rigid translation** — volumes preserved; the closed-cell identity
#    $\sum_f \varepsilon(c,f) S_f = 0$ makes the net swept flux zero
# 3. **Isotropic linear scaling** $d(x) = \alpha(x - x_0)$ — finite volume
#    change with $V^{n+1}/V^n = 1 + \mathrm{Dim}\,\alpha$ to $O(\alpha^2)$,
#    residual at round-off by construction
#
# A refinement sweep confirms the invariant is exact (not merely
# converging) across $N \in \{8, 16, 32\}$.
#
# ## Acceptance Gates
# - Zero motion: residual $= 0$; rigid translation: nondimensional
#   residual $< 10^{-11}$, closed-cell net flux $< 10^{-12}$, volumes
#   preserved to $10^{-12}$
# - Scaling: volume ratio matches $1 + \mathrm{Dim}\,\alpha$ within
#   $5\alpha^2$; residual $< 10^{-10} \bar{V}/\Delta t$
# - Refinement: $\max_c |r_c| \Delta t / V_{\text{cell}} < 10^{-10}$ at
#   every $N$

using FiniteVolumeMethod
using FiniteVolumeMethod: MeshMotionState, compute_displacement!, update_mesh!, verify_gcl
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

# ## Zero Motion
mesh0 = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
ms0 = MeshMotionState(mesh0)
compute_displacement!(ms0, SolidBodyMotion{2, Float64}(t -> SVector(0.0, 0.0)), mesh0, 1.0)
update_mesh!(mesh0, ms0, 1.0e-3)
_, res_zero = verify_gcl(ms0.phi_mesh, ms0.V_old, mesh0.cell_volumes, mesh0, 1.0e-3)
phi_zero = all(==(0.0), ms0.phi_mesh)

# ## Rigid Translation
mesh1 = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
ms1 = MeshMotionState(mesh1)
V1_before = copy(mesh1.cell_volumes)
compute_displacement!(
    ms1, SolidBodyMotion{2, Float64}(t -> SVector(0.25 * t, -0.1 * t)), mesh1, 1.0
)
update_mesh!(mesh1, ms1, 0.1)
_, res_translation = verify_gcl(ms1.phi_mesh, ms1.V_old, mesh1.cell_volumes, mesh1, 0.1)
volume_drift = maximum(
    abs(mesh1.cell_volumes[c] - V1_before[c]) for c in 1:length(V1_before)
)
net_flux = zeros(length(mesh1.cell_volumes))
for f in 1:size(mesh1.face_cells, 2)
    P = mesh1.face_cells[1, f]
    Nb = mesh1.face_cells[2, f]
    net_flux[P] -= ms1.phi_mesh[f]
    if Nb != 0
        net_flux[Nb] += ms1.phi_mesh[f]
    end
end
closed_cell_flux = maximum(abs, net_flux)

# ## Isotropic Scaling
alpha = 0.05
mesh2 = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
ms2 = MeshMotionState(mesh2)
for c in 1:length(mesh2.cell_volumes)
    x = mesh2.cell_centers[1, c] - 0.5
    y = mesh2.cell_centers[2, c] - 0.5
    ms2.displacement[c] = SVector(alpha * x, alpha * y)
end
V2_before = copy(mesh2.cell_volumes)
update_mesh!(mesh2, ms2, 0.1)
_, res_scaling = verify_gcl(ms2.phi_mesh, ms2.V_old, mesh2.cell_volumes, mesh2, 0.1)
V2_mean = sum(V2_before) / length(V2_before)
scaling_residual_ok = res_scaling < 1.0e-10 * V2_mean / 0.1
volume_ratio = sum(mesh2.cell_volumes) / sum(V2_before)
ratio_error = abs(volume_ratio - (1 + 2 * alpha))

# ## Refinement Non-Regression
refine_sizes = [8, 16, 32]
refine_residuals = map(refine_sizes) do N
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    compute_displacement!(
        ms, SolidBodyMotion{2, Float64}(t -> SVector(0.3 * t, 0.2 * t)), mesh, 1.0
    )
    update_mesh!(mesh, ms, 0.1)
    _, max_res = verify_gcl(ms.phi_mesh, ms.V_old, mesh.cell_volumes, mesh, 0.1)
    max_res * 0.1 / (1.0 / (N * N))
end

# ## Visualisation — Invariant Across Refinement
fig1 = Figure(fontsize = 24, size = (600, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "N", ylabel = "max residual · Δt / V_cell",
    yscale = log10, title = "GCL exactness under refinement"
)
scatterlines!(
    ax1, refine_sizes, refine_residuals .+ 1.0e-18, marker = :circle,
    color = :blue, linewidth = 2, markersize = 12
)
hlines!(ax1, [1.0e-10], color = :black, linestyle = :dash, label = "gate")
axislegend(ax1, position = :rt)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("ale_gcl_invariants.png"), fig1)
end

# ## Acceptance
@test res_zero < 1.0e-14 #src
@test phi_zero #src
@test res_translation < 1.0e-11 #src
@test volume_drift < 1.0e-12 #src
@test closed_cell_flux < 1.0e-12 #src
@test scaling_residual_ok #src
@test ratio_error < 5 * alpha^2 #src
@test all(r -> r < 1.0e-10, refine_residuals) #src
@assert res_zero < 1.0e-14 #hide
@assert phi_zero #hide
@assert res_translation < 1.0e-11 #hide
@assert volume_drift < 1.0e-12 #hide
@assert closed_cell_flux < 1.0e-12 #hide
@assert scaling_residual_ok #hide
@assert ratio_error < 5 * alpha^2 #hide
@assert all(r -> r < 1.0e-10, refine_residuals) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "zero_motion_residual" => res_zero,
            "translation_residual" => res_translation,
            "closed_cell_flux" => closed_cell_flux,
            "scaling_volume_ratio" => volume_ratio,
            "refinement_residuals" => refine_residuals,
        ),
        artifacts = ["ale_gcl_invariants.png"],
        notes = [
            "Verification-stage exact-invariant evidence for dynamic_mesh: the ALE Geometric Conservation Law holds to round-off under zero motion, rigid translation, and isotropic linear scaling, and remains exact across mesh refinement.",
        ],
        summary = Dict(
            "alpha" => alpha,
            "refine_sizes" => refine_sizes,
        ),
    )
end

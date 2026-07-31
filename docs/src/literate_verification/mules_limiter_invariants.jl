# # MULES Flux-Limiter Invariants
# The MULES limiter (Multidimensional Universal Limiter for Explicit
# Solution; Weller 2006, a flux-corrected-transport scheme in the sense of
# Zalesak 1979) blends an upwind flux with a high-order flux,
# $F_{\text{lim}} = F_{\text{up}} + \lambda (F_{\text{hi}} - F_{\text{up}})$
# with $\lambda \in [0, 1]$ chosen so the updated $\alpha$ stays in
# $[0, 1]$. Four exact guarantees of that construction are verified on
# `mules_limit_flux!`:
# 1. **Upwind reduction** — $F_{\text{hi}} = F_{\text{up}}$ (no
#    anti-diffusion) gives $F_{\text{lim}} = F_{\text{up}}$ identically
# 2. **Full anti-diffusion when safe** — far from the bounds
#    ($\alpha = 0.5$, tiny $F_{\text{ad}}$), $\lambda$ saturates to 1
# 3. **Segment property** — $F_{\text{lim}}$ lies on the closed segment
#    between $F_{\text{up}}$ and $F_{\text{hi}}$ on every face
# 4. **Boundedness** — one explicit Euler step with the limited flux keeps
#    $\alpha \in [0, 1]$ on every cell, even under an aggressive
#    anti-diffusive flux field
#
# ## Acceptance Gates
# - Reduction exact (bitwise equality); saturation to relative $10^{-12}$
# - Segment property to $10^{-12}$; post-step bounds to $10^{-10}$
# - Degenerate $\Delta t \to 0$ input produces finite output

using FiniteVolumeMethod
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

mules! = FiniteVolumeMethod.mules_limit_flux!

mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
n_cells = length(mesh.cell_volumes)
n_faces = size(mesh.face_cells, 2)

make_flux(value) = FaceFluxField(:phi, mesh; value = value)

# ## Invariant 1 — Upwind Reduction
alpha_half = CollocatedScalarField(:alpha, mesh; value = 0.5)
phi_up = make_flux(0.0)
phi_hi = make_flux(0.0)
for f in 1:n_faces
    phi_up.values[f] = 0.25 * (f % 7 - 3)
    phi_hi.values[f] = phi_up.values[f]
end
limited = make_flux(0.0)
mules!(limited, alpha_half, phi_up, phi_hi, mesh, 0.01)
reduction_exact = all(limited.values[f] == phi_up.values[f] for f in 1:n_faces)

# ## Invariant 2 — Saturation (λ = 1) When Bounds Are Distant
phi_up2 = make_flux(0.0)
phi_hi2 = make_flux(1.0e-6)
limited2 = make_flux(0.0)
mules!(limited2, alpha_half, phi_up2, phi_hi2, mesh, 1.0e-6)
saturation_error = maximum(
    abs(limited2.values[f] - phi_hi2.values[f]) / 1.0e-6 for f in 1:n_faces
)

# ## Invariant 3 — Segment Property
alpha_03 = CollocatedScalarField(:alpha, mesh; value = 0.3)
phi_up3 = make_flux(0.0)
phi_hi3 = make_flux(0.0)
for f in 1:n_faces
    phi_up3.values[f] = 0.1 * sin(f)
    phi_hi3.values[f] = phi_up3.values[f] + 0.5 * cos(f)
end
limited3 = make_flux(0.0)
mules!(limited3, alpha_03, phi_up3, phi_hi3, mesh, 1.0e-3)
segment_ok = all(1:n_faces) do f
    lo = min(phi_up3.values[f], phi_hi3.values[f])
    hi = max(phi_up3.values[f], phi_hi3.values[f])
    lo - 1.0e-12 <= limited3.values[f] <= hi + 1.0e-12
end

# ## Invariant 4 — Boundedness Under Aggressive Anti-Diffusion
phi_up4 = make_flux(0.0)
phi_hi4 = make_flux(0.0)
for f in 1:n_faces
    phi_hi4.values[f] = 0.9 * sin(f)
end
limited4 = make_flux(0.0)
dt = 1.0e-2
mules!(limited4, alpha_half, phi_up4, phi_hi4, mesh, dt)
alpha_next = copy(alpha_half.internal)
for f in 1:n_faces
    F = limited4.values[f] * dt
    P = mesh.face_cells[1, f]
    N = mesh.face_cells[2, f]
    alpha_next[P] -= F / mesh.cell_volumes[P]
    if N != 0
        alpha_next[N] += F / mesh.cell_volumes[N]
    end
end
bounds_ok = all(c -> -1.0e-10 <= alpha_next[c] <= 1.0 + 1.0e-10, 1:n_cells)

# ## Degenerate Input — dt → 0 Stays Finite
limited5 = make_flux(0.0)
mules!(limited5, alpha_half, make_flux(0.1), make_flux(0.2), mesh, 1.0e-10)
finite_ok = all(isfinite, limited5.values)

# ## Visualisation — Segment Property and Post-Step Bounds
fig1 = Figure(fontsize = 24, size = (1000, 450))
ax1 = Axis(
    fig1[1, 1], xlabel = "F_up", ylabel = "F_lim − F_up",
    title = "Limited flux stays on the segment"
)
scatter!(
    ax1, [phi_up3.values[f] for f in 1:n_faces],
    [limited3.values[f] - phi_up3.values[f] for f in 1:n_faces],
    color = :blue, markersize = 8, label = "limited"
)
scatter!(
    ax1, [phi_up3.values[f] for f in 1:n_faces],
    [phi_hi3.values[f] - phi_up3.values[f] for f in 1:n_faces],
    color = :gray, marker = :cross, markersize = 8, label = "high-order"
)
axislegend(ax1, position = :rt)
ax2 = Axis(
    fig1[1, 2], xlabel = "cell", ylabel = "α after limited step",
    title = "Boundedness"
)
scatter!(ax2, 1:n_cells, alpha_next, color = :blue, markersize = 8)
hlines!(ax2, [0.0, 1.0], color = :black, linestyle = :dash)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("mules_invariants.png"), fig1)
end

# ## Acceptance
@test reduction_exact #src
@test saturation_error < 1.0e-12 #src
@test segment_ok #src
@test bounds_ok #src
@test finite_ok #src
@assert reduction_exact #hide
@assert saturation_error < 1.0e-12 #hide
@assert segment_ok #hide
@assert bounds_ok #hide
@assert finite_ok #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "saturation_relative_error" => saturation_error,
            "alpha_post_step_min" => minimum(alpha_next),
            "alpha_post_step_max" => maximum(alpha_next),
            "faces_checked" => n_faces,
        ),
        artifacts = ["mules_invariants.png"],
        notes = [
            "Benchmark-stage evidence for multiphase_vof: the four exact FCT guarantees of the MULES face-flux limiter (Zalesak 1979; Weller 2006) — upwind reduction, safe saturation, segment property, and post-step boundedness.",
        ],
        summary = Dict(
            "mesh" => [8, 8],
            "n_faces" => n_faces,
        ),
    )
end

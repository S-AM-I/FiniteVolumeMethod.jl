# # Log-Layer Equilibrium (Standard k-ε)
# In the inertial sublayer of a wall-bounded turbulent flow the standard
# k-ε closure admits an exact local equilibrium: production of turbulent
# kinetic energy balances dissipation ($P_k = \varepsilon$) under the
# log-law scaling (Launder & Spalding 1974 constants,
# $\kappa = 0.41$, $C_\mu = 0.09$):
# ```math
# U(y) = \frac{u_\tau}{\kappa} \ln\frac{y}{y_0}, \qquad
# k = \frac{u_\tau^2}{\sqrt{C_\mu}}, \qquad
# \varepsilon(y) = \frac{u_\tau^3}{\kappa y}.
# ```
# From these, $\partial U/\partial y = u_\tau/(\kappa y)$,
# $\nu_t = C_\mu k^2/\varepsilon = \kappa y u_\tau$, and therefore
# $P_k = \nu_t\,|S|^2 = u_\tau^3/(\kappa y) = \varepsilon$ — the ratio
# $P_k/\varepsilon \equiv 1$ at every height. This closed-form invariant is
# verified cell-by-cell on a prescribed field, exercising the discrete
# strain-rate operator `compute_strain_rate` on a real mesh.
#
# ## Acceptance Gates
# - $P_k/\varepsilon \in (0.85, 1.15)$ over the interior band (the 15%
#   margin covers the $O(h^2\,\mathrm{d}^2U/\mathrm{d}y^2)$ truncation of
#   the discrete gradient on a logarithmic profile)
# - Algebraic invariant $\nu_t = \kappa y u_\tau$ to relative $10^{-12}$
# - Durbin realizability cap ($\alpha = 0.6$) inactive throughout the
#   equilibrium state

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_strain_rate
using StaticArrays
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

u_tau = 0.05
kappa = 0.41
C_mu = 0.09

# ## Prescribed Log-Layer State
# The physical wall coordinate is $y_{\text{phys}} = y_{\text{mesh}} + y_0$
# with $y_0 = 0.05$, keeping the $\varepsilon = u_\tau^3/(\kappa y)$
# singularity outside the domain.
y_offset = 0.05
Nx = 8
Ny = 40
Lx = 1.0
Ly = 0.5

mesh = build_cartesian_unstructured_mesh(Nx, Ny, Lx, Ly)
n_cells = length(mesh.cell_volumes)

U = CollocatedVectorField(:U, mesh)
for c in 1:n_cells
    y_phys = mesh.cell_centers[2, c] + y_offset
    U.internal[c] = SVector((u_tau / kappa) * log(y_phys / y_offset), 0.0)
end

k_val = fill(u_tau^2 / sqrt(C_mu), n_cells)
eps_val = [u_tau^3 / (kappa * (mesh.cell_centers[2, c] + y_offset)) for c in 1:n_cells]
nu_t = [C_mu * k_val[c]^2 / eps_val[c] for c in 1:n_cells]

# ## Discrete Production vs Dissipation
# $|S|$ comes from the FVM gradient of the prescribed $U$; production is
# $P_k = \nu_t |S|^2$. The interior band excludes boundary-stencil cells and
# the high-curvature region near the wall.
S_mag = compute_strain_rate(U, mesh)
P_k = [nu_t[c] * S_mag[c]^2 for c in 1:n_cells]

interior = [
    c for c in 1:n_cells
        if 0.3 * Ly < mesh.cell_centers[2, c] < 0.7 * Ly &&
        0.2 * Lx < mesh.cell_centers[1, c] < 0.8 * Lx
]
ratios = [P_k[c] / eps_val[c] for c in interior]

# ## Algebraic Invariant and Realizability Cap
# Independent of the discrete gradient: $\nu_t = \kappa y u_\tau$ exactly,
# and the Durbin cap $\nu_t \leq \alpha k / |S|$ evaluates to
# $(\alpha/\sqrt{C_\mu})\,\kappa y u_\tau \approx 2\nu_t$ at $\alpha = 0.6$ —
# inactive by design in equilibrium.
alpha_durbin = 0.6
y_sweep = range(0.01, 0.5; length = 20)
invariant_errors = map(y_sweep) do y
    k_local = u_tau^2 / sqrt(C_mu)
    eps_local = u_tau^3 / (kappa * y)
    nu_t_local = C_mu * k_local^2 / eps_local
    abs(nu_t_local - kappa * y * u_tau) / (kappa * y * u_tau)
end
cap_inactive = all(y_sweep) do y
    k_local = u_tau^2 / sqrt(C_mu)
    S_local = u_tau / (kappa * y)
    nu_t_local = kappa * y * u_tau
    nu_t_local < alpha_durbin * k_local / S_local
end

# ## Visualisation — Equilibrium Ratio
ys_interior = [mesh.cell_centers[2, c] for c in interior]
fig1 = Figure(fontsize = 24, size = (600, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "P_k / ε", ylabel = "y",
    title = "Log-layer equilibrium (interior band)"
)
vlines!(ax1, [1.0], color = :black, linewidth = 2, label = "equilibrium")
vlines!(ax1, [0.85, 1.15], color = :gray, linestyle = :dash, label = "±15% gate")
scatter!(ax1, ratios, ys_interior, color = :blue, markersize = 8, label = "cells")
axislegend(ax1, position = :rt)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("kepsilon_loglayer_ratio.png"), fig1)
end

# ## Visualisation — Prescribed Profiles
y_dense = range(0.0, Ly; length = 200)
fig2 = Figure(fontsize = 24, size = (900, 400))
axU = Axis(fig2[1, 1], xlabel = "U(y)", ylabel = "y", title = "Log-law U")
lines!(
    axU, (u_tau / kappa) .* log.((y_dense .+ y_offset) ./ y_offset), y_dense,
    color = :blue, linewidth = 2
)
axE = Axis(fig2[1, 2], xlabel = "ε(y)", ylabel = "y", title = "Dissipation")
lines!(
    axE, u_tau^3 ./ (kappa .* (y_dense .+ y_offset)), y_dense,
    color = :red, linewidth = 2
)
axN = Axis(fig2[1, 3], xlabel = "ν_t(y)", ylabel = "y", title = "Eddy viscosity")
lines!(
    axN, kappa .* (y_dense .+ y_offset) .* u_tau, y_dense,
    color = :green, linewidth = 2
)
resize_to_layout!(fig2)
fig2
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("kepsilon_loglayer_profiles.png"), fig2)
end

# ## Acceptance
@test length(interior) > 10 #src
@test all(r -> 0.85 < r < 1.15, ratios) #src
@test all(e -> e < 1.0e-12, invariant_errors) #src
@test cap_inactive #src
@assert length(interior) > 10 #hide
@assert all(r -> 0.85 < r < 1.15, ratios) #hide
@assert all(e -> e < 1.0e-12, invariant_errors) #hide
@assert cap_inactive #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "ratio_min" => minimum(ratios),
            "ratio_max" => maximum(ratios),
            "cells_checked" => length(interior),
            "max_invariant_relative_error" => maximum(invariant_errors),
        ),
        artifacts = ["kepsilon_loglayer_ratio.png", "kepsilon_loglayer_profiles.png"],
        notes = [
            "Benchmark-stage evidence for turbulence_rans: the P_k = epsilon local equilibrium of the standard k-epsilon closure under log-law scaling, with production computed from the discrete strain-rate operator on a real mesh.",
            "Complements the DHIT decay verification case, which covers the source terms without shear.",
        ],
        summary = Dict(
            "u_tau" => u_tau,
            "kappa" => kappa,
            "C_mu" => C_mu,
            "mesh" => [Nx, Ny],
        ),
    )
end

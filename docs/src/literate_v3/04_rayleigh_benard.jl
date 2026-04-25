# # Tutorial 04 — Rayleigh-Bénard Convection (Boussinesq)
#
# Demonstrates the v3 thermal solver with Boussinesq buoyancy. We
# heat the bottom plate, cool the top, and let buoyancy drive the
# circulation. To stay under the 30-second budget we run at
# `Ra ≈ 1 × 10³` (sub-critical for a unit-aspect cell, but enough to
# exercise every coupling on a 12×12 mesh).
#
# Runtime budget: ~5–10 s on a laptop (12×12 mesh, 50 outer iterations).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/04_rayleigh_benard.jl
# ```
#
# What it demonstrates:
# - `FluidThermalProperties` with `beta` set to enable Boussinesq buoyancy
# - `solve_simple_thermal` energy + momentum coupling
# - Computing a bulk Nusselt number from the cell-average temperature
#   gradient near the heated wall

using FiniteVolumeMethod
using LinearSolve
using StaticArrays
using Printf

include(joinpath(@__DIR__, "..", "..", "..", "test", "TestHelpers.jl"))

const L = 1.0
const T_hot = 305.0
const T_cold = 295.0
const ΔT = T_hot - T_cold
const T_ref = 0.5 * (T_hot + T_cold)

# Boussinesq with diffusion-dominated parameters: we pick relatively
# large ν, α so the velocity stays bounded on a 12×12 mesh in 50
# SIMPLE iterations. The resulting Ra is subcritical (< 1708) so the
# solution is essentially conductive with a weak buoyancy-driven
# secondary flow — enough to exercise every coupling term without
# blowing up.
const β = 1.0e-3
const g = 9.81
const ν = 0.05
const α_th = 0.05      # Pr = 1
const Cp = 1005.0
const ρ = 1.0
const k_th = α_th * ρ * Cp
const Ra = β * g * ΔT * L^3 / (ν * α_th)

mesh = build_cartesian_unstructured_mesh(12, 12, L, L)

bcs = Dict{Symbol, AbstractBoundaryCondition}(
    :left => SlipWallBC(),
    :right => SlipWallBC(),
    :bottom => NoSlipWallBC(),
    :top => NoSlipWallBC(),
)

algo = SIMPLE(; alpha_U = 0.5, alpha_p = 0.3, max_iterations = 50, tolerance = 1.0e-12)
prob = IncompressibleProblem(mesh, bcs, algo; nu = ν, density = ρ)

# Buoyancy is enabled by passing `beta != 0` and `g`. The Boussinesq
# source uses `(T - T_ref)`.
thermal_props = FluidThermalProperties{2}(;
    Cp = 1005.0, k = k_th,
    beta = β, T_ref = T_ref,
    g = SVector(0.0, -g),
)

bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
    :left => thermal_insulated_bc(),
    :right => thermal_insulated_bc(),
    :bottom => thermal_inlet_bc(T_hot),
    :top => thermal_inlet_bc(T_cold),
)

result, thermal_state = solve_simple_thermal(
    prob, thermal_props;
    bcs_T = bcs_T,
    T_init = T_ref + 1.0,  # small offset triggers buoyancy on iteration 1
)

# Compute a crude Nusselt number: average heat flux through the bottom
# plate divided by the conductive flux ΔT/L. We approximate the wall
# gradient with a one-sided difference between the bottom row of cells
# and the prescribed wall temperature.
function bottom_wall_flux(T_internal, T_hot, L, nx, ny)
    dx, dy = L / nx, L / ny
    flux_total = 0.0
    for i in 1:nx
        c = (1 - 1) * nx + i  # bottom row
        grad = (T_internal[c] - T_hot) / (dy / 2)
        flux_total += -grad * dx
    end
    return flux_total / L
end

nx, ny = 12, 12
T_internal = thermal_state.T_field.internal
q_avg = bottom_wall_flux(T_internal, T_hot, L, nx, ny)
Nu = q_avg * L / (α_th * ΔT)  # Nu = q_wall · L / (α · ΔT)

println("=== Rayleigh-Bénard Boussinesq ===")
@printf "Ra (target)       : %.2e\n" Ra
@printf "ν                 : %.4e m²/s\n" ν
@printf "α (thermal)       : %.4e m²/s\n" α_th
@printf "iterations        : %d\n" result.iterations
@printf "converged         : %s\n" result.converged
@printf "T min / max       : %.2f / %.2f K\n" minimum(T_internal) maximum(T_internal)
@printf "max |U|           : %.4e m/s\n" maximum(u -> abs(u[1]) + abs(u[2]), result.state.U.internal)
@printf "Nusselt (bulk)    : %.3f\n" Nu

# KNOWN ISSUE: On a 12×12 mesh the Boussinesq solver diverges for
# Ra ≳ 10³ unless the momentum equation is strongly under-relaxed
# (`alpha_U` ≲ 0.3) AND the buoyancy term is ramped across iterations.
# This tutorial runs at Ra ≈ 40 to stay robust under default
# under-relaxation. For the Ra = 1e4 / 1e5 published-benchmark cases
# see the V&V suite.
#
# The scripted Nusselt number above is a coarse one-sided-difference
# estimate across the bottom cell row; below Ra_critical ≈ 1708 it
# should asymptote to Nu ≈ 1 (pure conduction), but the crude
# gradient formula over 12 cells gives an order-of-magnitude check,
# not an accurate number.
#
# Manifest feature  : phase3.boussinesq_thermal (experimental)
# V&V tests         : test/thermal.jl, test/v_and_v_thermal_types.jl

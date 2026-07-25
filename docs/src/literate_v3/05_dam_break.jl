# # Tutorial 05 — 2D Dam-Break (VOF Multiphase)
#
# Demonstrates the v3 volume-of-fluid multiphase solver on a short
# dam-break run. We initialise the left half of the domain as water
# (α = 1) and the right half as air (α = 0), release gravity, and
# advance a few PISO outer iterations.
#
# Runtime budget: ~5 s on a laptop (16×8 mesh, 2 time steps).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/05_dam_break.jl
# ```
#
# What it demonstrates:
# - `TwoPhaseProperties` for water/air with surface tension
# - `solve_vof` signature: mesh + BCs + tspan + dt
# - Pulling the mixture density / α field after the run
#
# KNOWN ISSUE: The α transport uses hard clipping instead of full
# MULES/isoAdvector, so for longer runs or finer meshes you will see
# interface smearing. See `src/multiphase/boundedness.jl` and the v3
# roadmap for MULES integration.

using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: NeumannBC
using LinearSolve
using StaticArrays
using Printf

# Located relative to the installed package rather than to this file, so the
# path resolves both when run as a script and when Literate executes it from
# the generated-docs directory.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

const Lx = 2.0
const Ly = 1.0
mesh = build_cartesian_unstructured_mesh(16, 8, Lx, Ly)

# Default TwoPhaseProperties maps to water (ρ=1000, μ=1e-3) over
# air (ρ=1, μ=1.8e-5) with σ = 0.072 N/m.
props = TwoPhaseProperties()

# All walls no-slip on velocity; fix pressure at the right wall so the
# pressure Poisson problem is not singular.
bcs_U = Dict{Symbol, AbstractBoundaryCondition}(
    :left => NoSlipWallBC(),
    :right => NoSlipWallBC(),
    :bottom => NoSlipWallBC(),
    :top => NoSlipWallBC(),
)
bcs_p = Dict{Symbol, AbstractBoundaryCondition}(
    :left => NeumannBC(0.0),
    :right => FixedPressureBC(0.0),
    :bottom => NeumannBC(0.0),
    :top => NeumannBC(0.0),
)
bcs_alpha = Dict{Symbol, AbstractBoundaryCondition}(
    :left => NeumannBC(0.0),
    :right => NeumannBC(0.0),
    :bottom => NeumannBC(0.0),
    :top => NeumannBC(0.0),
)

# Initial condition: water fills the left half of the domain.
alpha_init_func = x -> x[1] < 0.5 * Lx ? 1.0 : 0.0

# Gravity acts downward in -y. For a short tspan the front motion is
# dominated by the pressure release, not by fully-developed free
# surface dynamics.
g = SVector(0.0, -9.81)

result, vof_state = solve_vof(
    mesh, props, bcs_U, bcs_p, bcs_alpha,
    (0.0, 0.02), 0.01;
    alpha_init = alpha_init_func,
    g = g,
    algorithm = PISO(; n_correctors = 1),
)

# Compute a rough "front" — the x-coordinate of the rightmost water cell
# at the bottom row.
nx, ny = 16, 8
dx = Lx / nx
bottom_row = [(1 - 1) * nx + i for i in 1:nx]
water_cells_bottom = [c for c in bottom_row if vof_state.alpha.internal[c] > 0.5]
x_front = isempty(water_cells_bottom) ? 0.0 : mesh.cell_centers[1, maximum(water_cells_bottom)] + dx / 2

println("=== 2D dam-break (VOF) ===")
@printf "PISO outer its    : %d\n" result.iterations
@printf "converged         : %s\n" result.converged
@printf "α ∈ [0, 1]?       : %s\n" all(x -> 0.0 <= x <= 1.0, vof_state.alpha.internal)
@printf "α sum (mass)      : %.4f\n" sum(vof_state.alpha.internal)
@printf "ρ min / max       : %.2f / %.2f kg/m³\n" minimum(vof_state.rho) maximum(vof_state.rho)
@printf "front position    : %.3f m (initial = %.3f)\n" x_front 0.5 * Lx

# Manifest feature  : phase7.vof_incompressible (experimental)
# V&V tests         : test/multiphase_vof.jl

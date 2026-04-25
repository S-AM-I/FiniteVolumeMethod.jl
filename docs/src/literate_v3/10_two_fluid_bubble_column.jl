# # Tutorial 10 — Two-Fluid Bubble Column (Eulerian-Eulerian API)
#
# Demonstrates the v3 Eulerian two-fluid solver on a small 4×4 closed
# domain. A 30% gas void fraction is initialised uniformly and the
# solver is run for a handful of outer iterations with gravity off —
# the intent is to show API wiring (drag, mass transfer, per-phase
# BCs) rather than to reproduce a physical bubble column.
#
# Runtime budget: ~3 s on a laptop (4×4 mesh, 5 outer iterations).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/10_two_fluid_bubble_column.jl
# ```
#
# What it demonstrates:
# - `TwoFluidProperties` with distinct ρ, μ for liquid and gas
# - `TwoFluidProblem` with per-phase BCs (`bcs_Ul`, `bcs_Ug`, `bcs_p`)
# - `solve_two_fluid` returning a `TwoFluidSolveResult`
# - Verifying total gas volume fraction is preserved in a closed
#   zero-gravity system
#
# KNOWN ISSUE: The α transport falls back to upwind + hard clip
# (no MULES) and non-orthogonal correction is absent. Both are
# documented roadmap items for v3.1+.

using FiniteVolumeMethod
using LinearAlgebra: norm
using LinearSolve  # registers default linear solver
using StaticArrays
using Printf

include(joinpath(@__DIR__, "..", "..", "..", "test", "TestHelpers.jl"))

# `solve_two_fluid` and `TwoFluidProblem` are not exported at the top
# level; we pull them from the FiniteVolumeMethod module directly.
const solve_two_fluid = FiniteVolumeMethod.solve_two_fluid
const TwoFluidProblem = FiniteVolumeMethod.TwoFluidProblem

mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)

props = TwoFluidProperties(;
    rho_l = 1000.0, rho_g = 1.2,
    mu_l = 1.0e-3, mu_g = 1.81e-5,
    sigma = 0.072, d_b = 1.0e-3, C_D = 1.0,
)

noslip = NoSlipWallBC()
bcs_Ul = Dict{Symbol, AbstractBoundaryCondition}(
    :left => noslip, :right => noslip,
    :bottom => noslip, :top => noslip,
)
bcs_Ug = copy(bcs_Ul)
bcs_p = Dict{Symbol, AbstractBoundaryCondition}()

# Zero gravity — we want the conservation check, not a physical run.
prob = TwoFluidProblem(
    mesh, props;
    bcs_Ul = bcs_Ul, bcs_Ug = bcs_Ug, bcs_p = bcs_p,
    gravity = SVector(0.0, 0.0),
)

alpha_init = 0.3
result = solve_two_fluid(
    prob, TwoFluidSolver();
    alpha_g_init = alpha_init, dt = 1.0e-2, max_outer = 5,
    tol = 1.0e-8, verbose = false,
)

max_Ul = maximum(norm, result.state.U_l.internal)
max_Ug = maximum(norm, result.state.U_g.internal)

V = mesh.cell_volumes
total_alpha_g_V = sum(result.state.alpha_g.internal .* V)
expected_V = alpha_init * sum(V)

println("=== Eulerian-Eulerian two-fluid bubble column ===")
@printf "outer iterations    : %d\n" result.iterations
@printf "converged           : %s\n" result.converged
@printf "max |U_l|           : %.2e m/s\n" max_Ul
@printf "max |U_g|           : %.2e m/s\n" max_Ug
@printf "∑ α_g · V (actual)  : %.6f\n" total_alpha_g_V
@printf "∑ α_g · V (expected): %.6f\n" expected_V
@printf "relative error      : %.2e\n" abs(total_alpha_g_V - expected_V) / expected_V

# Manifest feature  : phase7.two_fluid_eulerian (experimental)
# V&V tests         : test/v_and_v_two_fluid_solver.jl

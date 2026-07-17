# # Tutorial 09 — Dynamic-Mesh Oscillator (ALE)
#
# Demonstrates the v3 arbitrary-Lagrangian-Eulerian (ALE) solver with
# a rigid-body motion. We translate the mesh sinusoidally in x and
# solve for an incompressible flow on the moving mesh over two PISO
# corrector steps. This is not a 6-DOF rigid-body oscillator (that
# would couple the fluid force back to the solid) but shows the API
# surface for prescribed-motion ALE runs.
#
# Runtime budget: ~3 s on a laptop (8×8 mesh, 2 time steps).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/09_dynamic_mesh_oscillator.jl
# ```
#
# What it demonstrates:
# - `SolidBodyMotion` wrapping a displacement closure `t -> SVector(...)`
# - `solve_ale` signature: mesh + motion + bcs_U + bcs_p + tspan + dt
# - PISO dispatch for transient incompressible flow on a moving mesh
#
# KNOWN ISSUE: Only prescribed motion is implemented — a true 6-DOF
# coupling would read forces via `compute_forces` and feed them back
# into a rigid-body ODE integrator. See `src/dynamic_mesh/` and the
# v3 roadmap.

using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: NeumannBC
using LinearSolve
using StaticArrays
using Printf

include(joinpath(@__DIR__, "..", "..", "..", "test", "TestHelpers.jl"))

mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)

# Sinusoidal x-displacement: u_mesh(t) = 0.01 sin(2π f t).
const f_osc = 2.0   # Hz
motion = SolidBodyMotion{2, Float64}(t -> SVector(0.01 * sin(2π * f_osc * t), 0.0))

bcs_U = Dict{Symbol, AbstractBoundaryCondition}(
    :left => NoSlipWallBC(),
    :right => NoSlipWallBC(),
    :bottom => NoSlipWallBC(),
    :top => FixedVelocityBC((1.0, 0.0)),
)
bcs_p = Dict{Symbol, AbstractBoundaryCondition}(
    :left => NeumannBC(0.0),
    :right => NeumannBC(0.0),
    :bottom => NeumannBC(0.0),
    :top => NeumannBC(0.0),
)

result = solve_ale(
    mesh, motion, bcs_U, bcs_p,
    (0.0, 0.02), 0.01;
    nu = 0.01,
    algorithm = PISO(; n_correctors = 1),
)

println("=== ALE with prescribed body motion ===")
@printf "PISO outer its     : %d\n" result.iterations
@printf "converged          : %s\n" result.converged
@printf "continuity history : "
for r in result.residuals[:continuity]
    @printf "%.3e  " r
end
println()

# Check mesh was actually displaced: at t = 0.01 the prescribed
# displacement is 0.01 · sin(2π · 2 · 0.01) ≈ 1.25e-3.
x_expected = 0.01 * sin(2π * f_osc * 0.01)
@printf "expected |x_disp|@t=0.01 : %.4e\n" abs(x_expected)

# Manifest feature  : phase10.ale_prescribed_motion (experimental)
# V&V tests         : test/dynamic_mesh.jl

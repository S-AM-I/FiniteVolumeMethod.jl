# # Tutorial 02 — Compressible Low-Mach Channel (CompressibleSIMPLE)
#
# Demonstrates the `CompressibleSIMPLE` (rhoSimpleFoam analogue) on a
# small driven channel using `IdealGas` with a constant-μ closure. At
# very low Mach number the compressible solution should track the
# incompressible reference; this tutorial seeds the user with the
# correct API for the density-coupled momentum/pressure loop.
#
# Runtime budget: ~5 s on a laptop (8×8 mesh, 100 outer iterations).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/02_compressible_channel.jl
# ```
#
# What it demonstrates:
# - `IdealGas` thermodynamic model (γ = 1.4, R = 287.05 J/(kg·K), μ = 1.8e-5)
# - `CompressibleProblem` with a `CompressibleSIMPLE` algorithm
# - `solve_compressible` returning a state with both `base.U` (velocity)
#   and `rho` (cell density)
# - Reading the EOS-coupled density update at the end of the run

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: IdealGas
using LinearAlgebra: norm
using LinearSolve
using StaticArrays
using Printf

# Located relative to the installed package rather than to this file, so the
# path resolves both when run as a script and when Literate executes it from
# the generated-docs directory.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

# An 8×8 driven cavity at Mach ≈ 3e-6 (lid speed = 1e-3 m/s).
mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)

bcs = Dict{Symbol, AbstractBoundaryCondition}(
    :left => NoSlipWallBC(),
    :right => NoSlipWallBC(),
    :bottom => NoSlipWallBC(),
    :top => FixedVelocityBC(SVector(1.0e-3, 0.0)),
)

# IdealGas with an internally constant μ. Sutherland's law is wired up
# as `Sutherland()` — swap if you want temperature-dependent viscosity.
gas = IdealGas(; gamma = 1.4, R = 287.05, mu = 1.8e-5)

# NOTE: `CompressibleSIMPLE` and `CompressibleProblem` are not yet
# exported from the top-level module — they live as
# `FiniteVolumeMethod.CompressibleSIMPLE` etc. Import them locally for
# convenience.
const CompressibleSIMPLE = FiniteVolumeMethod.CompressibleSIMPLE
const CompressibleProblem = FiniteVolumeMethod.CompressibleProblem

algo = CompressibleSIMPLE(;
    alpha_U = 0.7, alpha_p = 0.3, alpha_rho = 0.9,
    max_iterations = 100, tolerance = 1.0e-8,
)
prob = CompressibleProblem(mesh, bcs, algo, gas; T_ref = 300.0, solve_energy = false)

result = FiniteVolumeMethod.solve_compressible(
    prob;
    linear_solver = LUFactorization(),
    p0 = 1.01325e5, verbose = false,
)

# `result.state` is a `CompressibleState` with a nested IncompressibleState
# in `.base` plus density / temperature / face-density / μ vectors.
nc = length(mesh.cell_volumes)
U = result.state.base.U.internal
ρ = result.state.rho

ρ_ref = 1.01325e5 / (gas.R * 300.0)

println("=== Compressible SIMPLE channel (low-Mach) ===")
@printf "iterations           : %d\n" result.iterations
@printf "converged            : %s\n" result.converged
@printf "max |U|              : %.4e m/s\n" maximum(norm, U)
@printf "ρ_ref (EOS at p0,T0) : %.4f kg/m³\n" ρ_ref
@printf "min ρ (cell)         : %.6f\n" minimum(ρ)
@printf "max ρ (cell)         : %.6f\n" maximum(ρ)
@printf "Δρ / ρ_ref           : %.2e\n" (maximum(ρ) - minimum(ρ)) / ρ_ref

# At Mach ≈ 3e-6 the density should stay within 1% of ρ_ref. The
# velocity field should be identical to the incompressible SIMPLE run
# at the corresponding (μ, ρ).
#
# Manifest feature  : pressure_based.compressible_simple (experimental)
# V&V tests         : test/v_and_v_compressible.jl

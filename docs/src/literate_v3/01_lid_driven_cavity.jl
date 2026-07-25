# # Tutorial 01 — Lid-Driven Cavity (Incompressible SIMPLE)
#
# This tutorial walks through the canonical 2D lid-driven cavity at
# `Re = 100` using the v3 collocated incompressible solver with the
# steady-state SIMPLE pressure-velocity coupling.
#
# Runtime budget: ~5–10 s on a laptop (32×32 mesh, 200 SIMPLE iterations).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/01_lid_driven_cavity.jl
# ```
#
# What it demonstrates:
# - Building a Cartesian unstructured mesh via the test helper
# - Wiring `NoSlipWallBC` walls and a moving top lid via `FixedVelocityBC`
# - Running `solve(prob, SIMPLE())` and pulling out fields with the SciML
#   symbolic indexing (`sol[:U]`, `sol[:p]`)
# - Sampling the centre-line `u(y)` profile so the user has a numerical
#   sanity check that ties the run to Ghia et al. (1982).

using FiniteVolumeMethod
using LinearSolve
using StaticArrays
using Printf

# We reuse the tutorial mesh helper that ships with the test suite. It
# builds a Cartesian unstructured mesh with patches `:left`, `:right`,
# `:bottom`, `:top` and the face orientation conventions that the
# collocated assembly expects.
# Located relative to the installed package rather than to this file, so the
# path resolves both when run as a script and when Literate executes it from
# the generated-docs directory.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

# Problem parameters: L = 1, U_lid = 1 → Re = U_lid · L / ν.
const L = 1.0
const U_lid = 1.0
const Re = 100.0
const ν = U_lid * L / Re

mesh = build_cartesian_unstructured_mesh(32, 32, L, L)

bcs = Dict{Symbol, AbstractBoundaryCondition}(
    :left => NoSlipWallBC(),
    :right => NoSlipWallBC(),
    :bottom => NoSlipWallBC(),
    :top => FixedVelocityBC((U_lid, 0.0)),
)

# A pressure reference is fixed automatically inside SIMPLE; we leave
# the four wall patches as no-slip / moving wall.
algo = SIMPLE(; alpha_U = 0.7, alpha_p = 0.3, max_iterations = 200, tolerance = 1.0e-7)
prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = ν, density = 1.0)

# Solve — the SciML-style `solve(prob, alg)` returns an
# `IncompressibleSolution` with symbolic field access.
sol = solve(prob, algo; linear_solver = LUFactorization())

# Pull out the velocity field. `sol[:U]` is `Vector{SVector{2, Float64}}`.
U = sol[:U]
p = sol[:p]

# Sample the geometric centre-line x = L/2: collect cells whose centre
# lies on the column closest to `L/2` and sort by `y`.
nx, ny = 32, 32
dx = L / nx
i_mid = nx ÷ 2  # column index just left of centre
center_cells = [(j - 1) * nx + i_mid for j in 1:ny]
ys = [mesh.cell_centers[2, c] for c in center_cells]
us = [U[c][1] for c in center_cells]

println("=== Lid-driven cavity Re=$(Int(Re)) ===")
@printf "iterations            : %d\n" sol.iterations
@printf "converged             : %s\n" sol.converged
@printf "continuity residual   : %.3e\n" sol.residuals[:continuity][end]
@printf "momentum-x residual   : %.3e\n" sol.residuals[:Ux][end]
@printf "min  u                : %+.4f\n" minimum(us)
@printf "max  u                : %+.4f\n" maximum(us)
@printf "u at y = %.3f         : %+.4f\n" ys[ny ÷ 2] us[ny ÷ 2]

# Sanity check: at Re=100 the Ghia centre-line minimum is ≈ -0.21.
# On a 32×32 mesh we expect single-digit-percent agreement on the gross
# magnitude, dropping below 5% near the lid where the gradient is
# strongest. See `validation/manifest.toml` ("incompressible_simple")
# and `test/incompressible_sciml.jl` for the published-benchmark suite
# this tutorial subsets.
#
# Manifest feature  : phase1.simple_collocated (experimental)
# V&V tests         : test/incompressible.jl, test/incompressible_sciml.jl

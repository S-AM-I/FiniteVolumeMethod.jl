# # Tutorial 07 — P1 Radiation Model in an Enclosure
#
# Demonstrates the P1 (spherical harmonics P₁) radiation model on a
# small enclosure with Marshak wall boundary conditions. We solve the
# radiation intensity diffusion equation for the incident radiation
# field `G` at a uniform wall temperature and verify it recovers the
# equilibrium value `4 σ T⁴`.
#
# Runtime budget: ~2 s on a laptop (6×6 mesh, single linear solve).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/07_radiation_p1.jl
# ```
#
# What it demonstrates:
# - `P1Model` construction with absorption coefficient `a`
# - `marshak_wall_bc(rad, T_wall)` helper for radiative walls
# - `solve_p1_radiation` returning a `CollocatedScalarField{Float64}`
# - Sanity-checking against the Stefan-Boltzmann equilibrium

using FiniteVolumeMethod
using LinearSolve  # registers the default sparse direct solver
using Printf

# Located relative to the installed package rather than to this file, so the
# path resolves both when run as a script and when Literate executes it from
# the generated-docs directory.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)

# P1 radiation with absorption coefficient a = 1.0 (moderate optical
# depth). Scattering is absent in the current implementation.
rad = P1Model(; a = 1.0)

# Uniform wall temperature so the analytical equilibrium is
# G_eq = 4 σ T⁴ throughout the domain.
T_wall = 1500.0
T_field = FiniteVolumeMethod.CollocatedScalarField(:T, mesh; value = T_wall)

bcs_G = Dict{Symbol, AbstractBoundaryCondition}(
    :left => marshak_wall_bc(rad, T_wall),
    :right => marshak_wall_bc(rad, T_wall),
    :bottom => marshak_wall_bc(rad, T_wall),
    :top => marshak_wall_bc(rad, T_wall),
)

G = solve_p1_radiation(rad, T_field, mesh, bcs_G)

G_eq = 4 * STEFAN_BOLTZMANN * T_wall^4

println("=== P1 radiation in an enclosure ===")
@printf "T_wall (uniform)     : %.1f K\n" T_wall
@printf "Stefan-Boltzmann σ   : %.4e W/(m²·K⁴)\n" STEFAN_BOLTZMANN
@printf "G equilibrium target : %.4e W/m²\n" G_eq
@printf "min G (interior)     : %.4e\n" minimum(G.internal)
@printf "max G (interior)     : %.4e\n" maximum(G.internal)
@printf "mean G / G_eq        : %.4f\n" (sum(G.internal) / length(G.internal)) / G_eq

# For a uniform T field with Marshak BCs the solution is uniform in
# the infinite-medium limit; on a 6×6 mesh we expect G to be within
# ~10–20% of the analytical equilibrium value. The V&V tests run at
# `rtol = 0.2`.
#
# Manifest feature  : phase9.p1_radiation (experimental)
# V&V tests         : test/radiation.jl, test/v_and_v_radiation_source.jl

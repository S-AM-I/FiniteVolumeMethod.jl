# # Tutorial 03 — k-ε Turbulent Channel
#
# Demonstrates the v3 RANS turbulence stack on a small driven channel.
# We use the standard k-ε model with `DirichletBC` inlet
# turbulence values and zero-gradient outflow.
#
# Runtime budget: ~5–10 s on a laptop (16×8 mesh, 30 outer iterations).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/03_kepsilon_channel.jl
# ```
#
# What it demonstrates:
# - Wiring `StandardKEpsilon` into `solve_simple_turbulent`
# - Specifying separate BC dicts for `:k` and `:epsilon`
# - Reading the turbulence state (`k`, `ε`, `ν_t`) and computing a
#   bulk friction velocity from the wall shear stress
#
# KNOWN ISSUE: At very small grid counts, k-ε's epsilon equation can
# spuriously plateau because the production term dominates the local
# dissipation budget; bumping `max_iterations` does not help. This
# tutorial therefore prints a `u_τ` estimate but does not assert
# log-law agreement on the coarse mesh — see the V&V suite for that
# (`test/turbulence_rans.jl`).

using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using LinearSolve
using LinearAlgebra: norm
using StaticArrays
using Printf

# Located relative to the installed package rather than to this file, so the
# path resolves both when run as a script and when Literate executes it from
# the generated-docs directory.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

const Lx = 2.0
const Ly = 1.0
const ν = 0.01
const U_in = 0.1

mesh = build_cartesian_unstructured_mesh(8, 4, Lx, Ly)

bcs = Dict{Symbol, AbstractBoundaryCondition}(
    :left => FixedVelocityBC((U_in, 0.0)),
    :right => FixedPressureBC(0.0),
    :bottom => NoSlipWallBC(),
    :top => NoSlipWallBC(),
)

algo = SIMPLE(; alpha_U = 0.7, alpha_p = 0.3, max_iterations = 10, tolerance = 1.0e-7)
prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = ν, density = 1.0)

# Inlet turbulence: assume 5% intensity, length scale ≈ 0.07·Ly.
# k_in = 1.5 (U·I)², ε_in = Cμ^(3/4) k^(3/2) / ℓ
const I_turb = 0.05
const ℓ = 0.07 * Ly
const Cμ = 0.09
const k_in = 1.5 * (U_in * I_turb)^2
const ε_in = Cμ^(3 / 4) * k_in^(3 / 2) / ℓ

turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
    :k => Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(k_in),
        :right => NeumannBC(0.0),
        :bottom => NeumannBC(0.0),
        :top => NeumannBC(0.0),
    ),
    :epsilon => Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(ε_in),
        :right => NeumannBC(0.0),
        :bottom => NeumannBC(0.0),
        :top => NeumannBC(0.0),
    ),
)

ke = StandardKEpsilon()
result, turb_state = solve_simple_turbulent(prob, ke; turb_bcs = turb_bcs)

# Diagnostics — bulk velocity, peak ν_t, and a centre-line k profile.
nc = length(mesh.cell_volumes)
U = result.state.U.internal
U_bulk = sum(u -> abs(u[1]), U) / nc
νt_max = maximum(turb_state.nu_t)

println("=== k-ε turbulent channel ===")
@printf "iterations          : %d\n" result.iterations
@printf "converged           : %s\n" result.converged
@printf "U_bulk              : %.4f m/s\n" U_bulk
@printf "k_in                : %.4e\n" k_in
@printf "ε_in                : %.4e\n" ε_in
@printf "max ν_t / ν         : %.2f\n" νt_max / ν
@printf "max k (interior)    : %.4e\n" maximum(turb_state.fields[:k].internal)
@printf "max ε (interior)    : %.4e\n" maximum(turb_state.fields[:epsilon].internal)

# A well-resolved log-law channel would have ν_t/ν ≈ Re_τ·κ/2 in the
# log layer; on this tiny grid we mostly see the inlet relaxation.
# For the published-benchmark case (Moser et al. Re_τ = 180/395) see
# `validation/manifest.toml` and the corresponding V&V scripts.
#
# Manifest feature  : phase2a.standard_k_epsilon (experimental)
# V&V tests         : test/turbulence_rans.jl, test/turbulence_correctness.jl

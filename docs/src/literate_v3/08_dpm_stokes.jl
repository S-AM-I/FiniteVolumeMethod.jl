# # Tutorial 08 — Lagrangian DPM with Stokes Drag
#
# Demonstrates the v3 Lagrangian discrete-phase model by injecting a
# single particle into a uniform shear flow and advancing it under
# Stokes drag + gravity. Prints the resulting trajectory.
#
# Runtime budget: ~1 s on a laptop (4×4 mesh, 200 sub-steps).
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/08_dpm_stokes.jl
# ```
#
# What it demonstrates:
# - `ParticleTracker` construction and `inject_particles!` with a
#   `Vector{SVector}` of seed positions
# - Setting per-particle properties via `set_particle_properties!`
# - Advancing with `advance_particles!` using `StokesDrag` and gravity

using FiniteVolumeMethod
using StaticArrays
using Printf

include(joinpath(@__DIR__, "..", "..", "..", "test", "TestHelpers.jl"))

mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)

# Quiescent fluid so only gravity + drag act on the particle.
U = FiniteVolumeMethod.CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))

tracker = ParticleTracker{2, Float64}()
inject_particles!(tracker, [SVector(0.5, 0.9)])
p = tracker.particles[1]
p.cell_index = 13  # cell near (0.5, 0.9) on a 4×4 grid

# Glass bead in water: d_p = 100 μm, ρ_p = 2500 kg/m³.
const d_p = 1.0e-4
const ρ_p = 2500.0
const ρ_f = 1000.0
const μ_f = 1.0e-3

set_particle_properties!(p; diameter = d_p, density = ρ_p)

gravity = SVector(0.0, -9.81)
dt = 1.0e-4
n_steps = 200

# Record the trajectory as we advance.
trajectory = Vector{SVector{2, Float64}}()
push!(trajectory, p.position)

for step in 1:n_steps
    advance_particles!(
        tracker, U, mesh, dt;
        drag_model = StokesDrag(),
        rho_f = ρ_f, mu_f = μ_f,
        gravity = gravity,
    )
    push!(trajectory, p.position)
    p.active || break
end

# Terminal velocity for Stokes drag in a quiescent fluid:
# v_t = (ρ_p - ρ_f) · g · d_p² / (18 μ_f)
v_terminal = (ρ_p - ρ_f) * 9.81 * d_p^2 / (18 * μ_f)

println("=== Lagrangian DPM with Stokes drag ===")
@printf "particle d_p       : %.2e m\n" d_p
@printf "particle ρ_p       : %.1f kg/m³\n" ρ_p
@printf "v_terminal (analy) : %.4e m/s\n" v_terminal
@printf "steps taken        : %d\n" (length(trajectory) - 1)
@printf "start position     : (%.4f, %.4f)\n" trajectory[1][1] trajectory[1][2]
@printf "final position     : (%.4f, %.4f)\n" trajectory[end][1] trajectory[end][2]
@printf "final velocity     : (%.4e, %+.4e) m/s\n" p.velocity[1] p.velocity[2]
@printf "active at end      : %s\n" p.active

# Manifest feature  : phase11.lagrangian_dpm (experimental)
# V&V tests         : test/lagrangian_dpm.jl

# ============================================================
# 05 — 2D Ideal MHD Orszag-Tang Vortex
# Dashboard panels tested: Solution (8-variable heatmap), Monitor
# ============================================================

using FiniteVolumeMethod, StaticArrays, JSON3, HTTP

# Setup
eos = IdealGasEOS(5.0 / 3.0)
law = IdealMHDEquations{2}(eos)
mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 64, 64)

# Orszag-Tang vortex initial condition
# Primitive: [ρ, vx, vy, vz, P, Bx, By, Bz]
function ic(x, y)
    ρ = 25.0 / (36π)
    P = 5.0 / (12π)
    vx = -sin(2π * y)
    vy = sin(2π * x)
    vz = 0.0
    Bx = -sin(2π * y) / √(4π)
    By = sin(4π * x) / √(4π)
    Bz = 0.0
    return SVector(ρ, vx, vy, vz, P, Bx, By, Bz)
end

prob = HyperbolicProblem2D(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    ic;
    final_time = 0.5, cfl = 0.3,
)

# Create session + monitor callback
session = create_session_data(prob)
cb = hyperbolic_monitor(; interval = 10, session_data = session, law = law, mesh = mesh)

# Solve
_, _, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)
println("05_mhd_2d: t_final = $t_final, $(length(session.snapshots)) snapshots")

# Export
outfile = joinpath(@__DIR__, "..", "output", "05_mhd_2d.fvm-session.json")
export_session(session, outfile)
println("  → $outfile")

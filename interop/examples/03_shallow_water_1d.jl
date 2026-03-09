# ============================================================
# 03 — 1D Shallow Water Dam Break
# Dashboard panels tested: Solution (1D), variable names h/hv
# ============================================================

using FiniteVolumeMethod, StaticArrays, JSON3, HTTP

# Setup
law = ShallowWaterEquations{1}(; g = 9.81)
mesh = StructuredMesh1D(0.0, 10.0, 100)

# Dam break: h=2 on left, h=1 on right (primitive: [h, u])
ic(x) = x < 5.0 ? SVector(2.0, 0.0) : SVector(1.0, 0.0)

prob = HyperbolicProblem(
    law, mesh, HLLSolver(), NoReconstruction(),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.5, cfl = 0.4,
)

# Create session + monitor callback
session = create_session_data(prob)
cb = hyperbolic_monitor(; interval = 5, session_data = session, law = law, mesh = mesh)

# Solve
_, _, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)
println("03_shallow_water_1d: t_final = $t_final, $(length(session.snapshots)) snapshots")

# Export
outfile = joinpath(@__DIR__, "..", "output", "03_shallow_water_1d.fvm-session.json")
export_session(session, outfile)
println("  → $outfile")

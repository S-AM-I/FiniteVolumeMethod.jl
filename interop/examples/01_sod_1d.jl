# ============================================================
# 01 — 1D Sod Shock Tube (Euler)
# Dashboard panels tested: Solution (1D line chart), Monitor
# ============================================================

using FiniteVolumeMethod, StaticArrays, JSON3, HTTP

# Setup
eos = IdealGasEOS(1.4)
law = EulerEquations{1}(eos)
mesh = StructuredMesh1D(0.0, 1.0, 200)

# Sod initial condition (primitive: [ρ, v, P])
ic(x) = x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1)

prob = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.2, cfl = 0.4,
)

# Create session + monitor callback
session = create_session_data(prob)
cb = hyperbolic_monitor(; interval = 10, session_data = session, law = law, mesh = mesh)

# Solve
_, _, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)
println("01_sod_1d: t_final = $t_final, $(length(session.snapshots)) snapshots")

# Export
outfile = joinpath(@__DIR__, "..", "output", "01_sod_1d.fvm-session.json")
export_session(session, outfile)
println("  → $outfile")

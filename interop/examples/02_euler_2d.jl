# ============================================================
# 02 — 2D Euler Shock-Bubble Interaction
# Dashboard panels tested: Solution (2D heatmap), Monitor
# ============================================================

using FiniteVolumeMethod, StaticArrays, JSON3, HTTP

# Setup
eos = IdealGasEOS(1.4)
law = EulerEquations{2}(eos)
mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 64, 64)

# 2D Sod-like initial condition (primitive: [ρ, vx, vy, P])
function ic(x, y)
    if x < 0.5
        return SVector(1.0, 0.0, 0.0, 1.0)
    else
        return SVector(0.125, 0.0, 0.0, 0.1)
    end
end

prob = HyperbolicProblem2D(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
    ic;
    final_time = 0.2, cfl = 0.4,
)

# Create session + monitor callback
session = create_session_data(prob)
cb = hyperbolic_monitor(; interval = 5, session_data = session, law = law, mesh = mesh)

# Solve
_, _, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)
println("02_euler_2d: t_final = $t_final, $(length(session.snapshots)) snapshots")

# Export
outfile = joinpath(@__DIR__, "..", "output", "02_euler_2d.fvm-session.json")
export_session(session, outfile)
println("  → $outfile")

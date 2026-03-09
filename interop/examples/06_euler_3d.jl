# ============================================================
# 06 — 3D Euler Shock (z-slice selector test)
# Dashboard panels tested: Solution (3D z-slice heatmap)
# ============================================================

using FiniteVolumeMethod, StaticArrays, JSON3, HTTP

# Setup
eos = IdealGasEOS(1.4)
law = EulerEquations{3}(eos)
mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 16, 16, 16)

# 3D Sod-like shock along x-axis
# Primitive: [ρ, vx, vy, vz, P]
function ic(x, y, z)
    if x < 0.5
        return SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    else
        return SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    end
end

prob = HyperbolicProblem3D(
    law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
    TransmissiveBC(), TransmissiveBC(),
    TransmissiveBC(), TransmissiveBC(),
    TransmissiveBC(), TransmissiveBC(),
    ic;
    final_time = 0.1, cfl = 0.3,
)

# Create session + monitor callback
session = create_session_data(prob)
cb = hyperbolic_monitor(; interval = 5, session_data = session, law = law, mesh = mesh)

# Solve
_, _, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)
println("06_euler_3d: t_final = $t_final, $(length(session.snapshots)) snapshots")

# Export
outfile = joinpath(@__DIR__, "..", "output", "06_euler_3d.fvm-session.json")
export_session(session, outfile)
println("  → $outfile")

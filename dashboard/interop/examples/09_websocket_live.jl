# ============================================================
# 09 — WebSocket Live Streaming Demo
# Dashboard panels tested: WebSocket live connection
#
# Usage:
#   1. Run this script: julia --project=.. 09_websocket_live.jl
#   2. Open fvm-dashboard-v5.html in browser
#   3. Enter ws://localhost:8765 and click Connect
#   4. Watch snapshots stream in live
# ============================================================

using FiniteVolumeMethod, StaticArrays, JSON3, HTTP

# Setup
eos = IdealGasEOS(1.4)
law = EulerEquations{1}(eos)
mesh = StructuredMesh1D(0.0, 1.0, 100)

# Sod shock tube
ic(x) = x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1)

prob = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.2, cfl = 0.4,
)

# Create session
session = create_session_data(prob)

# Start WebSocket server
println("Starting WebSocket server on ws://localhost:8765 ...")
println("Open fvm-dashboard-v5.html and connect to see live streaming.")
server, push_snapshot! = serve_dashboard(; port = 8765, session_data = session)

# Wait for dashboard to connect
println("Waiting 5 seconds for dashboard connection...")
sleep(5)

# Solve with live streaming callback
t_start = time()
cb = function (U, t, step, dt)
    if step % 5 != 0
        return nothing
    end
    wall = time() - t_start
    U_interior = U[3:(ncells(mesh) + 2)]
    totals = conserved_totals(law, U_interior, mesh)
    snap = FVMSnapshot(t, step, U_interior, 0.0, totals, dt, wall)
    push!(session.snapshots, snap)
    push_snapshot!(snap)
    return nothing
end

_, _, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)
println("09_websocket_live: t_final = $t_final, $(length(session.snapshots)) snapshots streamed")

# Also export the final session to file
outfile = joinpath(@__DIR__, "..", "output", "09_websocket_live.fvm-session.json")
export_session(session, outfile)
println("  → $outfile")

# Keep server alive briefly so dashboard can finish receiving
println("Server will shut down in 5 seconds...")
sleep(5)
close(server)
println("Done.")

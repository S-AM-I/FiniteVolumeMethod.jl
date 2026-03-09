# ============================================================
# 04 — 1D Reactive Euler (2 species)
# Dashboard panels tested: Species Profiles (rho_Y_fuel, rho_Y_product)
# ============================================================

using FiniteVolumeMethod, StaticArrays, JSON3, HTTP

# Setup
eos = IdealGasEOS(1.4)
law = ReactiveEulerEquations{1}(eos, (:fuel, :product))
mesh = StructuredMesh1D(0.0, 1.0, 200)

# Initial condition: shocked fuel on left, unburned on right
# Primitive: [ρ, v, P, Y_fuel, Y_product]
function ic(x)
    if x < 0.3
        # Post-detonation: high pressure, mostly product
        return SVector(1.4, 0.0, 2.0, 0.1, 0.9)
    else
        # Unburned fuel ahead of the wave
        return SVector(1.0, 0.0, 1.0, 1.0, 0.0)
    end
end

prob = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.15, cfl = 0.3,
)

# Create session + monitor callback
session = create_session_data(prob)
cb = hyperbolic_monitor(; interval = 10, session_data = session, law = law, mesh = mesh)

# Solve
_, _, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)
println("04_reactive_euler_1d: t_final = $t_final, $(length(session.snapshots)) snapshots")

# Export
outfile = joinpath(@__DIR__, "..", "output", "04_reactive_euler_1d.fvm-session.json")
export_session(session, outfile)
println("  → $outfile")

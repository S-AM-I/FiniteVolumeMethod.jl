# ============================================================
# 08 — Parabolic Diffusion (FVMProblem)
# Dashboard panels tested: Monitor only (known limitation: no solution rendering)
# ============================================================

using FiniteVolumeMethod, StaticArrays, JSON3, HTTP
using DelaunayTriangulation, OrdinaryDiffEq

# Setup triangular mesh
tri = triangulate_rectangle(0, 1, 0, 1, 15, 15; single_boundary = true)
geom = FVMGeometry(tri)

# Boundary condition: u = 0 on boundary (Dirichlet)
bc = (x, y, t, u, p) -> zero(u)
BCs = BoundaryConditions(geom, bc, Dirichlet)

# Diffusion coefficient
D = (x, y, t, u, p) -> 1.0

# Initial condition: smooth bump
f = (x, y) -> sin(π * x) * sin(π * y)
initial_condition = [f(p[1], p[2]) for p in DelaunayTriangulation.each_point(tri)]

prob = FVMProblem(
    geom, BCs;
    diffusion_function = D,
    initial_condition = initial_condition,
    final_time = 0.1,
)
ode_prob = ODEProblem(prob)

# Create session + monitor callback
session = FVMSessionData(;
    problem_type = "FVMProblem",
    law_name = "DiffusionEquation",
    mesh_info = mesh_to_dict(geom),
    variable_names = ["u"],
    parameters = Dict{String, Any}("diffusion" => "constant D=1.0", "solver" => "Tsit5"),
)
monitor = FVMMonitorCallback(; interval = 10, session_data = session)

# Solve
sol = solve(ode_prob, Tsit5(); callback = monitor, saveat = 0.1)
println("08_parabolic_diffusion: $(length(session.snapshots)) snapshots")

# Export
outfile = joinpath(@__DIR__, "..", "output", "08_parabolic_diffusion.fvm-session.json")
export_session(session, outfile)
println("  → $outfile")
println("  Note: Dashboard shows Monitor panel only — FVMGeometry solution rendering is a known limitation.")

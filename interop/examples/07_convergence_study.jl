# ============================================================
# 07 — Convergence Study (1D Euler smooth advection)
# Dashboard panels tested: Compare panel, log-log convergence plot
# ============================================================

using FiniteVolumeMethod, StaticArrays, JSON3, HTTP

eos = IdealGasEOS(1.4)
law = EulerEquations{1}(eos)

# Smooth advection: sinusoidal density perturbation on uniform flow
# Primitive: [ρ, v, P]
ρ0 = 1.0; v0 = 1.0; P0 = 1.0
A = 0.2  # perturbation amplitude

function make_ic(x)
    ρ = ρ0 + A * sin(2π * x)
    return SVector(ρ, v0, P0)
end

# Exact solution at t = T (periodic, one full advection cycle)
T = 1.0
exact_rho(x) = ρ0 + A * sin(2π * (x - v0 * T))

resolutions = [50, 100, 200, 400]

# Use the finest-resolution session to store convergence data
convergence_session = FVMSessionData(;
    problem_type = "ConvergenceStudy",
    law_name = "EulerEquations{1}",
    mesh_info = Dict{String, Any}("type" => "StructuredMesh1D", "study" => "smooth_advection"),
    variable_names = variable_names(law),
    parameters = Dict{String, Any}(
        "cfl" => 0.4,
        "solver" => "HLLCSolver",
        "reconstruction" => "CellCenteredMUSCL",
        "resolutions" => resolutions,
    ),
)

for N in resolutions
    mesh = StructuredMesh1D(0.0, 1.0, N)

    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), make_ic;
        final_time = T, cfl = 0.4,
    )

    session = create_session_data(prob)
    cb = hyperbolic_monitor(; interval = max(1, N), session_data = session, law = law, mesh = mesh)

    x, U, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)

    # Compute error norms
    dx = 1.0 / N
    L1_err = 0.0
    L2_err = 0.0
    Linf_err = 0.0
    for i in 1:N
        xc = (i - 0.5) * dx
        err = abs(U[i][1] - exact_rho(xc))
        L1_err += err * dx
        L2_err += err^2 * dx
        Linf_err = max(Linf_err, err)
    end
    L2_err = sqrt(L2_err)

    add_convergence_point!(
        convergence_session, N,
        Dict("L1" => L1_err, "L2" => L2_err, "Linf" => Linf_err)
    )

    println("  N=$N: L1=$L1_err, L2=$L2_err, Linf=$Linf_err")

    # Also export individual session for Compare panel overlay
    outfile = joinpath(@__DIR__, "..", "output", "07_convergence_N$(N).fvm-session.json")
    export_session(session, outfile)
end

# Export convergence summary
outfile = joinpath(@__DIR__, "..", "output", "07_convergence_study.fvm-session.json")
export_session(convergence_session, outfile)
println("07_convergence_study: $(length(convergence_session.convergence_data)) data points → $outfile")

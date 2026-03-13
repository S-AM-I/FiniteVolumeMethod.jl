using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Conservation Verification
# This example verifies that the hyperbolic solver conserves mass, momentum,
# and energy to machine precision when using periodic boundary conditions.
#
# ## Mathematical Setup
# We solve the 1D Euler equations with a smooth initial condition and
# periodic BCs. The total mass $\sum \rho\,\Delta x$, momentum
# $\sum \rho v\,\Delta x$, and energy $\sum E\,\Delta x$ should remain
# constant (up to round-off) for all time.
#
# ## Inputs
# - **Resolution**: $N = 200$
# - **IC**: $\rho = 1 + 0.2\sin(2\pi x)$, $v = 0.5$, $P = 1.0$
# - **BC**: Periodic
# - **Solver**: HLLC + MUSCL(Minmod)
# - **CFL**: 0.4

using FiniteVolumeMethod
using OrdinaryDiffEq
using SciMLBase: ReturnCode, remake
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

N = 200
dx = 1.0 / N

function conservation_ic(x)
    rho = 1.0 + 0.2 * sin(2 * pi * x)
    v = 0.5
    P = 1.0
    return SVector(rho, v, P)
end

function conserved_totals(U, dx)
    mass = 0.0
    momentum = 0.0
    energy = 0.0
    for u in U
        mass += u[1] * dx
        momentum += u[2] * dx
        energy += u[3] * dx
    end
    return mass, momentum, energy
end

# ## Sequential Canonical SciML Solves
n_intervals = 20
dt_interval = 0.025
t_checkpoints = [i * dt_interval for i in 0:n_intervals]

mesh = StructuredMesh1D(0.0, 1.0, N)
prob = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), conservation_ic;
    final_time = t_checkpoints[end], cfl = 0.4
)
ode_prob = sciml_problem(prob)
dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
total_mass = Float64[]
total_momentum = Float64[]
total_energy = Float64[]

let
    mass = 0.0
    momentum = 0.0
    energy = 0.0
    for i in 1:N
        x = mesh.xmin + (i - 0.5) * dx
        u = primitive_to_conserved(law, conservation_ic(x))
        mass += u[1] * dx
        momentum += u[2] * dx
        energy += u[3] * dx
    end
    push!(total_mass, mass)
    push!(total_momentum, momentum)
    push!(total_energy, energy)
end

current_state = nothing
current_dt = dt0
let
    local_state = current_state
    local_dt = current_dt
    for interval in 1:n_intervals
        t_start = t_checkpoints[interval]
        t_end = t_checkpoints[interval + 1]

        interval_prob = if interval == 1
            remake(prob; initial_time = t_start, final_time = t_end)
        else
            previous_state = local_state
            restart_ic = let U_prev = previous_state, xmin = mesh.xmin
                function (x)
                    i = clamp(Int(floor((x - xmin) / dx)) + 1, 1, N)
                    return conserved_to_primitive(law, U_prev[i])
                end
            end
            remake(prob; initial_condition = restart_ic, initial_time = t_start, final_time = t_end)
        end

        sol = solve(interval_prob, SSPRK33(); adaptive = false, dt = local_dt)
        @test sol.retcode == ReturnCode.Success #src
        accessor = solution_accessor(interval_prob)
        local_state = get_conserved(accessor, sol, length(sol.t))

        mass, momentum, energy = conserved_totals(local_state, dx)
        push!(total_mass, mass)
        push!(total_momentum, momentum)
        push!(total_energy, energy)

        interval_sciml = sciml_problem(interval_prob)
        local_dt = compute_initial_dt(interval_sciml.p, sol.u[end])
    end
end

# ## Relative Conservation Error
mass_err = [abs(total_mass[i] - total_mass[1]) / abs(total_mass[1]) for i in eachindex(total_mass)]
momentum_err = [abs(total_momentum[i] - total_momentum[1]) / abs(total_momentum[1]) for i in eachindex(total_momentum)]
energy_err = [abs(total_energy[i] - total_energy[1]) / abs(total_energy[1]) for i in eachindex(total_energy)]

eps_floor = 1.0e-16
mass_err_plot = max.(mass_err, eps_floor)
momentum_err_plot = max.(momentum_err, eps_floor)
energy_err_plot = max.(energy_err, eps_floor)

# ## Visualisation
fig = Figure(fontsize = 24, size = (1500, 400))
titles = ["Mass", "Momentum", "Energy"]
data = [mass_err_plot, momentum_err_plot, energy_err_plot]
for (idx, (title, err_data)) in enumerate(zip(titles, data))
    local ax
    ax = Axis(
        fig[1, idx], xlabel = "t", ylabel = "Relative error",
        yscale = log10, title = title
    )
    scatterlines!(ax, t_checkpoints, err_data, color = :blue, markersize = 6, linewidth = 1.5)
    hlines!(ax, [eps(Float64)], color = :red, linestyle = :dash, linewidth = 1, label = "Machine ε")
    idx == 1 && axislegend(ax, position = :rb)
end
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "conservation_verification.png") fig #src

# ## Test Assertions
@test maximum(mass_err) < 1.0e-10 #src
@test maximum(momentum_err) < 1.0e-10 #src
@test maximum(energy_err) < 1.0e-10 #src
@assert maximum(mass_err) < 1.0e-10 #hide
@assert maximum(momentum_err) < 1.0e-10 #hide
@assert maximum(energy_err) < 1.0e-10 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "max_mass_drift" => maximum(mass_err),
            "max_momentum_drift" => maximum(momentum_err),
            "max_energy_drift" => maximum(energy_err),
        ),
        artifacts = ["conservation_verification.png"],
        notes = [
            "Canonical SciML execution path via sciml_problem(prob), remake(prob; ...), and solve(prob, SSPRK33()).",
            "This evidence entry is the hyperbolic invariant-stage conservation case.",
        ],
        summary = Dict(
            "times" => t_checkpoints,
            "mass_error" => mass_err,
            "momentum_error" => momentum_err,
            "energy_error" => energy_err,
        ),
    )
end

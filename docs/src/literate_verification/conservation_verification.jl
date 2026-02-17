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

# ## Compute Initial Conserved Totals
# We first solve for a very short time to establish the initial conserved state,
# then run in sequential intervals collecting the totals.
n_intervals = 20
dt_interval = 0.025
t_checkpoints = [i * dt_interval for i in 0:n_intervals]

total_mass = Float64[]
total_momentum = Float64[]
total_energy = Float64[]

# Compute initial totals
mesh = StructuredMesh1D(0.0, 1.0, N)

## Initial state: compute conserved quantities from IC
begin
    m = 0.0
    mom = 0.0
    en = 0.0
    for i in 1:N
        x = mesh.xmin + (i - 0.5) * dx
        w = conservation_ic(x)
        u = primitive_to_conserved(law, w)
        m += u[1] * dx
        mom += u[2] * dx
        en += u[3] * dx
    end
    push!(total_mass, m)
    push!(total_momentum, mom)
    push!(total_energy, en)
end

# ## Sequential Solve in Intervals
# We solve in intervals, using the solution from the previous interval
# as the initial condition for the next. To restart, we use a closure
# over the previous solution vector.
current_U = nothing  # will hold interior solution SVectors

for interval in 1:n_intervals
    t_start = (interval - 1) * dt_interval
    t_end = interval * dt_interval

    if interval == 1
        ## First interval: use the original IC function
        prob = HyperbolicProblem(
            law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), conservation_ic;
            initial_time = t_start, final_time = t_end, cfl = 0.4
        )
    else
        ## Subsequent intervals: create IC from previous solution
        prev_U = current_U
        restart_ic = let U_prev = prev_U, dx = mesh.dx, xmin = mesh.xmin
            function (x)
                ## Find the cell index for this x
                i = clamp(Int(floor((x - xmin) / dx)) + 1, 1, N)
                return conserved_to_primitive(law, U_prev[i])
            end
        end
        prob = HyperbolicProblem(
            law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), restart_ic;
            initial_time = t_start, final_time = t_end, cfl = 0.4
        )
    end

    x, U, _ = solve_hyperbolic(prob)
    global current_U = U

    ## Compute conserved totals
    m = 0.0
    mom = 0.0
    en = 0.0
    for i in eachindex(U)
        m += U[i][1] * dx
        mom += U[i][2] * dx
        en += U[i][3] * dx
    end
    push!(total_mass, m)
    push!(total_momentum, mom)
    push!(total_energy, en)
end

# ## Relative Conservation Error
mass_err = [abs(total_mass[i] - total_mass[1]) / abs(total_mass[1]) for i in eachindex(total_mass)]
momentum_err = [abs(total_momentum[i] - total_momentum[1]) / abs(total_momentum[1]) for i in eachindex(total_momentum)]
energy_err = [abs(total_energy[i] - total_energy[1]) / abs(total_energy[1]) for i in eachindex(total_energy)]

## Replace exact zeros with a small value for log-scale plotting
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

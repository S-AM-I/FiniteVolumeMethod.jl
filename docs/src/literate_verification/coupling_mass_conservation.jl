using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Coupled Mass and Momentum Conservation
# This example checks the invariant side of the operator-splitting coupling
# path. We evolve a smooth periodic Euler state with a cooling source that acts
# only on the energy equation. On a periodic domain, total mass and total
# momentum should remain invariant under the coupled update.
#
# ## Mathematical Setup
# The initial state is the same smooth periodic profile used in the coupling
# benchmark. The source term is `CoolingSource(T -> 0.05*T)`, so energy changes
# but mass and momentum should not.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)
velocity = 0.3
cooling = CoolingSource(T -> 0.05 * T; mu_mol = 1.0)
t_final = 0.05

smooth_coupled_ic(x) = SVector(
    1.0 + 0.1 * sin(2 * pi * x),
    velocity,
    1.0 + 0.05 * cos(2 * pi * x),
)

function conservation_case(ncells)
    mesh = StructuredMesh1D(0.0, 1.0, ncells)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), smooth_coupled_ic;
        final_time = t_final, cfl = 0.35,
    )
    dx = mesh.dx
    U0 = FiniteVolumeMethod.initialize_1d(prob)
    mass0 = sum(U0[i + 2][1] * dx for i in 1:ncells)
    momentum0 = sum(U0[i + 2][2] * dx for i in 1:ncells)

    _, U, t = solve_coupled(prob, cooling; splitting = StrangSplitting())
    mass1 = sum(U[i][1] * dx for i in 1:ncells)
    momentum1 = sum(U[i][2] * dx for i in 1:ncells)

    return (
        ncells = ncells,
        mass_error = abs(mass1 - mass0) / abs(mass0),
        momentum_error = abs(momentum1 - momentum0) / abs(momentum0),
        t = t,
    )
end

cell_counts = [50, 100, 200]
results = [conservation_case(ncells) for ncells in cell_counts]
mass_errors = [result.mass_error for result in results]
momentum_errors = [result.momentum_error for result in results]

fig = Figure(fontsize = 22, size = (850, 450))
ax = Axis(
    fig[1, 1],
    xlabel = "Cells",
    ylabel = "Relative invariant drift",
    yscale = log10,
    title = "Coupled Cooling Invariants",
)
scatterlines!(
    ax,
    cell_counts,
    max.(mass_errors, 1.0e-16);
    color = :steelblue,
    marker = :circle,
    linewidth = 2,
    markersize = 12,
    label = "mass",
)
scatterlines!(
    ax,
    cell_counts,
    max.(momentum_errors, 1.0e-16);
    color = :crimson,
    marker = :utriangle,
    linewidth = 2,
    markersize = 12,
    label = "momentum",
)
hlines!(ax, [1.0e-12], color = :black, linestyle = :dash, linewidth = 1.5, label = "1e-12")
axislegend(ax, position = :rt)
resize_to_layout!(fig)
fig

# ## Test Assertions
@test all(result -> isapprox(result.t, t_final; atol = 1.0e-12), results) #src
@test all(err -> err < 1.0e-12, mass_errors) #src
@test all(err -> err < 1.0e-12, momentum_errors) #src
@assert all(result -> isapprox(result.t, t_final; atol = 1.0e-12), results) #hide
@assert all(err -> err < 1.0e-12, mass_errors) #hide
@assert all(err -> err < 1.0e-12, momentum_errors) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "mass_errors" => mass_errors,
            "momentum_errors" => momentum_errors,
            "worst_mass_error" => maximum(mass_errors),
            "worst_momentum_error" => maximum(momentum_errors),
        ),
        artifacts = ["coupling_mass_conservation.png"],
        notes = [
            "Invariant exercises the operator-split cooling path on a periodic domain where the source acts only on energy.",
            "Mass and momentum must remain conserved to machine precision under the coupled update.",
        ],
        summary = Dict(
            "cell_counts" => cell_counts,
            "mass_errors" => mass_errors,
            "momentum_errors" => momentum_errors,
        ),
    )
end

using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Coupled Advection-Cooling Reference Benchmark
# This example benchmarks the operator-splitting coupling path on a smooth
# periodic Euler state with temperature-dependent cooling. The coupled problem
# has no closed-form solution, so we compare the pressure field against a
# trusted fine-grid reference produced by the same coupling formulation at
# substantially higher resolution.
#
# ## Mathematical Setup
# We evolve a smooth periodic state
# ```math
# \rho = 1 + 0.1\sin(2\pi x), \quad v = 0.3, \quad P = 1 + 0.05\cos(2\pi x)
# ```
# under the source term `CoolingSource(T -> 0.05*T)`. The benchmark metric is
# the L1 pressure error against a `N = 800` Strang-split reference solution.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)
velocity = 0.3

smooth_coupled_ic(x) = SVector(
    1.0 + 0.1 * sin(2 * pi * x),
    velocity,
    1.0 + 0.05 * cos(2 * pi * x),
)

cooling = CoolingSource(T -> 0.05 * T; mu_mol = 1.0)
t_final = 0.05

function run_coupled_cooling(ncells)
    mesh = StructuredMesh1D(0.0, 1.0, ncells)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), smooth_coupled_ic;
        final_time = t_final, cfl = 0.35,
    )
    x, U, t = solve_coupled(prob, cooling; splitting = StrangSplitting())
    W = [conserved_to_primitive(law, u) for u in U]
    return (; x, W, t)
end

restrict_reference(W_ref) = [(W_ref[2 * i - 1] + W_ref[2 * i]) / 2 for i in 1:(length(W_ref) ÷ 2)]

function restricted_reference(W_ref, n_ref, ncells)
    current = W_ref
    current_cells = n_ref
    while current_cells > ncells
        current = restrict_reference(current)
        current_cells ÷= 2
    end
    return current
end

reference_cells = 800
reference = run_coupled_cooling(reference_cells)
cell_counts = [100, 200, 400]

pressure_errors = Float64[]
for ncells in cell_counts
    coarse = run_coupled_cooling(ncells)
    W_ref_coarse = restricted_reference(reference.W, reference_cells, ncells)
    err = sum(abs(coarse.W[i][3] - W_ref_coarse[i][3]) for i in eachindex(coarse.W)) / ncells
    push!(pressure_errors, err)
end
pressure_rates = [log2(pressure_errors[i] / pressure_errors[i + 1]) for i in 1:(length(pressure_errors) - 1)]

fig = Figure(fontsize = 22, size = (800, 450))
ax = Axis(
    fig[1, 1],
    xlabel = "Cells",
    ylabel = L"L^1(P) \text{ vs fine coupled reference}",
    xscale = log2,
    yscale = log10,
    title = "Coupled Advection-Cooling Benchmark",
)
scatterlines!(
    ax,
    cell_counts,
    pressure_errors;
    color = :darkorange,
    marker = :diamond,
    linewidth = 2,
    markersize = 12,
)
resize_to_layout!(fig)
fig
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("coupling_cooling_reference.png"), fig)
end

# ## Test Assertions
@test reference.t ≈ t_final atol = 1.0e-12 #src
@test all(diff(pressure_errors) .< 0.0) #src
@test all(rate -> rate > 1.7, pressure_rates) #src
@test pressure_errors[end] < 1.1e-6 #src
@assert reference.t ≈ t_final atol = 1.0e-12 #hide
@assert all(diff(pressure_errors) .< 0.0) #hide
@assert all(rate -> rate > 1.7, pressure_rates) #hide
@assert pressure_errors[end] < 1.1e-6 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "pressure_errors" => pressure_errors,
            "pressure_rates" => pressure_rates,
            "finest_pressure_error" => pressure_errors[end],
        ),
        artifacts = ["coupling_cooling_reference.png"],
        notes = [
            "Benchmark uses a trusted N=800 Strang-split coupling solution as the fine-grid reference dataset.",
            "The pressure field is used because the temperature-dependent cooling source acts directly through the thermodynamic state.",
        ],
        summary = Dict(
            "cell_counts" => cell_counts,
            "pressure_errors" => pressure_errors,
            "pressure_rates" => pressure_rates,
            "reference_cells" => reference_cells,
        ),
    )
end

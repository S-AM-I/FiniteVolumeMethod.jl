```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/coupling_nullsource_identity.jl"
```

````julia
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Coupling Null-Source Identity
This example verifies the operator-splitting coupling layer against the
canonical non-coupled hyperbolic solve. When the source operator is the
exact no-op `NullSource`, Lie-Trotter splitting should reduce to the bare
hyperbolic update with no measurable change in the final conserved state.

## Mathematical Setup
We solve the 1D Euler equations for a Sod shock tube and compare:
1. the canonical SciML-backed hyperbolic solve
2. the coupling solve with `NullSource()` and `LieTrotterSplitting()`

The identity claim is implementation-level: the coupling wrapper must not
perturb the underlying finite-volume update when the source is truly zero.

````julia
using FiniteVolumeMethod
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase: ReturnCode
using StaticArrays
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

sod_ic(x) = x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1)

function nullsource_identity_case(ncells)
    mesh = StructuredMesh1D(0.0, 1.0, ncells)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), sod_ic;
        final_time = 0.1, cfl = 0.4,
    )

    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    canonical = get_conserved(solution_accessor(prob), sol, length(sol.t))

    x_split, split_state, t_split = solve_coupled(
        prob,
        NullSource();
        splitting = LieTrotterSplitting(),
    )

    max_abs_difference = maximum(
        maximum(abs.(canonical[i] - split_state[i])) for i in eachindex(canonical)
    )

    return (
        ncells = ncells,
        max_abs_difference = max_abs_difference,
        t_canonical = sol.t[end],
        t_split = t_split,
        split_density = [conserved_to_primitive(law, u)[1] for u in split_state],
        x = x_split,
    )
end

cell_counts = [50, 100, 200]
results = [nullsource_identity_case(ncells) for ncells in cell_counts]
max_differences = [result.max_abs_difference for result in results]

fig = Figure(fontsize = 22, size = (800, 450))
ax = Axis(
    fig[1, 1],
    xlabel = "Cells",
    ylabel = "max |U_canonical - U_coupled|",
    yscale = log10,
    title = "Null-Source Coupling Identity",
)
scatterlines!(
    ax,
    cell_counts,
    max.(max_differences, 1.0e-16);
    color = :teal,
    marker = :circle,
    linewidth = 2,
    markersize = 12,
)
hlines!(ax, [1.0e-12], color = :black, linestyle = :dash, linewidth = 1.5, label = "1e-12")
axislegend(ax, position = :rt)
resize_to_layout!(fig)
fig
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("coupling_nullsource_identity.png"), fig)
end
````

## Test Assertions

````julia
@assert all(result -> isapprox(result.t_canonical, result.t_split; atol = 1.0e-12), results) #hide
@assert all(diff -> diff < 1.0e-12, max_differences) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "max_abs_differences" => max_differences,
            "worst_case_difference" => maximum(max_differences),
        ),
        artifacts = ["coupling_nullsource_identity.png"],
        notes = [
            "Coupling verification compares solve_coupled(..., NullSource()) against the canonical SciML-backed hyperbolic solve.",
            "This evidence entry verifies that Lie-Trotter splitting reduces to the pure hyperbolic update when the source operator is identically zero.",
        ],
        summary = Dict(
            "cell_counts" => cell_counts,
            "max_abs_differences" => max_differences,
        ),
    )
end
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/coupling_nullsource_identity.jl).

```julia
using FiniteVolumeMethod
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase: ReturnCode
using StaticArrays
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

sod_ic(x) = x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1)

function nullsource_identity_case(ncells)
    mesh = StructuredMesh1D(0.0, 1.0, ncells)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), sod_ic;
        final_time = 0.1, cfl = 0.4,
    )

    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    canonical = get_conserved(solution_accessor(prob), sol, length(sol.t))

    x_split, split_state, t_split = solve_coupled(
        prob,
        NullSource();
        splitting = LieTrotterSplitting(),
    )

    max_abs_difference = maximum(
        maximum(abs.(canonical[i] - split_state[i])) for i in eachindex(canonical)
    )

    return (
        ncells = ncells,
        max_abs_difference = max_abs_difference,
        t_canonical = sol.t[end],
        t_split = t_split,
        split_density = [conserved_to_primitive(law, u)[1] for u in split_state],
        x = x_split,
    )
end

cell_counts = [50, 100, 200]
results = [nullsource_identity_case(ncells) for ncells in cell_counts]
max_differences = [result.max_abs_difference for result in results]

fig = Figure(fontsize = 22, size = (800, 450))
ax = Axis(
    fig[1, 1],
    xlabel = "Cells",
    ylabel = "max |U_canonical - U_coupled|",
    yscale = log10,
    title = "Null-Source Coupling Identity",
)
scatterlines!(
    ax,
    cell_counts,
    max.(max_differences, 1.0e-16);
    color = :teal,
    marker = :circle,
    linewidth = 2,
    markersize = 12,
)
hlines!(ax, [1.0e-12], color = :black, linestyle = :dash, linewidth = 1.5, label = "1e-12")
axislegend(ax, position = :rt)
resize_to_layout!(fig)
fig
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("coupling_nullsource_identity.png"), fig)
end


if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "max_abs_differences" => max_differences,
            "worst_case_difference" => maximum(max_differences),
        ),
        artifacts = ["coupling_nullsource_identity.png"],
        notes = [
            "Coupling verification compares solve_coupled(..., NullSource()) against the canonical SciML-backed hyperbolic solve.",
            "This evidence entry verifies that Lie-Trotter splitting reduces to the pure hyperbolic update when the source operator is identically zero.",
        ],
        summary = Dict(
            "cell_counts" => cell_counts,
            "max_abs_differences" => max_differences,
        ),
    )
end
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


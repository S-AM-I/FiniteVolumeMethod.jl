using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # AMR Regridding Conservation
# This example checks the conservative part of the dynamic AMR algorithm:
# prolongation, restriction, and coarse-fine flux correction during repeated
# regridding. A transported entropy pulse is evolved over a short time so the
# feature stays well away from the physical boundaries and the observed drift is
# dominated by the adaptive machinery rather than true domain outflow.
#
# ## Mathematical Setup
# We track the relative drift in the four Euler conserved quantities:
# total mass, $x$-momentum, $y$-momentum, and total energy.
#
# ## Reference
# - Berger, M.J. & Colella, P. (1989). Local Adaptive Mesh Refinement for
#   Shock Hydrodynamics. *Journal of Computational Physics*, 82, 64-84.

using Test #src
using CairoMakie

amr_common_path = joinpath(@__DIR__, "amr_common.jl")
isfile(amr_common_path) || (amr_common_path = joinpath(@__DIR__, "..", "literate_verification", "amr_common.jl"))
include(amr_common_path)

base_cells = [8, 12, 16]
results = [dynamic_conservation_case(base; max_level = 2) for base in base_cells]
max_drifts = [result.max_relative_drift for result in results]
component_drifts = [
    [result.relative_drift[index] for result in results] for index in eachindex(AMR_CONSERVED_NAMES)
]

fig = Figure(fontsize = 22, size = (850, 500))
ax = Axis(
    fig[1, 1],
    xlabel = "Base block cells per side",
    ylabel = "Maximum relative conserved drift",
    yscale = log10,
    title = "Dynamic AMR Conservation Drift",
)
scatterlines!(
    ax,
    base_cells,
    max_drifts;
    color = :crimson,
    marker = :rect,
    linewidth = 2,
    markersize = 12,
)
resize_to_layout!(fig)
fig

# ## Test Assertions
# The regridding/conservation drift should stay below 1e-3 and improve as the
# base AMR resolution is increased.
@test all(result -> isapprox(result.final_time, AMR_DYNAMIC_FINAL_TIME; atol = 1.0e-12), results) #src
@test all(result -> result.levels == 2, results) #src
@test all(diff(max_drifts) .< 0.0) #src
@test all(drift -> drift < 1.0e-3, max_drifts) #src
@assert all(result -> isapprox(result.final_time, AMR_DYNAMIC_FINAL_TIME; atol = 1.0e-12), results) #hide
@assert all(result -> result.levels == 2, results) #hide
@assert all(diff(max_drifts) .< 0.0) #hide
@assert all(drift -> drift < 1.0e-3, max_drifts) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "max_relative_drifts" => max_drifts,
            "component_drifts" => Dict(
                AMR_CONSERVED_NAMES[index] => component_drifts[index] for index in eachindex(AMR_CONSERVED_NAMES)
            ),
            "worst_drift" => maximum(max_drifts),
        ),
        artifacts = ["amr_regridding_conservation.png"],
        notes = [
            "This invariant exercises the dynamic solve_amr(prob) regridding path rather than the fixed-hierarchy canonical ODEProblem route.",
            "The short integration horizon keeps the entropy pulse far from the domain boundaries so the reported drift reflects adaptive transfer consistency.",
        ],
        summary = Dict(
            "base_cells" => base_cells,
            "max_relative_drifts" => max_drifts,
            "component_drifts" => Dict(
                AMR_CONSERVED_NAMES[index] => component_drifts[index] for index in eachindex(AMR_CONSERVED_NAMES)
            ),
        ),
    )
end

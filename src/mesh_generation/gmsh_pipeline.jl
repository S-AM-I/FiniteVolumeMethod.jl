# mesh_generation/gmsh_pipeline.jl — Gmsh orchestration stub + extension hook.
#
# Real Gmsh invocation lives in `ext/FVMGmshExt.jl` (weak-dep). This
# module provides:
#   - the user-facing `GmshPipeline{T}` parameter bundle,
#   - a runner `run_gmsh_pipeline(...)` that errors with a helpful
#     message when Gmsh.jl is not loaded,
#   - an `auto_remediate!` hook that is a no-op warning for v3.0 and
#     becomes a real cell-swap / edge-collapse pass in v3.1.

"""
    GmshPipeline

Scriptable Gmsh automation plan with quality thresholds. The `script`
may be either a path to a `.geo` file or a raw Gmsh scripting string.
"""
struct GmshPipeline
    script::String
    max_non_ortho::Float64
    max_skew::Float64
    max_aspect::Float64
end

function GmshPipeline(
        script::AbstractString;
        max_non_ortho::Real = 70.0,
        max_skew::Real = 0.85,
        max_aspect::Real = 100.0,
    )
    return GmshPipeline(
        String(script), Float64(max_non_ortho), Float64(max_skew), Float64(max_aspect),
    )
end

"""
    run_gmsh_pipeline(pipeline::GmshPipeline, out_path::AbstractString)

Invoke Gmsh via the `FVMGmshExt` extension. Errors with a helpful
message if Gmsh.jl is not loaded.
"""
function run_gmsh_pipeline(pipeline::GmshPipeline, out_path::AbstractString)
    return error(
        "Gmsh.jl required — run `using Gmsh` to activate FVMGmshExt",
    )
end

"""
    auto_remediate!(mesh, quality_report, thresholds)

Placeholder remediation pass for bad cells. v3.0 emits a warning and
leaves the mesh untouched; v3.1 will implement real swap/collapse.
"""
function auto_remediate!(mesh, quality_report, thresholds)
    @warn "auto_remediate! deferred to v3.1 — no mesh modification performed" maxlog = 1
    return mesh
end

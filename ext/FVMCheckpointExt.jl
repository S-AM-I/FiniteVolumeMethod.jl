module FVMCheckpointExt

using FiniteVolumeMethod
using JLD2

function _checkpoint_metadata(metadata)
    normalized = FiniteVolumeMethod.stringify_keys(Dict{Any, Any}(metadata))
    return merge(
        Dict{String, Any}(
            "checkpoint_format_version" => 1,
            "checkpoint_writer" => "FiniteVolumeMethod.FVMCheckpointExt",
        ),
        normalized,
    )
end

"""
    save_checkpoint(cm::CheckpointManager, state, step; metadata=Dict())

Save a simulation checkpoint using JLD2.

Creates a file in `cm.dir` named `checkpoint_<step>.jld2`. Automatically
prunes old checkpoints, keeping only the most recent `cm.keep_recent` files.

# Arguments
- `cm::CheckpointManager`: Checkpoint configuration.
- `state`: Simulation state to save (any serializable object).
- `step::Int`: Current time-step index.
- `metadata::Dict{String,Any}`: Additional metadata to store.
"""
function FiniteVolumeMethod.save_checkpoint(
        cm::FiniteVolumeMethod.CheckpointManager,
        state,
        step::Int;
        metadata::AbstractDict = Dict{String, Any}(),
    )
    mkpath(cm.dir)
    filename = joinpath(cm.dir, "checkpoint_$(step).jld2")
    checkpoint_metadata = _checkpoint_metadata(metadata)

    JLD2.jldsave(filename; state = state, step = step, metadata = checkpoint_metadata)

    # Prune old checkpoints
    existing = sort(filter(f -> endswith(f, ".jld2"), readdir(cm.dir)))
    while length(existing) > cm.keep_recent
        old_file = popfirst!(existing)
        rm(joinpath(cm.dir, old_file); force = true)
    end

    return filename
end

"""
    load_checkpoint(cm::CheckpointManager, filename_or_step)

Load a simulation checkpoint from JLD2.

If `filename_or_step` is an `Int`, loads `checkpoint_<step>.jld2` from `cm.dir`.
If it is a `String`, loads that file directly.

Returns a `Dict` with keys `"state"`, `"step"`, and `"metadata"`.
"""
function FiniteVolumeMethod.load_checkpoint(
        cm::FiniteVolumeMethod.CheckpointManager,
        filename_or_step::Union{Int, AbstractString},
    )
    filename = if filename_or_step isa Int
        joinpath(cm.dir, "checkpoint_$(filename_or_step).jld2")
    else
        filename_or_step
    end

    isfile(filename) || error("Checkpoint file not found: $filename")

    data = JLD2.load(filename)
    return Dict{String, Any}(
        "state" => data["state"],
        "step" => data["step"],
        "metadata" => FiniteVolumeMethod.stringify_keys(get(data, "metadata", Dict{String, Any}())),
    )
end

end # module

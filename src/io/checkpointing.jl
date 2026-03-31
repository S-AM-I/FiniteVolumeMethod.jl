# Checkpointing stubs — real implementations in ext/FVMHdf5Ext.jl
# Provides the CheckpointManager type and stub functions for save/load.
# Full implementations require HDF5 and are provided by the extension.

"""
    CheckpointManager

Manages scheduling and execution of simulation checkpoints.
"""
struct CheckpointManager
    interval::Int
    dir::String
    keep_recent::Int
end

CheckpointManager(; interval = 100, dir = "checkpoints", keep_recent = 3) =
    CheckpointManager(interval, dir, keep_recent)

"""Save simulation state to a checkpoint file. Requires the JLD2 extension."""
function save_checkpoint end
"""Load simulation state from a checkpoint file. Requires the JLD2 extension."""
function load_checkpoint end

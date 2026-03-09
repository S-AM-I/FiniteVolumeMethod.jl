# Output manager — migrated from Simu.jl SimuIO
# Provides scheduling, targeting, diagnostics, provenance, and output management.

using Dates

"""
    OutputSchedule{T}

Times or cadence at which to write output.
"""
struct OutputSchedule{T}
    write_times::Vector{T}     # explicit times, or
    write_every::Int           # every N steps (if > 0)
end

# Convenience constructors
OutputSchedule(start::T, stop::T, dt::T) where {T} =
    OutputSchedule{T}(collect(start:dt:stop), 0)

"""
    OutputTarget

Describes where and how to write (file, memory, database, etc.).
"""
struct OutputTarget
    kind::Symbol               # :file, :memory, :database, etc.
    destination::String        # path, URL, or identifier
    format::Symbol             # :vtk, :hdf5, :csv, :jld2, ...
end

"""
    Diagnostic

One diagnostic quantity such as max temperature or total mass.
"""
struct Diagnostic
    name::Symbol
    description::String
    compute!::Function         # (state, model, disc, t) -> value
end

"""
    SimulationConfig

Raw user configuration (e.g., parsed from TOML/JSON).
"""
struct SimulationConfig <: AbstractConfig
    options::Dict{Symbol, Any}
end

"""
    Provenance

Metadata needed for reproducibility.
"""
struct Provenance
    simulation_id::String
    timestamp::DateTime
    git_commit::Union{String, Nothing}
    code_version::String
    user::String
    host::String
end

"""
    OutputManager{T}

Config and state for outputs and diagnostics.
"""
struct OutputManager{T} <: AbstractOutputManager
    schedule::OutputSchedule{T}
    targets::Vector{OutputTarget}
    diagnostics::Vector{Diagnostic}
    last_write_time::T
    provenance::Provenance
end

"""
    validate_schedule(schedule::OutputSchedule)
"""
function validate_schedule(schedule::OutputSchedule)
    (schedule.write_every >= 0) || throw(ArgumentError("write_every must be non-negative"))
    (schedule.write_every == 0 || isempty(schedule.write_times)) || throw(ArgumentError("write_every and write_times are mutually exclusive"))
    return schedule
end

"""
    next_write_time(manager::OutputManager, step::Int)
"""
function next_write_time(manager::OutputManager, step::Int)
    validate_schedule(manager.schedule)
    if manager.schedule.write_every > 0
        return manager.last_write_time + manager.schedule.write_every, true
    else
        for t in manager.schedule.write_times
            t > manager.last_write_time && return (t, true)
        end
    end
    return (manager.last_write_time, false)
end

"""
    run_diagnostics(manager::OutputManager, state, model, disc, t)
"""
function run_diagnostics(manager::OutputManager, state, model, disc, t)
    results = Dict{Symbol, Any}()
    for diag in manager.diagnostics
        results[diag.name] = diag.compute!(state, model, disc, t)
    end
    return results
end

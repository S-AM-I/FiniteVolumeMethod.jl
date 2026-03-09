# orchestration.jl - Simulation orchestration and control logic
# Migrated from Simu.jl SimuEngine

"""
    Simulation{Prob,Alg,Sol,CB,Diag}

Container holding all components of a numerical simulation.
"""
struct Simulation{Prob, Alg, Sol, CB, Diag}
    problem::Prob
    algorithm::Alg
    solution::Sol
    callbacks::CB
    diagnostics::Diag
end

"""
    TimeGrid

Explicit list of times (e.g. for output or events).
"""
struct TimeGrid{T} <: AbstractTimeGrid
    times::Vector{T}
end

"""
    TimeStepHistory

History of chosen time steps for adaptivity and analysis.
"""
struct TimeStepHistory{T}
    t::Vector{T}
    dt::Vector{T}
    accepted::BitVector
end

"""
    TimeStepHistory(t)

Initialize history with a starting time vector.
"""
TimeStepHistory(t::Vector{T}) where {T} = TimeStepHistory{T}(t, Vector{T}(), BitVector())

"""
    TimeController{T}

Holds logic for time step selection and stopping criteria.
"""
mutable struct TimeController{T} <: AbstractController
    t::T              # current time (could mirror state.t)
    dt::T             # current time step
    dt_min::T
    dt_max::T
    t_end::T
    max_steps::Int
    adaptivity::Bool
    safety::T         # Safety factor (default 0.9)
    order::Int        # Method order (default 2)
    time_grid::AbstractTimeGrid
    history::TimeStepHistory{T}
end

"""
    TimeController(t_start, dt, t_end; dt_min, dt_max, max_steps, adaptivity, safety, order, time_grid)

Constructor for TimeController.
"""
function TimeController(
        t_start::T, dt::T, t_end::T;
        dt_min = 1.0e-10, dt_max = Inf,
        max_steps = 10000, adaptivity = true,
        safety = 0.9, order = 2,
        time_grid = TimeGrid(T[])
    ) where {T}
    history = TimeStepHistory(Vector{T}())
    return TimeController{T}(t_start, dt, dt_min, dt_max, t_end, max_steps, adaptivity, safety, order, time_grid, history)
end

"""
    EventType

Enum for categorizing simulation events.
"""
@enum EventType begin
    ROOT_EVENT        # zero-crossing of some monitor
    TIME_EVENT        # fire at specific time(s)
end

"""
    Event <: AbstractEvent

Discrete event specification with trigger condition and action.
"""
struct Event <: AbstractEvent
    name::Symbol
    etype::EventType
    condition!::Function   # (state, t) -> value for ROOT_EVENT
    action!::Function      # (sim) -> maybe-updated sim
end

"""
    ControllerWithEvents{T}

A controller that manages both time-stepping and discrete events.
"""
struct ControllerWithEvents{T} <: AbstractController
    time::TimeController{T}
    events::Vector{Event}
end

"""
    propose_step(ctrl::TimeController, error_estimate)

Propose the next time step based on adaptivity logic and current error.
"""
function propose_step(ctrl::TimeController, error_estimate)
    if !ctrl.adaptivity
        return ctrl.dt
    end

    # error_estimate is typically the normalized error (should be < 1.0)
    # Formula: h_new = h_old * safety * (1 / error)^(1/order)

    # Avoid zero division
    err = max(error_estimate, 1.0e-16)

    # Calculate factor
    exponent = 1.0 / ctrl.order
    factor = ctrl.safety * (1.0 / err)^exponent

    # Limit growth/shrink to avoid instability
    factor = clamp(factor, 0.5, 2.0)

    proposed = ctrl.dt * factor
    return clamp(proposed, ctrl.dt_min, ctrl.dt_max)
end

"""
    accept_step!(ctrl::TimeController, dt_new)

Update controller state after a successful time step.
"""
function accept_step!(ctrl::TimeController, dt_new)
    push!(ctrl.history.t, ctrl.t)
    push!(ctrl.history.dt, dt_new)
    push!(ctrl.history.accepted, true)
    ctrl.t += dt_new
    ctrl.dt = dt_new
    return ctrl
end

"""
    fire_events!(ctrl::ControllerWithEvents, sim)

Evaluate all event conditions and execute actions for triggered events.
"""
function fire_events!(ctrl::ControllerWithEvents, sim)
    current = sim
    for ev in ctrl.events
        if ev.etype == TIME_EVENT && any(isapprox(ctrl.time.t, t; atol = 1.0e-8) for t in ev.condition!(current.state, ctrl.time.t))
            updated = ev.action!(current)
            current = updated === nothing ? current : updated
        elseif ev.etype == ROOT_EVENT && abs(ev.condition!(current.state, ctrl.time.t)) < 1.0e-8
            updated = ev.action!(current)
            current = updated === nothing ? current : updated
        end
    end
    return current
end

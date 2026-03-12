# ============================================================
# Dashboard Callbacks for Parabolic (Vertex-Centered) Solvers
# ============================================================
#
# Wraps SciMLBase.DiscreteCallback to record FVMSnapshots during
# ODE-based solves (FVMProblem / FVMSystem).

"""
    FVMMonitorCallback(; interval=1, session_data)

Create a `DiscreteCallback` that records snapshots of the parabolic solver state
into `session_data` every `interval` accepted time steps.

# Keyword Arguments
- `interval::Int`: Record every `interval`-th step (default: every step).
- `session_data::FVMSessionData`: Session to append snapshots to.

# Returns
A `DiscreteCallback` compatible with DifferentialEquations.jl solvers.

# Example
```julia
session = FVMSessionData(problem_type="FVMProblem", ...)
cb = FVMMonitorCallback(; interval=10, session_data=session)
sol = solve(prob, Tsit5(); callback=cb)
```
"""
function FVMMonitorCallback(;
        interval::Int = 1,
        session_data::FVMSessionData,
    )
    t_start = time()
    step_counter = Ref(0)

    condition = function (u, t, integrator)
        step_counter[] += 1
        return step_counter[] % interval == 0
    end

    affect! = function (integrator)
        wall = time() - t_start
        u = integrator.u
        t_val = integrator.t
        dt_val = integrator.dt
        totals = Dict{String, Float64}("total" => sum(u))
        snap = FVMSnapshot(
            Float64(t_val), step_counter[], copy(u), 0.0, totals, Float64(dt_val), wall,
        )
        push!(session_data.snapshots, snap)
        return nothing
    end

    return DiscreteCallback(condition, affect!; save_positions = (false, false))
end

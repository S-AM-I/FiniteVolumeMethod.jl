# estimation.jl - Parameter Estimation and Calibration
# Migrated from Simu.jl SimuEngine/estimation.jl
# LinearAlgebra is already imported by the parent module.
# NOTE: Optim.jl dependency removed. calibrate_model throws an informative error.

"""
    InverseProblem{F, G, D}

Definition of an inverse problem for parameter estimation.

# Fields
- `cost_func`: Function `cost_func(params) -> scalar` returning the objective value.
- `grad_func!`: Function `grad_func!(G, params)` computing the gradient in-place.
- `initial_params`: Initial parameter guess.
- `data`: Observed data (passed through to cost/gradient functions as needed).
"""
struct InverseProblem{F, G, D}
    cost_func::F
    grad_func!::G
    initial_params::Vector{Float64}
    data::D
end

"""
    calibrate_model(prob::InverseProblem; kwargs...)

Solve the inverse problem to find optimal parameters.

!!! note
    This function requires the Optim.jl package which is not a dependency of
    FiniteVolumeMethod.jl. Load Optim.jl and implement your own optimization loop,
    or use another optimization package.

# Example
```julia
using Optim
f(p) = prob.cost_func(p)
g!(G, p) = prob.grad_func!(G, p)
res = optimize(f, g!, prob.initial_params, BFGS(), Optim.Options(g_tol=1e-6))
p_opt = Optim.minimizer(res)
```
"""
function calibrate_model(prob::InverseProblem; kwargs...)
    error(
        "calibrate_model requires Optim.jl which is not a dependency of FiniteVolumeMethod.jl. " *
            "Please load Optim.jl and run optimization manually:\n" *
            "  using Optim\n" *
            "  f(p) = prob.cost_func(p)\n" *
            "  g!(G, p) = prob.grad_func!(G, p)\n" *
            "  res = optimize(f, g!, prob.initial_params, BFGS(), Optim.Options(g_tol=1e-6))\n" *
            "  p_opt = Optim.minimizer(res)"
    )
end

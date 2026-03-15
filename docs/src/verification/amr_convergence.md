```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/amr_convergence.jl"
```

````@example amr_convergence
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# AMR Smooth-Pulse Convergence
This example verifies the 2D AMR semidiscrete solve path on a fixed hierarchy.
A compact smooth entropy pulse is transported across the domain by a uniform
background velocity and compared against the exact solution after a short time.
The hierarchy is constructed once from the initial condition and then frozen so
the measured error reflects the AMR finite-volume update itself rather than
dynamic regridding effects.

## Mathematical Setup
We solve the 2D Euler equations on $[0,1]^2$ with
```math
\rho(x,y,t) = 1 + A\exp\left(-\frac{(x-x_c-v_x t)^2 + (y-y_c-v_y t)^2}{2\sigma^2}\right),
\quad (v_x, v_y) = (0.25, 0.15), \quad P = 1,
```
where $A = 0.35$, $\sigma = 0.08$, and $(x_c, y_c) = (0.35, 0.40)$.

## Reference
- Berger, M.J. & Colella, P. (1989). Local Adaptive Mesh Refinement for
  Shock Hydrodynamics. *Journal of Computational Physics*, 82, 64-84.

````@example amr_convergence
using CairoMakie

amr_common_path = joinpath(@__DIR__, "amr_common.jl")
isfile(amr_common_path) || (amr_common_path = joinpath(@__DIR__, "..", "literate_verification", "amr_common.jl"))
include(amr_common_path)

base_cells = [8, 16, 32]
results = [fixed_hierarchy_exact_error(base; max_level = 1) for base in base_cells]
errors = [result.density_error for result in results]
rates = [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]
active_cells = [result.active_cells for result in results]

fig = Figure(fontsize = 24, size = (750, 500))
ax = Axis(
    fig[1, 1],
    xlabel = "Base block cells per side",
    ylabel = L"L^1(\rho) \text{ error}",
    xscale = log2,
    yscale = log10,
    title = "Fixed-Hierarchy AMR Convergence",
)
scatterlines!(
    ax,
    base_cells,
    errors;
    color = :steelblue,
    marker = :circle,
    linewidth = 2,
    markersize = 12,
)
resize_to_layout!(fig)
fig
````

## Test Assertions
The fixed-hierarchy AMR solve should complete successfully, the error should
decrease monotonically, and the observed rates should be comfortably positive.

````@example amr_convergence
@assert all(result -> result.retcode == ReturnCode.Success, results) #hide
@assert all(result -> isapprox(result.final_time, AMR_VERIFICATION_FINAL_TIME; atol = 1.0e-12), results) #hide
@assert all(diff(errors) .< 0.0) #hide
@assert all(rate -> rate > 0.65, rates) #hide
@assert errors[end] < 8.0e-4 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "density_errors" => errors,
            "density_rates" => rates,
            "finest_density_error" => errors[end],
            "active_cells" => active_cells,
        ),
        artifacts = ["amr_convergence.png"],
        notes = [
            "Canonical fixed-hierarchy AMR execution via sciml_problem(prob) and solve(prob, SSPRK33()).",
            "Regridding is disabled after the initial hierarchy construction so the measured error isolates the semidiscrete AMR update.",
        ],
        summary = Dict(
            "base_cells" => base_cells,
            "active_cells" => active_cells,
            "density_errors" => errors,
            "density_rates" => rates,
        ),
    )
end
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/amr_convergence.jl).

```julia
using CairoMakie

amr_common_path = joinpath(@__DIR__, "amr_common.jl")
isfile(amr_common_path) || (amr_common_path = joinpath(@__DIR__, "..", "literate_verification", "amr_common.jl"))
include(amr_common_path)

base_cells = [8, 16, 32]
results = [fixed_hierarchy_exact_error(base; max_level = 1) for base in base_cells]
errors = [result.density_error for result in results]
rates = [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]
active_cells = [result.active_cells for result in results]

fig = Figure(fontsize = 24, size = (750, 500))
ax = Axis(
    fig[1, 1],
    xlabel = "Base block cells per side",
    ylabel = L"L^1(\rho) \text{ error}",
    xscale = log2,
    yscale = log10,
    title = "Fixed-Hierarchy AMR Convergence",
)
scatterlines!(
    ax,
    base_cells,
    errors;
    color = :steelblue,
    marker = :circle,
    linewidth = 2,
    markersize = 12,
)
resize_to_layout!(fig)
fig


if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "density_errors" => errors,
            "density_rates" => rates,
            "finest_density_error" => errors[end],
            "active_cells" => active_cells,
        ),
        artifacts = ["amr_convergence.png"],
        notes = [
            "Canonical fixed-hierarchy AMR execution via sciml_problem(prob) and solve(prob, SSPRK33()).",
            "Regridding is disabled after the initial hierarchy construction so the measured error isolates the semidiscrete AMR update.",
        ],
        summary = Dict(
            "base_cells" => base_cells,
            "active_cells" => active_cells,
            "density_errors" => errors,
            "density_rates" => rates,
        ),
    )
end
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


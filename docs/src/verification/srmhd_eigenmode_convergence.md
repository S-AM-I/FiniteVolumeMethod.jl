```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/srmhd_eigenmode_convergence.jl"
```

````julia
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# SRMHD Linear Eigenmode Convergence
This example verifies the SRMHD solver's ability to propagate all four
MHD wave families: fast magnetosonic, slow magnetosonic, Alfvén, and
entropy waves. Each eigenmode is initialised as a small perturbation
and evolved; self-convergence (comparing N vs 2N solutions) demonstrates
that the numerical error decreases under mesh refinement.

## Mathematical Setup
The 1D SRMHD equations linearised about a uniform background state
admit four wave families propagating at their characteristic speeds.
We test each by perturbing the density/velocity/magnetic field along
the corresponding eigenvector direction.

## Reference
- Anile, A.M. (1989). Relativistic Fluids and Magneto-Fluids.
  Cambridge University Press.
- Balsara, D.S. (2001). Total Variation Diminishing Scheme for
  Adiabatic and Isothermal MHD. J. Comput. Phys., 174, 614-648.

````julia
using FiniteVolumeMethod
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays
using CairoMakie

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = SRMHDEquations{1}(eos)
````

## Background State
Uniform background with moderate magnetic field.

````julia
rho0 = 1.0
P0 = 1.0
Bx0 = 1.0
amp = 1.0e-4  # small perturbation amplitude
````

## Eigenmode Definitions
Each mode perturbs different variables to excite the corresponding wave family.

````julia
eigenmodes = [
    (
        name = "Fast magnetosonic",
        ic = (x -> SVector(rho0 + amp * sin(2 * pi * x), 0.0, 0.0, 0.0, P0 + amp * gamma * P0 / rho0 * sin(2 * pi * x), Bx0, 0.0, 0.0)),
        var_idx = 1,  # measure density
        var_name = "ρ",
    ),
    (
        name = "Slow magnetosonic",
        ic = (x -> SVector(rho0, 0.0, 0.0, 0.0, P0, Bx0, amp * sin(2 * pi * x), 0.0)),
        var_idx = 7,  # measure By
        var_name = "By",
    ),
    (
        name = "Alfvén",
        ic = (x -> SVector(rho0, 0.0, 0.0, amp * sin(2 * pi * x), P0, Bx0, 0.0, amp * sin(2 * pi * x))),
        var_idx = 8,  # measure Bz
        var_name = "Bz",
    ),
    (
        name = "Entropy",
        ic = (x -> SVector(rho0 + amp * sin(2 * pi * x), 0.0, 0.0, 0.0, P0, Bx0, 0.0, 0.0)),
        var_idx = 1,  # measure density
        var_name = "ρ",
    ),
]
````

## Self-Convergence Study
We use self-convergence: run at resolutions N and 2N, then compare the
solutions on the coarse grid. The difference decreases under refinement.

````julia
t_final = 0.5
resolutions = [32, 64, 128]

all_errors = Dict{String, Vector{Float64}}()
all_rates = Dict{String, Vector{Float64}}()

function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

function run_mode(law, mode, N, t_final)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), mode.ic;
        final_time = t_final, cfl = 0.3,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    accessor = solution_accessor(prob)
    x = get_coordinates(accessor)
    U = get_conserved(accessor, sol, length(sol.t))
    t_end = sol.t[end]
    w = [conserved_to_primitive(law, U[i])[mode.var_idx] for i in eachindex(x)]
    return x, w, t_end
end

for mode in eigenmodes
    errors = Float64[]
    for N in resolutions
        # Run at N and 2N
        x_c, w_c, _ = run_mode(law, mode, N, t_final)
        _, w_f, _ = run_mode(law, mode, 2 * N, t_final)
        # Restrict fine solution to coarse grid (average pairs)
        w_f_coarse = [(w_f[2 * i - 1] + w_f[2 * i]) / 2 for i in 1:N]
        # L1 difference
        err = sum(abs(w_c[i] - w_f_coarse[i]) for i in 1:N) / N
        push!(errors, err)
    end
    all_errors[mode.name] = errors
    all_rates[mode.name] = convergence_rates(errors)
end
````

## Visualisation — Convergence Comparison

````julia
fig = Figure(fontsize = 22, size = (800, 600))
ax = Axis(
    fig[1, 1], xlabel = "N", ylabel = L"L^1 \text{ self-convergence error}",
    xscale = log2, yscale = log10,
    title = "SRMHD Eigenmode Self-Convergence",
)
colors = [:blue, :red, :green, :orange]
markers = [:circle, :utriangle, :diamond, :rect]
for (i, mode) in enumerate(eigenmodes)
    scatterlines!(
        ax, resolutions, all_errors[mode.name],
        color = colors[i], marker = markers[i],
        linewidth = 2, markersize = 10,
        label = mode.name,
    )
end
# Reference slope
e_ref = all_errors[eigenmodes[1].name][1]
N_ref = resolutions[1]
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 1,
    color = :black, linestyle = :dash, linewidth = 1, label = L"O(N^{-1})",
)
axislegend(ax, position = :lb)
resize_to_layout!(fig)
fig
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("srmhd_eigenmode_convergence.png"), fig)
end
````

## Test Assertions
Each eigenmode should show self-convergence at rate > 0.5 with MUSCL+HLL.

````julia
for mode in eigenmodes
    rates = all_rates[mode.name]
end
@assert all(all(r -> r > 0.5, all_rates[m.name]) for m in eigenmodes) #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "mode_errors" => all_errors,
            "mode_rates" => all_rates,
            "minimum_mode_rate" => minimum(vcat(values(all_rates)...)),
        ),
        artifacts = ["srmhd_eigenmode_convergence.png"],
        notes = [
            "Relativistic benchmark sweep across fast, slow, Alfven, and entropy SRMHD eigenmodes.",
            "This evidence entry is the relativistic benchmark-stage eigenmode propagation case.",
        ],
        summary = Dict(
            "resolutions" => resolutions,
            "mode_errors" => all_errors,
            "mode_rates" => all_rates,
            "mode_names" => [mode.name for mode in eigenmodes],
        ),
    )
end
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_verification/srmhd_eigenmode_convergence.jl).

```julia
using FiniteVolumeMethod
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays
using CairoMakie

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = SRMHDEquations{1}(eos)

rho0 = 1.0
P0 = 1.0
Bx0 = 1.0
amp = 1.0e-4  # small perturbation amplitude

eigenmodes = [
    (
        name = "Fast magnetosonic",
        ic = (x -> SVector(rho0 + amp * sin(2 * pi * x), 0.0, 0.0, 0.0, P0 + amp * gamma * P0 / rho0 * sin(2 * pi * x), Bx0, 0.0, 0.0)),
        var_idx = 1,  # measure density
        var_name = "ρ",
    ),
    (
        name = "Slow magnetosonic",
        ic = (x -> SVector(rho0, 0.0, 0.0, 0.0, P0, Bx0, amp * sin(2 * pi * x), 0.0)),
        var_idx = 7,  # measure By
        var_name = "By",
    ),
    (
        name = "Alfvén",
        ic = (x -> SVector(rho0, 0.0, 0.0, amp * sin(2 * pi * x), P0, Bx0, 0.0, amp * sin(2 * pi * x))),
        var_idx = 8,  # measure Bz
        var_name = "Bz",
    ),
    (
        name = "Entropy",
        ic = (x -> SVector(rho0 + amp * sin(2 * pi * x), 0.0, 0.0, 0.0, P0, Bx0, 0.0, 0.0)),
        var_idx = 1,  # measure density
        var_name = "ρ",
    ),
]

t_final = 0.5
resolutions = [32, 64, 128]

all_errors = Dict{String, Vector{Float64}}()
all_rates = Dict{String, Vector{Float64}}()

function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

function run_mode(law, mode, N, t_final)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), mode.ic;
        final_time = t_final, cfl = 0.3,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    accessor = solution_accessor(prob)
    x = get_coordinates(accessor)
    U = get_conserved(accessor, sol, length(sol.t))
    t_end = sol.t[end]
    w = [conserved_to_primitive(law, U[i])[mode.var_idx] for i in eachindex(x)]
    return x, w, t_end
end

for mode in eigenmodes
    errors = Float64[]
    for N in resolutions
        # Run at N and 2N
        x_c, w_c, _ = run_mode(law, mode, N, t_final)
        _, w_f, _ = run_mode(law, mode, 2 * N, t_final)
        # Restrict fine solution to coarse grid (average pairs)
        w_f_coarse = [(w_f[2 * i - 1] + w_f[2 * i]) / 2 for i in 1:N]
        # L1 difference
        err = sum(abs(w_c[i] - w_f_coarse[i]) for i in 1:N) / N
        push!(errors, err)
    end
    all_errors[mode.name] = errors
    all_rates[mode.name] = convergence_rates(errors)
end

fig = Figure(fontsize = 22, size = (800, 600))
ax = Axis(
    fig[1, 1], xlabel = "N", ylabel = L"L^1 \text{ self-convergence error}",
    xscale = log2, yscale = log10,
    title = "SRMHD Eigenmode Self-Convergence",
)
colors = [:blue, :red, :green, :orange]
markers = [:circle, :utriangle, :diamond, :rect]
for (i, mode) in enumerate(eigenmodes)
    scatterlines!(
        ax, resolutions, all_errors[mode.name],
        color = colors[i], marker = markers[i],
        linewidth = 2, markersize = 10,
        label = mode.name,
    )
end
# Reference slope
e_ref = all_errors[eigenmodes[1].name][1]
N_ref = resolutions[1]
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 1,
    color = :black, linestyle = :dash, linewidth = 1, label = L"O(N^{-1})",
)
axislegend(ax, position = :lb)
resize_to_layout!(fig)
fig
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("srmhd_eigenmode_convergence.png"), fig)
end

for mode in eigenmodes
    rates = all_rates[mode.name]
end

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "mode_errors" => all_errors,
            "mode_rates" => all_rates,
            "minimum_mode_rate" => minimum(vcat(values(all_rates)...)),
        ),
        artifacts = ["srmhd_eigenmode_convergence.png"],
        notes = [
            "Relativistic benchmark sweep across fast, slow, Alfven, and entropy SRMHD eigenmodes.",
            "This evidence entry is the relativistic benchmark-stage eigenmode propagation case.",
        ],
        summary = Dict(
            "resolutions" => resolutions,
            "mode_errors" => all_errors,
            "mode_rates" => all_rates,
            "mode_names" => [mode.name for mode in eigenmodes],
        ),
    )
end
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


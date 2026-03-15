module RepoPerformanceBaselines

using BenchmarkTools
using DelaunayTriangulation
using FiniteVolumeMethod
using OrdinaryDiffEq
using SciMLBase: ReturnCode
using StaticArrays
using TOML

const DEFAULT_BASELINE_PATH = joinpath(@__DIR__, "performance_baselines.toml")
const DEFAULT_TIME_WARN_RATIO = 2.0
const DEFAULT_TIME_FAIL_RATIO = 4.0
const DEFAULT_MEMORY_WARN_RATIO = 1.15
const DEFAULT_MEMORY_FAIL_RATIO = 1.5
const DEFAULT_ALLOC_WARN_RATIO = 1.05
const DEFAULT_ALLOC_FAIL_RATIO = 1.25
const DEFAULT_SAMPLES = 5
const DEFAULT_SECONDS = 0.5
const DEFAULT_EVALS = 1

struct PerformanceScenario
    id::String
    feature::Symbol
    description::String
    time_warn_ratio::Float64
    time_fail_ratio::Float64
    memory_warn_ratio::Float64
    memory_fail_ratio::Float64
    alloc_warn_ratio::Float64
    alloc_fail_ratio::Float64
    release_blocker::Bool
    runner::Function
end

default_baseline_path() = DEFAULT_BASELINE_PATH

function benchmark_parameters()
    samples = tryparse(Int, get(ENV, "FVM_PERFORMANCE_SAMPLES", string(DEFAULT_SAMPLES)))
    seconds = tryparse(Float64, get(ENV, "FVM_PERFORMANCE_SECONDS", string(DEFAULT_SECONDS)))
    evals = tryparse(Int, get(ENV, "FVM_PERFORMANCE_EVALS", string(DEFAULT_EVALS)))
    return (
        samples = something(samples, DEFAULT_SAMPLES),
        seconds = something(seconds, DEFAULT_SECONDS),
        evals = something(evals, DEFAULT_EVALS),
    )
end

function stable_performance_scenarios()
    return [
        PerformanceScenario(
            "hyperbolic_euler_1d",
            :hyperbolic,
            "Canonical 1D Euler Sod solve through the SciML contract.",
            DEFAULT_TIME_WARN_RATIO,
            DEFAULT_TIME_FAIL_RATIO,
            DEFAULT_MEMORY_WARN_RATIO,
            DEFAULT_MEMORY_FAIL_RATIO,
            DEFAULT_ALLOC_WARN_RATIO,
            DEFAULT_ALLOC_FAIL_RATIO,
            true,
            _hyperbolic_runner(),
        ),
        PerformanceScenario(
            "parabolic_diffusion_2d",
            :parabolic,
            "Node-based diffusion solve for the parabolic stable family.",
            DEFAULT_TIME_WARN_RATIO,
            DEFAULT_TIME_FAIL_RATIO,
            DEFAULT_MEMORY_WARN_RATIO,
            DEFAULT_MEMORY_FAIL_RATIO,
            DEFAULT_ALLOC_WARN_RATIO,
            DEFAULT_ALLOC_FAIL_RATIO,
            true,
            _parabolic_runner(),
        ),
        PerformanceScenario(
            "mhd_ct_uniform_2d",
            :mhd_ct,
            "2D CT-MHD uniform-field solve through the canonical SSPRK33 path.",
            DEFAULT_TIME_WARN_RATIO,
            DEFAULT_TIME_FAIL_RATIO,
            DEFAULT_MEMORY_WARN_RATIO,
            DEFAULT_MEMORY_FAIL_RATIO,
            DEFAULT_ALLOC_WARN_RATIO,
            DEFAULT_ALLOC_FAIL_RATIO,
            true,
            _mhd_runner(),
        ),
        PerformanceScenario(
            "relativistic_srmhd_1d",
            :relativistic,
            "1D SRMHD smooth-wave solve through the canonical SSPRK33 path.",
            DEFAULT_TIME_WARN_RATIO,
            DEFAULT_TIME_FAIL_RATIO,
            DEFAULT_MEMORY_WARN_RATIO,
            DEFAULT_MEMORY_FAIL_RATIO,
            DEFAULT_ALLOC_WARN_RATIO,
            DEFAULT_ALLOC_FAIL_RATIO,
            true,
            _relativistic_runner(),
        ),
    ]
end

function run_suite(; parameters = benchmark_parameters())
    return [measure_scenario(scenario; parameters) for scenario in stable_performance_scenarios()]
end

function measure_scenario(scenario::PerformanceScenario; parameters = benchmark_parameters())
    # Warm up once so the benchmark measures steady-state execution rather than compilation.
    runner = scenario.runner
    warmup_result = runner()
    getproperty(warmup_result, :retcode) == ReturnCode.Success ||
        throw(ArgumentError("Performance scenario `$(scenario.id)` failed warmup with retcode `$(warmup_result.retcode)`."))

    GC.gc()
    samples = parameters.samples
    seconds = parameters.seconds
    evals = parameters.evals
    trial = @benchmark $runner() samples = samples seconds = seconds evals = evals
    estimate = BenchmarkTools.median(trial)
    return (
        id = scenario.id,
        feature = scenario.feature,
        description = scenario.description,
        time_ns = Int(estimate.time),
        memory_bytes = Int(estimate.memory),
        allocs = Int(estimate.allocs),
        samples = parameters.samples,
        seconds = parameters.seconds,
        evals = parameters.evals,
        time_warn_ratio = scenario.time_warn_ratio,
        time_fail_ratio = scenario.time_fail_ratio,
        memory_warn_ratio = scenario.memory_warn_ratio,
        memory_fail_ratio = scenario.memory_fail_ratio,
        alloc_warn_ratio = scenario.alloc_warn_ratio,
        alloc_fail_ratio = scenario.alloc_fail_ratio,
        release_blocker = scenario.release_blocker,
    )
end

function load_baselines(path::AbstractString = DEFAULT_BASELINE_PATH)
    raw = TOML.parsefile(path)
    rows = get(raw, "cases", Any[])
    return (
        baseline_version = get(raw, "baseline_version", 1),
        benchmark_samples = get(raw, "benchmark_samples", DEFAULT_SAMPLES),
        benchmark_seconds = get(raw, "benchmark_seconds", DEFAULT_SECONDS),
        benchmark_evals = get(raw, "benchmark_evals", DEFAULT_EVALS),
        cases = Dict(
            entry["id"] => (
                    id = entry["id"],
                    feature = Symbol(entry["feature"]),
                    description = entry["description"],
                    time_ns = Int(entry["time_ns"]),
                    memory_bytes = Int(entry["memory_bytes"]),
                    allocs = Int(entry["allocs"]),
                    time_warn_ratio = _baseline_ratio(entry, "time_warn_ratio", "time_ratio_limit", DEFAULT_TIME_WARN_RATIO),
                    time_fail_ratio = _baseline_ratio(entry, "time_fail_ratio", "time_ratio_limit", DEFAULT_TIME_FAIL_RATIO),
                    memory_warn_ratio = _baseline_ratio(entry, "memory_warn_ratio", "memory_ratio_limit", DEFAULT_MEMORY_WARN_RATIO),
                    memory_fail_ratio = _baseline_ratio(entry, "memory_fail_ratio", "memory_ratio_limit", DEFAULT_MEMORY_FAIL_RATIO),
                    alloc_warn_ratio = _baseline_ratio(entry, "alloc_warn_ratio", "alloc_ratio_limit", DEFAULT_ALLOC_WARN_RATIO),
                    alloc_fail_ratio = _baseline_ratio(entry, "alloc_fail_ratio", "alloc_ratio_limit", DEFAULT_ALLOC_FAIL_RATIO),
                    release_blocker = get(entry, "release_blocker", true),
                ) for entry in rows
        ),
    )
end

function compare_to_baselines(measurements, baselines)
    comparisons = NamedTuple[]
    baseline_ids = Set(keys(baselines.cases))
    measurement_ids = Set(entry.id for entry in measurements)

    for missing_id in sort!(collect(setdiff(baseline_ids, measurement_ids)); by = identity)
        push!(
            comparisons,
            (
                id = missing_id,
                feature = baselines.cases[missing_id].feature,
                passed = false,
                status = :fail,
                warning_only = false,
                reasons = ["missing current measurement for `$missing_id`"],
                warnings = String[],
                measurement = nothing,
                baseline = baselines.cases[missing_id],
                release_blocker = baselines.cases[missing_id].release_blocker,
            ),
        )
    end

    for measurement in measurements
        if !haskey(baselines.cases, measurement.id)
            push!(
                comparisons,
                (
                    id = measurement.id,
                    feature = measurement.feature,
                    passed = false,
                    status = :fail,
                    warning_only = false,
                    reasons = ["missing baseline entry for `$(measurement.id)`"],
                    warnings = String[],
                    measurement,
                    baseline = nothing,
                    release_blocker = true,
                ),
            )
            continue
        end

        baseline = baselines.cases[measurement.id]
        reasons = String[]
        warnings = String[]
        time_ratio = measurement.time_ns / baseline.time_ns
        memory_ratio = baseline.memory_bytes == 0 ? 1.0 : measurement.memory_bytes / baseline.memory_bytes
        alloc_ratio = baseline.allocs == 0 ? 1.0 : measurement.allocs / baseline.allocs

        _check_ratio!(
            warnings,
            reasons,
            "median runtime ratio",
            time_ratio,
            baseline.time_warn_ratio,
            baseline.time_fail_ratio,
        )
        _check_ratio!(
            warnings,
            reasons,
            "memory ratio",
            memory_ratio,
            baseline.memory_warn_ratio,
            baseline.memory_fail_ratio,
        )
        _check_ratio!(
            warnings,
            reasons,
            "allocation ratio",
            alloc_ratio,
            baseline.alloc_warn_ratio,
            baseline.alloc_fail_ratio,
        )
        passed = isempty(reasons)
        warning_only = passed && !isempty(warnings)

        push!(
            comparisons,
            (
                id = measurement.id,
                feature = measurement.feature,
                passed = passed,
                status = passed ? (warning_only ? :warn : :pass) : :fail,
                warning_only = warning_only,
                reasons = reasons,
                warnings = warnings,
                measurement = measurement,
                baseline = baseline,
                release_blocker = baseline.release_blocker,
            ),
        )
    end

    return comparisons
end

function write_baselines(path::AbstractString, measurements)
    previous_limits = isfile(path) ? load_baselines(path).cases : Dict{String, Any}()
    benchmark_samples = isempty(measurements) ? DEFAULT_SAMPLES : first(measurements).samples
    benchmark_seconds = isempty(measurements) ? DEFAULT_SECONDS : first(measurements).seconds
    benchmark_evals = isempty(measurements) ? DEFAULT_EVALS : first(measurements).evals
    rows = Dict{String, Any}[]

    for measurement in sort!(collect(measurements); by = entry -> entry.id)
        limits = get(previous_limits, measurement.id, nothing)
        push!(
            rows,
            Dict{String, Any}(
                "id" => measurement.id,
                "feature" => string(measurement.feature),
                "description" => measurement.description,
                "time_ns" => measurement.time_ns,
                "memory_bytes" => measurement.memory_bytes,
                "allocs" => measurement.allocs,
                "time_warn_ratio" => isnothing(limits) ? measurement.time_warn_ratio : limits.time_warn_ratio,
                "time_fail_ratio" => isnothing(limits) ? measurement.time_fail_ratio : limits.time_fail_ratio,
                "memory_warn_ratio" => isnothing(limits) ? measurement.memory_warn_ratio : limits.memory_warn_ratio,
                "memory_fail_ratio" => isnothing(limits) ? measurement.memory_fail_ratio : limits.memory_fail_ratio,
                "alloc_warn_ratio" => isnothing(limits) ? measurement.alloc_warn_ratio : limits.alloc_warn_ratio,
                "alloc_fail_ratio" => isnothing(limits) ? measurement.alloc_fail_ratio : limits.alloc_fail_ratio,
                "release_blocker" => isnothing(limits) ? measurement.release_blocker : limits.release_blocker,
            ),
        )
    end

    payload = Dict{String, Any}(
        "baseline_version" => 2,
        "benchmark_samples" => benchmark_samples,
        "benchmark_seconds" => benchmark_seconds,
        "benchmark_evals" => benchmark_evals,
        "cases" => rows,
    )

    open(path, "w") do io
        TOML.print(io, payload)
    end
    return path
end

_baseline_ratio(entry, preferred_key::AbstractString, fallback_key::AbstractString, default::Float64) =
    Float64(get(entry, preferred_key, get(entry, fallback_key, default)))

function _check_ratio!(
        warnings::Vector{String},
        failures::Vector{String},
        label::AbstractString,
        ratio::Float64,
        warn_limit::Float64,
        fail_limit::Float64,
    )
    rounded_ratio = round(ratio; digits = 2)
    if ratio > fail_limit
        push!(failures, "$label $rounded_ratio exceeds fail limit $fail_limit")
    elseif ratio > warn_limit
        push!(warnings, "$label $rounded_ratio exceeds warning limit $warn_limit")
    end
    return nothing
end

function _hyperbolic_runner()
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 200)
    prob = HyperbolicProblem(
        law,
        mesh,
        HLLCSolver(),
        CellCenteredMUSCL(),
        TransmissiveBC(),
        TransmissiveBC(),
        x -> x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1);
        final_time = 0.1,
        cfl = 0.4,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    algorithm = SSPRK33()
    return () -> solve(prob, algorithm; adaptive = false, dt = dt0, save_everystep = false, save_start = false, dense = false)
end

function _parabolic_runner()
    tri = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, 8, 8; single_boundary = true)
    mesh = FVMGeometry(tri)
    boundary_value = (x, y, t, u, p) -> 0.0
    BCs = BoundaryConditions(mesh, boundary_value, Dirichlet)
    initial_condition = [sin(pi * x) * sin(pi * y) for (x, y) in DelaunayTriangulation.each_point(tri)]
    diffusion = (x, y, t, u, p) -> 1.0
    prob = FVMProblem(mesh, BCs; diffusion_function = diffusion, initial_condition, final_time = 0.01)
    algorithm = Tsit5()
    return () -> solve(prob, algorithm; adaptive = false, dt = 5.0e-4, save_everystep = false, save_start = false, dense = false)
end

function _mhd_runner()
    eos = IdealGasEOS(5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 32, 32)
    prob = HyperbolicProblem2D(
        law,
        mesh,
        HLLDSolver(),
        NoReconstruction(),
        PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(),
        (x, y) -> SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0);
        final_time = 0.03,
        cfl = 0.3,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    algorithm = SSPRK33(; stage_limiter! = mhd_stage_limiter(ode_prob.p))
    return () -> solve(prob, algorithm; adaptive = false, dt = dt0, save_everystep = false, save_start = false, dense = false)
end

function _relativistic_runner()
    eos = IdealGasEOS(5.0 / 3.0)
    law = SRMHDEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 160)
    amplitude = 0.01
    prob = HyperbolicProblem(
        law,
        mesh,
        HLLSolver(),
        CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(),
        x -> SVector(
            1.0,
            0.0,
            amplitude * sin(2pi * x),
            amplitude * cos(2pi * x),
            1.0,
            1.0,
            amplitude * sin(2pi * x),
            amplitude * cos(2pi * x),
        );
        final_time = 0.1,
        cfl = 0.4,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    algorithm = SSPRK33()
    return () -> solve(prob, algorithm; adaptive = false, dt = dt0, save_everystep = false, save_start = false, dense = false)
end

end

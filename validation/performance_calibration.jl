module RepoPerformanceCalibration

using Statistics
using TOML

include(joinpath(@__DIR__, "performance_baselines.jl"))
using .RepoPerformanceBaselines

const DEFAULT_REPETITIONS = 5

function default_repetitions()
    repetitions = tryparse(Int, get(ENV, "FVM_PERFORMANCE_CALIBRATION_REPETITIONS", string(DEFAULT_REPETITIONS)))
    return something(repetitions, DEFAULT_REPETITIONS)
end

function run_calibration(;
        repetitions::Integer = default_repetitions(),
        baseline_path::AbstractString = RepoPerformanceBaselines.default_baseline_path(),
        parameters = RepoPerformanceBaselines.benchmark_parameters(),
    )
    repetitions > 0 || throw(ArgumentError("performance calibration requires at least one repetition"))
    baselines = RepoPerformanceBaselines.load_baselines(baseline_path)
    trials = [RepoPerformanceBaselines.run_suite(; parameters) for _ in 1:repetitions]
    return summarize_trials(
        trials,
        baselines;
        baseline_path,
        benchmark_parameters = parameters,
    )
end

function summarize_trials(
        trials,
        baselines;
        baseline_path::AbstractString = RepoPerformanceBaselines.default_baseline_path(),
        benchmark_parameters = RepoPerformanceBaselines.benchmark_parameters(),
    )
    isempty(trials) && throw(ArgumentError("performance calibration requires at least one completed trial"))

    trial_maps = [Dict(entry.id => entry for entry in trial) for trial in trials]
    cases = NamedTuple[]

    for case_id in sort!(collect(keys(baselines.cases)); by = identity)
        baseline = baselines.cases[case_id]
        measurements = [trial_map[case_id] for trial_map in trial_maps if haskey(trial_map, case_id)]
        length(measurements) == length(trial_maps) ||
            throw(ArgumentError("performance calibration is missing measurements for `$case_id`"))

        time_ratios = [measurement.time_ns / baseline.time_ns for measurement in measurements]
        memory_ratios = baseline.memory_bytes == 0 ? fill(1.0, length(measurements)) :
            [measurement.memory_bytes / baseline.memory_bytes for measurement in measurements]
        alloc_ratios = baseline.allocs == 0 ? fill(1.0, length(measurements)) :
            [measurement.allocs / baseline.allocs for measurement in measurements]

        time_summary = _metric_summary(time_ratios, baseline.time_warn_ratio, baseline.time_fail_ratio)
        memory_summary = _metric_summary(memory_ratios, baseline.memory_warn_ratio, baseline.memory_fail_ratio)
        alloc_summary = _metric_summary(alloc_ratios, baseline.alloc_warn_ratio, baseline.alloc_fail_ratio)
        status = _combine_status((time_summary, memory_summary, alloc_summary))

        push!(
            cases,
            (
                id = case_id,
                feature = baseline.feature,
                description = baseline.description,
                release_blocker = baseline.release_blocker,
                status = status,
                repetitions = length(measurements),
                baseline_time_ns = baseline.time_ns,
                baseline_memory_bytes = baseline.memory_bytes,
                baseline_allocs = baseline.allocs,
                time = time_summary,
                memory = memory_summary,
                allocations = alloc_summary,
            ),
        )
    end

    return (
        calibration_version = 1,
        repetitions = length(trials),
        baseline_path = baseline_path,
        benchmark_parameters = benchmark_parameters,
        cases = cases,
    )
end

function write_report(path::AbstractString, calibration)
    payload = Dict{String, Any}(
        "performance_calibration_version" => calibration.calibration_version,
        "repetitions" => calibration.repetitions,
        "baseline_path" => calibration.baseline_path,
        "benchmark_samples" => calibration.benchmark_parameters.samples,
        "benchmark_seconds" => calibration.benchmark_parameters.seconds,
        "benchmark_evals" => calibration.benchmark_parameters.evals,
        "cases" => [
            Dict{String, Any}(
                    "id" => case.id,
                    "feature" => string(case.feature),
                    "description" => case.description,
                    "release_blocker" => case.release_blocker,
                    "status" => string(case.status),
                    "repetitions" => case.repetitions,
                    "baseline_time_ns" => case.baseline_time_ns,
                    "baseline_memory_bytes" => case.baseline_memory_bytes,
                    "baseline_allocs" => case.baseline_allocs,
                    "time_ratio_min" => case.time.minimum,
                    "time_ratio_median" => case.time.median,
                    "time_ratio_max" => case.time.maximum,
                    "time_warn_ratio" => case.time.warn_limit,
                    "time_fail_ratio" => case.time.fail_limit,
                    "time_warn_headroom" => case.time.warn_headroom,
                    "time_fail_headroom" => case.time.fail_headroom,
                    "memory_ratio_min" => case.memory.minimum,
                    "memory_ratio_median" => case.memory.median,
                    "memory_ratio_max" => case.memory.maximum,
                    "memory_warn_ratio" => case.memory.warn_limit,
                    "memory_fail_ratio" => case.memory.fail_limit,
                    "memory_warn_headroom" => case.memory.warn_headroom,
                    "memory_fail_headroom" => case.memory.fail_headroom,
                    "alloc_ratio_min" => case.allocations.minimum,
                    "alloc_ratio_median" => case.allocations.median,
                    "alloc_ratio_max" => case.allocations.maximum,
                    "alloc_warn_ratio" => case.allocations.warn_limit,
                    "alloc_fail_ratio" => case.allocations.fail_limit,
                    "alloc_warn_headroom" => case.allocations.warn_headroom,
                    "alloc_fail_headroom" => case.allocations.fail_headroom,
                ) for case in calibration.cases
        ],
    )

    open(path, "w") do io
        TOML.print(io, payload)
    end
    return path
end

function markdown_report(calibration)
    lines = String[]
    push!(lines, "# Performance Calibration")
    push!(lines, "")
    push!(lines, "- repetitions: `$(calibration.repetitions)`")
    push!(lines, "- baseline file: `$(calibration.baseline_path)`")
    push!(
        lines,
        "- benchmark parameters: `samples=$(calibration.benchmark_parameters.samples)`, `seconds=$(calibration.benchmark_parameters.seconds)`, `evals=$(calibration.benchmark_parameters.evals)`",
    )
    push!(lines, "")
    push!(lines, "| Case | Status | Runtime Ratio (min / median / max) | Memory Ratio (min / median / max) | Allocation Ratio (min / median / max) |")
    push!(lines, "| --- | --- | --- | --- | --- |")
    for case in calibration.cases
        push!(
            lines,
            "| `$(case.id)` | `$(case.status)` | $(round(case.time.minimum; digits = 2)) / $(round(case.time.median; digits = 2)) / $(round(case.time.maximum; digits = 2)) | $(round(case.memory.minimum; digits = 2)) / $(round(case.memory.median; digits = 2)) / $(round(case.memory.maximum; digits = 2)) | $(round(case.allocations.minimum; digits = 2)) / $(round(case.allocations.median; digits = 2)) / $(round(case.allocations.maximum; digits = 2)) |",
        )
    end
    return join(lines, "\n")
end

function _metric_summary(values, warn_limit::Float64, fail_limit::Float64)
    rounded_minimum = round(minimum(values); digits = 3)
    rounded_median = round(median(values); digits = 3)
    rounded_maximum = round(maximum(values); digits = 3)
    warn_headroom = round(warn_limit / rounded_maximum; digits = 3)
    fail_headroom = round(fail_limit / rounded_maximum; digits = 3)
    status = rounded_maximum > fail_limit ? :fail : (rounded_maximum > warn_limit ? :warn : :pass)
    return (
        minimum = rounded_minimum,
        median = rounded_median,
        maximum = rounded_maximum,
        warn_limit = warn_limit,
        fail_limit = fail_limit,
        warn_headroom = warn_headroom,
        fail_headroom = fail_headroom,
        status = status,
    )
end

function _combine_status(metrics)
    any(metric -> metric.status == :fail, metrics) && return :fail
    any(metric -> metric.status == :warn, metrics) && return :warn
    return :pass
end

end

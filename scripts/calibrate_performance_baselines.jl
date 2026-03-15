#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "validation", "performance_calibration.jl"))
using .RepoPerformanceCalibration

function parse_args(args)
    options = Dict{String, String}()
    for arg in args
        startswith(arg, "--") || error("Unsupported argument `$arg`.")
        key, value = occursin('=', arg) ? split(arg[3:end], '='; limit = 2) : (arg[3:end], "true")
        options[key] = value
    end
    return options
end

function default_output_path()
    reports_dir = joinpath(@__DIR__, "..", "validation", "reports")
    mkpath(reports_dir)
    return joinpath(reports_dir, "performance_calibration.toml")
end

function markdown_output_path(report_path::AbstractString)
    root, _ = splitext(report_path)
    return root * ".md"
end

function main(args)
    options = parse_args(args)
    repetitions = parse(Int, get(options, "repetitions", string(RepoPerformanceCalibration.default_repetitions())))
    baseline_path = get(options, "baseline", joinpath(@__DIR__, "..", "validation", "performance_baselines.toml"))
    output_path = get(options, "output", default_output_path())

    calibration = RepoPerformanceCalibration.run_calibration(;
        repetitions,
        baseline_path,
    )
    RepoPerformanceCalibration.write_report(output_path, calibration)

    markdown_path = markdown_output_path(output_path)
    open(markdown_path, "w") do io
        write(io, RepoPerformanceCalibration.markdown_report(calibration))
        write(io, "\n")
    end

    println("Performance calibration written to: $output_path")
    println("Markdown summary written to: $markdown_path")
    println()
    println(RepoPerformanceCalibration.markdown_report(calibration))
    return nothing
end

main(ARGS)

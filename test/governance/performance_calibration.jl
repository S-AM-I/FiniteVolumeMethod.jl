using Test

include(joinpath(dirname(dirname(@__DIR__)), "validation", "performance_calibration.jl"))
using .RepoPerformanceCalibration

baselines = (
    cases = Dict(
        "hyperbolic_euler_1d" => (
            id = "hyperbolic_euler_1d",
            feature = :hyperbolic,
            description = "Synthetic calibration baseline.",
            time_ns = 100,
            memory_bytes = 200,
            allocs = 10,
            time_warn_ratio = 1.5,
            time_fail_ratio = 2.0,
            memory_warn_ratio = 1.1,
            memory_fail_ratio = 1.5,
            alloc_warn_ratio = 1.05,
            alloc_fail_ratio = 1.25,
            release_blocker = true,
        ),
    ),
)

trials = [
    [(id = "hyperbolic_euler_1d", feature = :hyperbolic, description = "Synthetic calibration baseline.", time_ns = 110, memory_bytes = 205, allocs = 10)],
    [(id = "hyperbolic_euler_1d", feature = :hyperbolic, description = "Synthetic calibration baseline.", time_ns = 120, memory_bytes = 210, allocs = 10)],
    [(id = "hyperbolic_euler_1d", feature = :hyperbolic, description = "Synthetic calibration baseline.", time_ns = 125, memory_bytes = 215, allocs = 11)],
]

calibration = RepoPerformanceCalibration.summarize_trials(
    trials,
    baselines;
    baseline_path = "validation/performance_baselines.toml",
    benchmark_parameters = (samples = 5, seconds = 0.5, evals = 1),
)

@testset "Performance Calibration" begin
    @test calibration.repetitions == 3
    @test length(calibration.cases) == 1

    case = only(calibration.cases)
    @test case.id == "hyperbolic_euler_1d"
    @test case.status == :warn
    @test case.time.minimum == 1.1
    @test case.time.median == 1.2
    @test case.time.maximum == 1.25
    @test case.memory.maximum == 1.075
    @test case.allocations.maximum == 1.1

    markdown = RepoPerformanceCalibration.markdown_report(calibration)
    @test occursin("Performance Calibration", markdown)
    @test occursin("hyperbolic_euler_1d", markdown)
end

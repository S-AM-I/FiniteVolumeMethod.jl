using Test

include(joinpath(dirname(@__DIR__), "validation", "performance_baselines.jl"))
using .RepoPerformanceBaselines

const REPO_ROOT = dirname(@__DIR__)
const BASELINE_PATH = joinpath(REPO_ROOT, "validation", "performance_baselines.toml")

function update_performance_baselines()
    return get(ENV, "FVM_UPDATE_PERFORMANCE_BASELINES", "false") == "true"
end

measurements = RepoPerformanceBaselines.run_suite()
if update_performance_baselines()
    RepoPerformanceBaselines.write_baselines(BASELINE_PATH, measurements)
end

baselines = RepoPerformanceBaselines.load_baselines(BASELINE_PATH)
comparisons = RepoPerformanceBaselines.compare_to_baselines(measurements, baselines)

@testset "Performance Baselines" begin
    @test !isempty(measurements)
    @test length(measurements) == 4
    @test isfile(BASELINE_PATH)

    for comparison in comparisons
        @testset "$(comparison.id)" begin
            isempty(comparison.warnings) || @info("Performance baseline warning", id = comparison.id, warnings = comparison.warnings)
            comparison.passed || @info("Performance baseline regression", id = comparison.id, reasons = comparison.reasons)
            @test comparison.passed
        end
    end
end

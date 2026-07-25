using Test

include(joinpath(dirname(dirname(@__DIR__)), "validation", "backend_parity.jl"))
using .RepoBackendParity

@testset "Backend parity" begin
    summary = RepoBackendParity.summarize_suite()

    @test !isempty(summary.results)
    @test summary.status in (:pass, :not_run, :fail)

    for result in summary.results
        @test result.status in (:pass, :not_run)
        if result.status == :pass
            @test isempty(result.failures)
            @test !isempty(result.metrics)
        else
            @test result.backend == "cuda"
            @test occursin("CUDA", result.rationale)
        end
    end
end

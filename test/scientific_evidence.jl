using FiniteVolumeMethod
using Test

include(joinpath(dirname(@__DIR__), "validation", "manifest.jl"))
include(joinpath(dirname(@__DIR__), "validation", "evidence_runner.jl"))
using .RepoEvidenceRunner
using .RepoValidationManifest

const REPO_ROOT = dirname(@__DIR__)
manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

@testset verbose = true "Scientific Evidence Suite" begin
    mktempdir() do output_dir
        summaries = RepoEvidenceRunner.run_evidence_suite(manifest; repo_root = REPO_ROOT, output_dir)
        @test length(summaries) == length(manifest.scientific_evidence)
        for (entry, summary) in zip(manifest.scientific_evidence, summaries)
            @test summary["status"] == "pass"
            @test isfile(RepoEvidenceRunner.evidence_summary_path(entry.id; output_dir))
            if entry.summary_required
                @test summary["recorded_result_count"] > 0
            end
        end
    end
end

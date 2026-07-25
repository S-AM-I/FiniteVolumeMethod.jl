using Test

include(joinpath(dirname(dirname(@__DIR__)), "validation", "manifest.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "validation", "evidence_runner.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "validation", "summary_replay.jl"))
using .RepoEvidenceRunner
using .RepoSummaryReplay
using .RepoValidationManifest

const REPO_ROOT = dirname(dirname(@__DIR__))
manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

@testset "Evidence summary replay" begin
    selected_ids = Set(["evidence-euler-mms", "evidence-flux-balance"])
    entries = filter(entry -> entry.id in selected_ids, manifest.scientific_evidence)
    @test length(entries) == 2

    mktempdir() do tmpdir
        reference_dir = joinpath(tmpdir, "reference")
        candidate_dir = joinpath(tmpdir, "candidate")

        RepoEvidenceRunner.run_evidence_entries(entries; repo_root = REPO_ROOT, output_dir = reference_dir)
        RepoEvidenceRunner.run_evidence_entries(entries; repo_root = REPO_ROOT, output_dir = candidate_dir)

        diffs = RepoSummaryReplay.compare_summary_directories(
            reference_dir,
            candidate_dir;
            entry_ids = selected_ids,
        )
        @test isempty(diffs)
    end
end

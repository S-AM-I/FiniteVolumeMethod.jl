using FiniteVolumeMethod
using Test

include(joinpath(dirname(@__DIR__), "validation", "manifest.jl"))
using .RepoValidationManifest
include(joinpath(dirname(@__DIR__), "validation", "evidence_runner.jl"))
using .RepoEvidenceRunner

const REPO_ROOT = dirname(@__DIR__)
manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

function stable_claim_features(manifest)
    return sort!(
        [
            feature for (feature, entry) in manifest.features
                if entry.maturity == :stable && entry.role == :claim_bearing_solver
        ];
        by = string,
    )
end

function smoke_entries(manifest)
    entries = RepoValidationManifest.ScientificEvidenceEntry[]
    for feature in stable_claim_features(manifest)
        feature_entries = RepoValidationManifest.scientific_evidence_for_feature(manifest, feature)
        verification_entries = filter(entry -> entry.ladder_stage == :verification, feature_entries)
        push!(entries, isempty(verification_entries) ? first(feature_entries) : first(verification_entries))
    end
    return entries
end

selected_entries = smoke_entries(manifest)

@testset verbose = true "Scientific Evidence Smoke" begin
    @test !isempty(selected_entries)
    @test length(selected_entries) == length(stable_claim_features(manifest))
    @test all(entry.ladder_stage == :verification for entry in selected_entries)

    mktempdir() do output_dir
        summaries = RepoEvidenceRunner.run_evidence_entries(selected_entries; repo_root = REPO_ROOT, output_dir)
        @test length(summaries) == length(selected_entries)
        for (entry, summary) in zip(selected_entries, summaries)
            @test summary["status"] == "pass"
            @test summary["ladder_stage"] == "verification"
            @test isfile(RepoEvidenceRunner.evidence_summary_path(entry.id; output_dir))
            if entry.summary_required
                @test summary["recorded_result_count"] > 0
            end
        end
    end
end

using Test

include(joinpath(dirname(@__DIR__), "validation", "manifest.jl"))
include(joinpath(dirname(@__DIR__), "validation", "release_audit.jl"))
using .RepoReleaseAudit
using .RepoValidationManifest

const REPO_ROOT = dirname(@__DIR__)
manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

function release_audit_output_root()
    return get(ENV, "FVM_RELEASE_AUDIT_OUTPUT_ROOT", joinpath(tempdir(), "fvm-release-audit"))
end

function rerun_release_audit_evidence()
    return get(ENV, "FVM_RELEASE_AUDIT_RERUN", "true") == "true"
end

@testset "Release Audit" begin
    audit = RepoReleaseAudit.audit_release(
        manifest;
        repo_root = REPO_ROOT,
        output_root = release_audit_output_root(),
        rerun_evidence = rerun_release_audit_evidence(),
    )

    @test isempty(audit.findings)
    @test audit.replay_report["status"] == "pass"
    @test length(audit.expected_evidence_entries) == length(
        filter(name -> endswith(name, ".toml"), readdir(audit.outputs.summaries_dir)),
    )
    @test Set(entry["feature"] for entry in audit.bundle_index["bundles"]) == Set(string.(audit.stable_features))
end

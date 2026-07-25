using Test

include(joinpath(dirname(dirname(@__DIR__)), "validation", "manifest.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "validation", "release_audit.jl"))
using .RepoReleaseAudit
using .RepoValidationManifest

const REPO_ROOT = dirname(dirname(@__DIR__))
manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

function release_audit_output_root()
    return get(ENV, "FVM_RELEASE_AUDIT_OUTPUT_ROOT", joinpath(tempdir(), "fvm-release-audit"))
end

function rerun_release_audit_evidence()
    return get(ENV, "FVM_RELEASE_AUDIT_RERUN", "true") == "true"
end

function run_release_audit_performance()
    return get(ENV, "FVM_RELEASE_AUDIT_PERFORMANCE", "true") == "true"
end

function run_release_audit_backend_parity()
    return get(ENV, "FVM_RELEASE_AUDIT_BACKEND_PARITY", "true") == "true"
end

@testset "Release Audit" begin
    audit = RepoReleaseAudit.audit_release(
        manifest;
        repo_root = REPO_ROOT,
        output_root = release_audit_output_root(),
        rerun_evidence = rerun_release_audit_evidence(),
        run_performance_audit = run_release_audit_performance(),
        run_backend_parity = run_release_audit_backend_parity(),
    )

    @test isempty(audit.findings)
    @test audit.replay_report["status"] == "pass"
    @test audit.performance_audit.passed
    @test audit.backend_parity.passed
    @test isfile(joinpath(audit.outputs.output_root, "performance_report.toml"))
    @test isfile(joinpath(audit.outputs.output_root, "backend_parity_report.toml"))
    @test haskey(audit.provenance, "reference_dataset_artifact")
    @test length(audit.expected_evidence_entries) == length(
        filter(name -> endswith(name, ".toml"), readdir(audit.outputs.summaries_dir)),
    )
    @test Set(entry["feature"] for entry in audit.bundle_index["bundles"]) == Set(string.(audit.stable_features))
end

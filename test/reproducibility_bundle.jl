using Test
using TOML

include(joinpath(dirname(@__DIR__), "validation", "manifest.jl"))
include(joinpath(dirname(@__DIR__), "validation", "reproducibility.jl"))
include(joinpath(dirname(@__DIR__), "validation", "generate_report.jl"))
using .RepoReproducibility
using .RepoValidationManifest
using .ValidationReport

const REPO_ROOT = dirname(@__DIR__)
manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

@testset "Reproduction bundle packaging" begin
    mktempdir() do tmpdir
        bundle_root = joinpath(tmpdir, "bundles")
        report_path = joinpath(tmpdir, "reports", "validation_report.md")

        bundles = RepoReproducibility.build_reproduction_bundles(
            manifest;
            repo_root = REPO_ROOT,
            output_root = bundle_root,
            features = [:coupling],
            rerun_evidence = true,
        )

        @test length(bundles) == 1
        bundle = only(bundles)
        @test bundle.feature == :coupling
        @test isfile(bundle.bundle_manifest_path)
        @test isfile(bundle.bundle_readme_path)
        @test isdir(bundle.summaries_dir)
        @test isdir(bundle.artifacts_dir)

        bundle_manifest = TOML.parsefile(bundle.bundle_manifest_path)
        @test bundle_manifest["feature"] == "coupling"
        @test length(bundle_manifest["evidence"]) == 3
        @test all(entry["status"] == "pass" for entry in bundle_manifest["evidence"])

        artifact_names = Set(readdir(bundle.artifacts_dir))
        @test artifact_names == Set(
            [
                "coupling_cooling_reference.png",
                "coupling_mass_conservation.png",
                "coupling_nullsource_identity.png",
            ],
        )

        report = ValidationReport.generate(
            joinpath(REPO_ROOT, "validation", "manifest.toml"),
            report_path;
            summary_dir = bundle.summaries_dir,
            bundle_dir = bundle_root,
        )
        @test isfile(report_path)
        @test occursin("## Executed Evidence Results", report)
        @test occursin("## Reproduction Bundles", report)
        @test occursin("evidence-coupling-nullsource-identity", report)
    end
end

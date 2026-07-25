using Test
using TOML

include(joinpath(dirname(dirname(@__DIR__)), "validation", "manifest.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "validation", "reproducibility.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "validation", "generate_report.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "validation", "release_packaging.jl"))
using .RepoReproducibility
using .RepoValidationManifest
using .RepoReleasePackaging
using .ValidationReport

const REPO_ROOT = dirname(dirname(@__DIR__))
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
        @test isfile(joinpath(bundle_root, "bundle_index.toml"))
        @test isfile(joinpath(bundle_root, "README.md"))

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

@testset "Release output packaging" begin
    mktempdir() do tmpdir
        outputs = RepoReleasePackaging.build_release_outputs(
            manifest;
            repo_root = REPO_ROOT,
            output_root = joinpath(tmpdir, "release_outputs"),
            bundle_features = [:hyperbolic, :parabolic],
            rerun_evidence = true,
        )

        @test isfile(outputs.report_path)
        @test isfile(outputs.bundle_index_path)
        @test isfile(outputs.release_index_path)
        @test isfile(outputs.provenance_path)
        @test isfile(outputs.replay_report_path)
        @test isdir(outputs.summaries_dir)
        @test isdir(outputs.bundles_dir)
        @test isdir(outputs.replay_summaries_dir)
        @test length(outputs.bundles) == 2
        @test length(filter(name -> endswith(name, ".toml"), readdir(outputs.summaries_dir))) == 6

        bundle_index = TOML.parsefile(outputs.bundle_index_path)
        @test bundle_index["bundle_count"] == 2
        @test Set(entry["feature"] for entry in bundle_index["bundles"]) == Set(["hyperbolic", "parabolic"])

        provenance = TOML.parsefile(outputs.provenance_path)
        @test provenance["provenance_version"] == 1
        @test provenance["manifest_version"] == manifest.manifest_version
        @test provenance["bundle_features"] == ["hyperbolic", "parabolic"]
        @test provenance["run_summary_replay"] == true
        @test Set(provenance["replay_entry_ids"]) == Set(["evidence-euler-mms", "evidence-poisson-convergence"])
        @test provenance["reference_dataset_artifact"]["name"] == "reference_datasets"
        @test length(provenance["reference_dataset_artifact"]["git_tree_sha1"]) == 40
        @test !isempty(provenance["reference_dataset_artifact"]["files"])
        @test !isempty(provenance["julia_version"])
        @test !isempty(provenance["git_commit"])

        replay_report = TOML.parsefile(outputs.replay_report_path)
        @test replay_report["replay_version"] == 1
        @test replay_report["status"] == "pass"
        @test Set(replay_report["entry_ids"]) == Set(["evidence-euler-mms", "evidence-poisson-convergence"])

        hyperbolic_artifacts = Set(readdir(joinpath(outputs.bundles_dir, "hyperbolic", "artifacts")))
        @test "euler_mms_solution.png" in hyperbolic_artifacts
        @test "sod_grid_convergence_comparison.png" in hyperbolic_artifacts

        parabolic_artifacts = Set(readdir(joinpath(outputs.bundles_dir, "parabolic", "artifacts")))
        @test "poisson_convergence_solution.png" in parabolic_artifacts
        @test "barenblatt_pattle_solution.png" in parabolic_artifacts

        report = read(outputs.report_path, String)
        @test occursin("## Executed Evidence Results", report)
        @test occursin("## Reproduction Bundles", report)
        @test occursin("evidence-euler-mms", report)
        @test occursin("evidence-poisson-convergence", report)
        @test !occursin("evidence-mhd-alfven", report)

        release_index = read(outputs.release_index_path, String)
        @test occursin("provenance.toml", release_index)
        @test occursin("replay_report.toml", release_index)
        @test occursin("Summary replay status: `pass`", release_index)
        @test occursin("Reference dataset artifact:", release_index)
    end
end

@testset "Release output packaging without replay" begin
    mktempdir() do tmpdir
        outputs = RepoReleasePackaging.build_release_outputs(
            manifest;
            repo_root = REPO_ROOT,
            output_root = joinpath(tmpdir, "release_outputs"),
            bundle_features = [:hyperbolic],
            rerun_evidence = true,
            run_summary_replay = false,
        )

        @test isfile(outputs.provenance_path)
        @test isfile(outputs.replay_report_path)
        @test !isdir(outputs.replay_summaries_dir)

        provenance = TOML.parsefile(outputs.provenance_path)
        @test provenance["run_summary_replay"] == false
        @test isempty(provenance["replay_entry_ids"])

        replay_report = TOML.parsefile(outputs.replay_report_path)
        @test replay_report["status"] == "not_run"
        @test replay_report["replay_dir"] == "not_run"
        @test isempty(replay_report["entry_ids"])

        release_index = read(outputs.release_index_path, String)
        @test occursin("Summary replay status: `not_run`", release_index)
    end
end

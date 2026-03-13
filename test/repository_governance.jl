using FiniteVolumeMethod
using TOML
using Test

include(joinpath(dirname(@__DIR__), "validation", "manifest.jl"))
using .RepoValidationManifest

const REPO_ROOT = dirname(@__DIR__)
manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

@testset "Validation manifest governance" begin
    @test !isempty(manifest.generated_pages)
    @test !isempty(manifest.scientific_evidence)
    @test !isempty(manifest.features)

    for entry in manifest.generated_pages
        @test isfile(joinpath(REPO_ROOT, entry.source))
        @test startswith(entry.page, "tutorials/") ||
            startswith(entry.page, "wyos/") ||
            startswith(entry.page, "verification/") ||
            startswith(entry.page, "hyperbolic/tutorials/")
    end

    for entry in manifest.scientific_evidence
        @test isfile(joinpath(REPO_ROOT, entry.path))
        @test haskey(manifest.features, entry.feature)
        @test startswith(entry.path, "docs/src/literate_verification/")
        @test entry.runtime_tier in (:ci, :local_full, :manual)
        @test entry.category in (:code_verification, :analytical_benchmark, :experimental_validation)
        @test entry.reference_kind in (
            :exact_solution, :manufactured_solution, :literature_table, :reference_dataset, :discrete_invariant
        )
        @test !isempty(entry.reference_source)
        @test !isempty(entry.metric)
        @test !isempty(entry.acceptance)
    end

    for entry in values(manifest.features)
        @test entry.maturity in (:stable, :provisional, :experimental)
        @test entry.validation in (:executed_examples, :convergence_verified, :targeted_tests, :smoke_tests)
        @test !isempty(entry.summary)
    end

    verification_pages = RepoValidationManifest.verification_pages(manifest)
    @test !isempty(verification_pages)
    for entry in verification_pages
        @test entry.category in (:code_verification, :analytical_benchmark, :experimental_validation)
        @test entry.reference_kind in (
            :exact_solution, :manufactured_solution, :literature_table, :reference_dataset, :discrete_invariant
        )
        @test !isnothing(entry.reference_source)
        @test !isnothing(entry.metric)
        @test !isnothing(entry.acceptance)
    end

    evidence_paths = Set(entry.path for entry in manifest.scientific_evidence)
    verification_sources = Set(entry.source for entry in verification_pages)
    @test evidence_paths ⊆ verification_sources

    package_features = Set(FiniteVolumeMethod.supported_features())
    manifest_features = Set(keys(manifest.features))
    @test package_features == manifest_features

    for feature in package_features
        @test FiniteVolumeMethod.feature_maturity(feature) == manifest.features[feature].maturity
        @test FiniteVolumeMethod.feature_validation_status(feature) == manifest.features[feature].validation
    end
end

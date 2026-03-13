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
    end

    for entry in values(manifest.features)
        @test entry.maturity in (:stable, :provisional, :experimental)
        @test entry.validation in (:executed_examples, :convergence_verified, :targeted_tests, :smoke_tests)
        @test !isempty(entry.summary)
    end

    package_features = Set(FiniteVolumeMethod.supported_features())
    manifest_features = Set(keys(manifest.features))
    @test package_features == manifest_features

    for feature in package_features
        @test FiniteVolumeMethod.feature_maturity(feature) == manifest.features[feature].maturity
        @test FiniteVolumeMethod.feature_validation_status(feature) == manifest.features[feature].validation
    end
end

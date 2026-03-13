using FiniteVolumeMethod
using Test

include(joinpath(dirname(@__DIR__), "validation", "manifest.jl"))
using .RepoValidationManifest

const REPO_ROOT = dirname(@__DIR__)
manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))
const RELEASE_CHECKLIST = joinpath(REPO_ROOT, "validation", "RELEASE_CHECKLIST.md")

@testset "Validation manifest governance" begin
    @test manifest.support_policy == :current_lts_and_stable
    @test isfile(RELEASE_CHECKLIST)
    @test !isempty(manifest.generated_pages)
    @test !isempty(manifest.scientific_evidence)
    @test !isempty(manifest.features)
    @test !isempty(manifest.exclusions)

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
            :exact_solution, :manufactured_solution, :literature_table, :reference_dataset, :discrete_invariant,
        )
        @test !isempty(entry.reference_source)
        @test !isempty(entry.metric)
        @test !isempty(entry.acceptance)
        @test entry.solver_family in (:parabolic, :hyperbolic, :mhd_ct, :relativistic, :amr, :coupling)
        @test entry.precision_policy in (:float64_cpu_reference, :not_applicable)
        @test entry.random_seed_policy in (:deterministic, :fixed_seed, :not_applicable)
        @test !isempty(entry.expected_artifacts)
    end

    for entry in values(manifest.features)
        @test entry.maturity in (:stable, :provisional, :experimental)
        @test entry.validation in (:executed_examples, :convergence_verified, :targeted_tests, :smoke_tests)
        @test entry.role in (:claim_bearing_solver, :research_support_tooling, :experimental_sandbox)
        if !isnothing(entry.solver_family)
            @test entry.solver_family in (:parabolic, :hyperbolic, :mhd_ct, :relativistic, :amr, :coupling, :research_tooling)
        end
        if !isnothing(entry.precision_policy)
            @test entry.precision_policy in (:float64_cpu_reference, :not_applicable)
        end
        if !isnothing(entry.random_seed_policy)
            @test entry.random_seed_policy in (:deterministic, :fixed_seed, :not_applicable)
        end
        if !isnothing(entry.backend_policy)
            @test entry.backend_policy in (:cpu_reference, :cpu_reference_gpu_experimental)
        end
        @test !isempty(entry.summary)
        @test !isempty(entry.limitations)
    end

    verification_pages = RepoValidationManifest.verification_pages(manifest)
    @test !isempty(verification_pages)
    for entry in verification_pages
        @test entry.category in (:code_verification, :analytical_benchmark, :experimental_validation)
        @test entry.reference_kind in (
            :exact_solution, :manufactured_solution, :literature_table, :reference_dataset, :discrete_invariant,
        )
        @test !isnothing(entry.reference_source)
        @test !isnothing(entry.metric)
        @test !isnothing(entry.acceptance)
    end

    for entry in manifest.exclusions
        @test isfile(joinpath(REPO_ROOT, entry.path))
        if !isnothing(entry.feature)
            @test haskey(manifest.features, entry.feature)
        end
        @test entry.status in (:demoted, :manual_review)
        @test !isempty(entry.reason)
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
        @test FiniteVolumeMethod.feature_role(feature) == manifest.features[feature].role
        @test FiniteVolumeMethod.feature_solver_family(feature) == manifest.features[feature].solver_family
        @test FiniteVolumeMethod.feature_claim_policy(feature) ==
            RepoValidationManifest.feature_claim_policy(manifest.features[feature])
        @test FiniteVolumeMethod.feature_limitations(feature) == manifest.features[feature].limitations
    end

    stable_claim_features = [
        feature for (feature, entry) in manifest.features
            if entry.maturity == :stable && entry.role == :claim_bearing_solver
    ]
    @test !isempty(stable_claim_features)

    for feature in stable_claim_features
        @test !isempty(filter(entry -> entry.feature == feature, manifest.scientific_evidence))
        @test !isempty(filter(entry -> entry.feature == feature, manifest.generated_pages))
    end
end

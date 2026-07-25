using Test

include(joinpath(dirname(dirname(@__DIR__)), "validation", "reference_artifacts.jl"))
using .RepoReferenceArtifacts

const REPO_ROOT = dirname(dirname(@__DIR__))

@testset "Reference dataset artifact" begin
    info = RepoReferenceArtifacts.ensure_reference_datasets(; repo_root = REPO_ROOT)

    @test info.name == "reference_datasets"
    @test length(info.git_tree_sha1) == 40
    @test isdir(info.path)
    @test Set(info.files) == Set(RepoReferenceArtifacts.REFERENCE_DATASET_FILES)

    for filename in RepoReferenceArtifacts.REFERENCE_DATASET_FILES
        @test isfile(RepoReferenceArtifacts.reference_dataset_path(filename; repo_root = REPO_ROOT))
    end
end

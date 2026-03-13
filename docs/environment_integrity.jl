using Test

include(joinpath(dirname(@__DIR__), "validation", "project_integrity.jl"))
using .RepoProjectIntegrity

const REPO_ROOT = dirname(@__DIR__)

@testset "Docs project integrity" begin
    docs_env = RepoProjectIntegrity.check_project_integrity(
        joinpath(REPO_ROOT, "docs", "Project.toml"),
        ["docs"];
        repo_root = REPO_ROOT,
        local_modules = [:RepoValidationManifest, :ValidationReport, :RepoProjectIntegrity],
    )
    @test isempty(docs_env.missing)
end

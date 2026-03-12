using Dates
using FiniteVolumeMethod
using Test

include(joinpath(dirname(@__DIR__), "validation", "manifest.jl"))
using .RepoValidationManifest

ct() = Dates.format(now(), "HH:MM:SS")

function safe_include(path; name = basename(path))
    mod = @eval module $(gensym()) end
    @info "[$(ct())] Scientific evidence: $name"
    return @testset verbose = true "Evidence: $name" begin
        Base.include(mod, path)
    end
end

const REPO_ROOT = dirname(@__DIR__)
manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

@testset verbose = true "Scientific Evidence Suite" begin
    for entry in manifest.scientific_evidence
        safe_include(joinpath(REPO_ROOT, entry.path); name = entry.id)
    end
end

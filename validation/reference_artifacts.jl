module RepoReferenceArtifacts

using Pkg.Artifacts:
    artifact_exists,
    artifact_hash,
    artifact_path,
    bind_artifact!,
    create_artifact

const REFERENCE_DATASET_ARTIFACT = "reference_datasets"
const REFERENCE_DATASET_FILES = [
    "balsara_mhd_tests.json",
    "devahldavis_1983.json",
    "ghia_cavity_1982.json",
    "toro_tests.json",
]

repo_root() = normpath(joinpath(@__DIR__, ".."))
default_artifacts_toml() = joinpath(repo_root(), "Artifacts.toml")
reference_dataset_source_dir(; repo_root::AbstractString = RepoReferenceArtifacts.repo_root()) =
    joinpath(repo_root, "test", "reference_data")

function ensure_reference_datasets(;
        repo_root::AbstractString = RepoReferenceArtifacts.repo_root(),
        artifacts_toml::AbstractString = default_artifacts_toml(),
        force_rebuild::Bool = false,
    )
    hash = artifact_hash(REFERENCE_DATASET_ARTIFACT, artifacts_toml)
    if force_rebuild || isnothing(hash) || !artifact_exists(hash)
        hash = _materialize_reference_artifact(
            reference_dataset_source_dir(; repo_root);
            artifacts_toml,
        )
    end
    return artifact_info(hash)
end

function artifact_info(hash)
    path = artifact_path(hash)
    return (
        name = REFERENCE_DATASET_ARTIFACT,
        git_tree_sha1 = _hash_string(hash),
        path = path,
        files = sort!(readdir(path); by = identity),
    )
end

function reference_dataset_path(filename::AbstractString; repo_root::AbstractString = RepoReferenceArtifacts.repo_root())
    info = ensure_reference_datasets(; repo_root)
    path = joinpath(info.path, filename)
    isfile(path) || throw(ArgumentError("Unknown reference dataset `$filename` in artifact `$(info.name)`."))
    return path
end

function _materialize_reference_artifact(source_dir::AbstractString; artifacts_toml::AbstractString)
    isdir(source_dir) || throw(ArgumentError("Missing reference dataset source directory `$source_dir`."))
    _validate_source_dir(source_dir)
    hash = create_artifact() do artifact_dir
        for filename in REFERENCE_DATASET_FILES
            cp(joinpath(source_dir, filename), joinpath(artifact_dir, filename); force = true)
        end
    end
    bind_artifact!(artifacts_toml, REFERENCE_DATASET_ARTIFACT, hash; force = true)
    return hash
end

function _validate_source_dir(source_dir::AbstractString)
    missing = String[]
    for filename in REFERENCE_DATASET_FILES
        isfile(joinpath(source_dir, filename)) || push!(missing, filename)
    end
    isempty(missing) ||
        throw(ArgumentError("Reference dataset source directory is missing: $(join(missing, ", "))"))
    return nothing
end

_hash_string(hash::AbstractString) = hash
_hash_string(hash) = bytes2hex(hash.bytes)

end

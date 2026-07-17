# Model registry — migrated from Simu.jl SimuIO
# Provides save/load of simulation setup as a portable package.
# Stdlib imports are hoisted into the FVMIO module header.

const MODEL_PACKAGE_SCHEMA_VERSION = 1

"""
    save_model_package(mesh, physics_config, ic, path)

Save a simulation setup as a portable package (directory).
- `mesh`: The mesh object.
- `physics_config`: Dictionary defining physics.
- `ic`: Initial condition vector.
- `path`: Directory to create.
"""
function save_model_package(mesh, physics_config::Dict, ic::Vector, path::String)
    if !isdir(path)
        mkpath(path)
    end

    # Save a metadata file describing the mesh type.
    mesh_meta = Dict(
        "schema_version" => MODEL_PACKAGE_SCHEMA_VERSION,
        "type" => string(typeof(mesh)),
    )
    open(joinpath(path, "mesh_meta.toml"), "w") do io
        TOML.print(io, stringify_keys(mesh_meta))
    end

    # Save Physics Config
    open(joinpath(path, "physics.toml"), "w") do io
        TOML.print(io, stringify_keys(physics_config))
    end

    # Save IC
    open(joinpath(path, "ic.dat"), "w") do io
        writedlm(io, ic)
    end

    @info "Model package saved" path
    return nothing
end

"""
    load_model_package(path)

Load a simulation setup.
Returns (mesh_meta, physics_config, ic).
Mesh reconstruction is up to the user based on meta/config.
"""
function load_model_package(path)
    mesh_meta = TOML.parsefile(joinpath(path, "mesh_meta.toml"))
    physics_config = TOML.parsefile(joinpath(path, "physics.toml"))
    ic = vec(readdlm(joinpath(path, "ic.dat")))
    return mesh_meta, physics_config, ic
end

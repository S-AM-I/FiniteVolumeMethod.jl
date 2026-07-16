module RepoProjectIntegrity

using TOML

const DEFAULT_IGNORED_DIRS = Set(["build", ".git", ".github"])
const KNOWN_STDLIBS = Set(
    Symbol.(
        [
            "Artifacts",
            "Base64",
            "Dates",
            "DelimitedFiles",
            "Distributed",
            "Downloads",
            "InteractiveUtils",
            "LinearAlgebra",
            "Logging",
            "Markdown",
            "Mmap",
            "Pkg",
            "Printf",
            "Profile",
            "Random",
            "Serialization",
            "SparseArrays",
            "Statistics",
            "TOML",
            "Tar",
            "Test",
            "UUIDs",
        ],
    ),
)

function declared_deps(project_toml::AbstractString)
    raw = TOML.parsefile(project_toml)
    return Set(Symbol.(keys(get(raw, "deps", Dict{String, Any}()))))
end

function is_stdlib(module_name::Symbol)
    return module_name in KNOWN_STDLIBS
end

function check_project_integrity(
        project_toml::AbstractString,
        scan_roots::Vector{String};
        repo_root::AbstractString,
        local_modules::Vector{Symbol} = Symbol[],
        ignored_dirs::Set{String} = DEFAULT_IGNORED_DIRS,
    )
    declared = declared_deps(project_toml)
    local_module_set = Set(local_modules)
    usage = Dict{Symbol, Vector{String}}()

    for rel_root in scan_roots
        root = joinpath(repo_root, rel_root)
        isdir(root) || continue
        for (dir, _, files) in walkdir(root)
            any(ignored -> occursin("/" * ignored * "/", dir * "/"), ignored_dirs) && continue
            for file in files
                endswith(file, ".jl") || continue
                path = joinpath(dir, file)
                relpath_ = relpath(path, repo_root)
                for module_name in extract_used_modules(path)
                    module_name in local_module_set && continue
                    is_stdlib(module_name) && continue
                    refs = get!(usage, module_name, String[])
                    relpath_ in refs || push!(refs, relpath_)
                end
            end
        end
    end

    missing = Dict(
        module_name => sort!(copy(refs))
            for (module_name, refs) in usage if module_name ∉ declared
    )

    return (
        declared = declared,
        used = usage,
        missing = missing,
    )
end

function extract_used_modules(path::AbstractString)
    modules = Symbol[]
    for raw_line in eachline(path)
        line = replace(raw_line, r"#.*$" => "")
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "using ") || startswith(stripped, "import ") || continue
        clause = startswith(stripped, "using ") ? stripped[7:end] : stripped[8:end]
        parts = occursin(":", clause) ? [split(clause, ':')[1]] : split(clause, ',')
        for part in parts
            token = strip(part)
            isempty(token) && continue
            startswith(token, ".") && continue
            token = split(token, '.')[1]
            # `import Foo as Bar` binds Foo under an alias; the package is Foo.
            token = strip(split(token, r"\s+as\s+")[1])
            isempty(token) && continue
            push!(modules, Symbol(token))
        end
    end
    return unique(modules)
end

end

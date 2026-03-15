using Dates

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const TEST_PROJECT = joinpath(REPO_ROOT, "test")
const DOCS_PROJECT = joinpath(REPO_ROOT, "docs")
const JULIA_BIN = joinpath(Sys.BINDIR, Base.julia_exename())
const CI_WRITABLE_DEPOT = joinpath(tempdir(), "fvm-ci-lane-depot")
const BASE_DEPOT_PATH = get(ENV, "JULIA_DEPOT_PATH", joinpath(homedir(), ".julia"))
const DEPOT_PATH_SEPARATOR = Sys.iswindows() ? ';' : ':'
const CI_DEPOT_PATH = string(CI_WRITABLE_DEPOT, DEPOT_PATH_SEPARATOR, BASE_DEPOT_PATH)

function parse_args(args)
    isempty(args) && error("Usage: julia --project=. scripts/run_ci_lane.jl <fast-api-interop|scientific-smoke|full-evidence|performance|release-audit> [--output-root=/tmp/path]")
    lane = Symbol(replace(first(args), "-" => "_"))
    output_root = joinpath(tempdir(), "fvm-release-audit")
    for arg in Iterators.drop(args, 1)
        if startswith(arg, "--output-root=")
            output_root = normpath(split(arg, "=", limit = 2)[2])
        else
            error("Unknown argument: $arg")
        end
    end
    return (; lane, output_root)
end

timestamp() = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")

function log_step(message::AbstractString)
    println("[$(timestamp())] $message")
    return nothing
end

function ci_env()
    return merge(
        copy(ENV),
        Dict(
            "JULIA_DEPOT_PATH" => CI_DEPOT_PATH,
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            "JULIA_PKG_OFFLINE" => "true",
        ),
    )
end

function julia_expr(project::AbstractString, expr::AbstractString; extra_env = Dict{String, String}())
    mkpath(CI_WRITABLE_DEPOT)
    cmd = Cmd(`$JULIA_BIN --project=$project -e $expr`; dir = REPO_ROOT)
    return setenv(cmd, merge(ci_env(), extra_env))
end

function run_step(label::AbstractString, cmd::Cmd)
    log_step(label)
    run(cmd)
    return nothing
end

function instantiate_test_project()
    return run_step(
        "Instantiate test project",
        julia_expr(
            TEST_PROJECT,
            "using Pkg; Pkg.develop(PackageSpec(path=$(repr(REPO_ROOT)))); Pkg.resolve(); Pkg.instantiate()",
        ),
    )
end

function instantiate_docs_project()
    return run_step(
        "Instantiate docs project",
        julia_expr(
            DOCS_PROJECT,
            "using Pkg; Pkg.develop(PackageSpec(path=$(repr(REPO_ROOT)))); Pkg.resolve(); Pkg.instantiate()",
        ),
    )
end

function run_test_file(filename::AbstractString; extra_env = Dict{String, String}())
    path = joinpath(REPO_ROOT, "test", filename)
    run_step("Run test/$filename", julia_expr(TEST_PROJECT, "include($(repr(path)))"; extra_env))
    return nothing
end

function run_docs_file(filename::AbstractString; extra_env = Dict{String, String}())
    path = joinpath(REPO_ROOT, "docs", filename)
    run_step("Run docs/$filename", julia_expr(DOCS_PROJECT, "include($(repr(path)))"; extra_env))
    return nothing
end

function run_fast_api_interop_lane()
    instantiate_test_project()
    for file in (
            "environment_integrity.jl",
            "sciml_audit.jl",
            "sciml_contract.jl",
            "test_remake.jl",
            "semidiscrete.jl",
            "semidiscrete_mhd.jl",
            "semidiscrete_imex.jl",
            "semidiscrete_amr.jl",
            "repository_governance.jl",
        )
        run_test_file(file)
    end
    return nothing
end

function run_scientific_smoke_lane()
    instantiate_test_project()
    run_test_file("scientific_smoke.jl")
    return nothing
end

function run_full_evidence_lane()
    instantiate_test_project()
    run_test_file("scientific_evidence.jl")
    return nothing
end

function run_performance_lane()
    instantiate_test_project()
    run_test_file("performance_baselines.jl")
    return nothing
end

function run_release_audit_lane(output_root::AbstractString)
    instantiate_test_project()
    run_test_file("environment_integrity.jl")
    run_test_file("repository_governance.jl")
    instantiate_docs_project()
    run_docs_file("environment_integrity.jl")
    run_test_file(
        "release_audit.jl";
        extra_env = Dict(
            "FVM_RELEASE_AUDIT_OUTPUT_ROOT" => output_root,
            "FVM_RELEASE_AUDIT_RERUN" => "true",
        ),
    )
    return nothing
end

function run_lane(lane::Symbol; output_root::AbstractString)
    log_step("Starting CI lane $(replace(string(lane), "_" => "-"))")
    if lane == :fast_api_interop
        run_fast_api_interop_lane()
    elseif lane == :scientific_smoke
        run_scientific_smoke_lane()
    elseif lane == :full_evidence
        run_full_evidence_lane()
    elseif lane == :performance
        run_performance_lane()
    elseif lane == :release_audit
        run_release_audit_lane(output_root)
    else
        error("Unknown lane: $(replace(string(lane), "_" => "-"))")
    end
    log_step("Completed CI lane $(replace(string(lane), "_" => "-"))")
    return nothing
end

options = parse_args(ARGS)
run_lane(options.lane; output_root = options.output_root)

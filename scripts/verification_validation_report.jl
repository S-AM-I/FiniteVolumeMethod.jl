using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function parse_args(args)
    output_path = joinpath(REPO_ROOT, "validation", "reports", "validation_report.md")
    summary_dir = joinpath(REPO_ROOT, "validation", "reports", "summaries")
    bundle_dir = joinpath(REPO_ROOT, "validation", "reproduction_bundles")
    rerun_evidence = true
    for arg in args
        if startswith(arg, "--output=")
            output_path = normpath(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--summary-dir=")
            summary_dir = normpath(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--bundle-dir=")
            bundle_dir = normpath(split(arg, "=", limit = 2)[2])
        elseif arg == "--reuse-summaries"
            rerun_evidence = false
        else
            error("Unknown argument: $arg")
        end
    end
    return (; output_path, summary_dir, bundle_dir, rerun_evidence)
end

options = parse_args(ARGS)
mkpath(options.summary_dir)

if options.rerun_evidence
    Pkg.activate(joinpath(REPO_ROOT, "test"))
    Pkg.develop(PackageSpec(path = REPO_ROOT))
    Pkg.instantiate()
end

include(joinpath(REPO_ROOT, "validation", "manifest.jl"))
include(joinpath(REPO_ROOT, "validation", "evidence_runner.jl"))
include(joinpath(REPO_ROOT, "validation", "generate_report.jl"))
using .RepoEvidenceRunner
using .RepoValidationManifest
using .ValidationReport

manifest = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))
if options.rerun_evidence
    RepoEvidenceRunner.run_evidence_suite(
        manifest;
        repo_root = REPO_ROOT,
        output_dir = options.summary_dir,
    )
end

ValidationReport.generate(
    joinpath(REPO_ROOT, "validation", "manifest.toml"),
    options.output_path;
    summary_dir = options.summary_dir,
    bundle_dir = options.bundle_dir,
)

println("Validation report written to $(options.output_path)")

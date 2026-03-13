module RepoEvidenceRunner

using Dates
using Test
using TOML

include(joinpath(@__DIR__, "manifest.jl"))
using .RepoValidationManifest

const DEFAULT_OUTPUT_DIR = joinpath(tempdir(), "fvm-evidence-summaries")

function run_evidence_entry(entry; repo_root::AbstractString, output_dir::AbstractString = DEFAULT_OUTPUT_DIR)
    entrypoint = joinpath(repo_root, entry.entrypoint)
    mod = @eval module $(gensym(:EvidenceModule)) end
    Base.include(mod, joinpath(@__DIR__, "evidence_capture.jl"))

    started_at = now()
    include_error = nothing
    testset = @testset verbose = true "Evidence: $(entry.id)" begin
        try
            Base.include(mod, entrypoint)
        catch err
            include_error = sprint(showerror, err)
            @test false
        end
    end
    finished_at = now()

    counts = Test.get_test_counts(testset)
    records = Base.invokelatest(() -> getfield(mod, :evidence_results)())
    summary_missing = entry.summary_required && isempty(records)
    status = counts.fails == 0 && counts.errors == 0 && isnothing(include_error) && !summary_missing

    summary = Dict{String, Any}(
        "id" => entry.id,
        "feature" => string(entry.feature),
        "path" => entry.path,
        "entrypoint" => entry.entrypoint,
        "ladder_stage" => string(entry.ladder_stage),
        "runtime_tier" => string(entry.runtime_tier),
        "category" => string(entry.category),
        "reference_kind" => string(entry.reference_kind),
        "reference_source" => entry.reference_source,
        "metric" => entry.metric,
        "acceptance" => entry.acceptance,
        "solver_family" => string(entry.solver_family),
        "precision_policy" => string(entry.precision_policy),
        "random_seed_policy" => string(entry.random_seed_policy),
        "expected_artifacts" => copy(entry.expected_artifacts),
        "summary_required" => entry.summary_required,
        "status" => status ? "pass" : "fail",
        "recorded_result_count" => length(records),
        "recorded_results" => copy(records),
        "counts" => Dict(
            "passes" => counts.passes,
            "fails" => counts.fails,
            "errors" => counts.errors,
            "broken" => counts.broken,
        ),
        "started_at" => Dates.format(started_at, dateformat"yyyy-mm-ddTHH:MM:SS"),
        "finished_at" => Dates.format(finished_at, dateformat"yyyy-mm-ddTHH:MM:SS"),
    )
    if !isnothing(include_error)
        summary["include_error"] = include_error
    end
    if summary_missing
        summary["summary_error"] = "summary_required evidence entry did not record any evidence result"
    end

    summary_path = evidence_summary_path(entry.id; output_dir)
    mkpath(dirname(summary_path))
    open(summary_path, "w") do io
        TOML.print(io, summary)
    end
    summary["summary_path"] = summary_path
    return summary
end

function run_evidence_suite(
        manifest;
        repo_root::AbstractString,
        output_dir::AbstractString = DEFAULT_OUTPUT_DIR,
    )
    mkpath(output_dir)
    return [
        run_evidence_entry(entry; repo_root, output_dir) for entry in manifest.scientific_evidence
    ]
end

evidence_summary_path(id::AbstractString; output_dir::AbstractString = DEFAULT_OUTPUT_DIR) =
    joinpath(output_dir, "$(id).toml")

end

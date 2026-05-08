#!/usr/bin/env julia
#
# Allocation-budget regression for FiniteVolumeMethod.
#
# Same harness shape as CRUDApplication and NuclearWaterChemistry. See
# CRUDApplication.jl/benchmarks/runbenchmarks.jl for the design.
#
# Targets focus on FVM's two stable claim-bearing solver families
# (parabolic + hyperbolic). Tightening the alloc envelope on these is
# the highest-leverage way to keep the package competitive with C++/
# Kokkos on CPU per the IJHPCA 2025 reference (PLAN.md §2).

using Pkg
Pkg.activate(@__DIR__)

using FiniteVolumeMethod
using Printf
using TOML

const BASELINE_PATH = joinpath(@__DIR__, "baseline.toml")
const TOLERANCE = 0.05

# ── Benchmark targets ────────────────────────────────────────────────────────
# Each measurement is named so the diff in CI is human-readable.

results = Dict{String, Int}()

# Parabolic baseline — currently a placeholder that exercises the public
# load. Replace with concrete `solve(::CylindricalDiffusion1DProblem)` /
# vertex-centred parabolic step when the harness is wired into FVM's
# scientific-smoke stack.
let
    a = @allocated FiniteVolumeMethod.eval(:(__bench_noop = nothing))
    results["parabolic/load_smoke"] = a
end

# ── Mode dispatch (identical to CRUDApplication's) ───────────────────────────

mode = isempty(ARGS) ? "compare" : ARGS[1]

if mode in ("--print", "print")
    println("Allocation measurements (bytes):")
    for k in sort(collect(keys(results)))
        @printf("  %-40s %12d\n", k, results[k])
    end
    exit(0)
elseif mode in ("--baseline", "baseline")
    open(BASELINE_PATH, "w") do io
        println(io, "# benchmarks/baseline.toml — pinned allocation counts in bytes.")
        println(io, "# Regenerate: julia --project=benchmarks benchmarks/runbenchmarks.jl --baseline")
        println(io, "# Tolerance: ", round(Int, TOLERANCE * 100), "%.")
        println(io)
        println(io, "[allocations]")
        for k in sort(collect(keys(results)))
            println(io, "\"", k, "\" = ", results[k])
        end
    end
    println("Wrote baseline to $BASELINE_PATH.")
    exit(0)
elseif mode in ("--compare", "compare")
    isfile(BASELINE_PATH) ||
        error("baseline file not found: $BASELINE_PATH. Run with --baseline first.")
    baseline = TOML.parsefile(BASELINE_PATH)
    haskey(baseline, "allocations") ||
        error("baseline missing [allocations]")
    pinned = Dict{String, Int}(k => Int(v) for (k, v) in baseline["allocations"])
    failures = String[]
    println("Allocation comparison (current / baseline, ratio):")
    for k in sort(collect(keys(results)))
        cur = results[k]
        if !haskey(pinned, k)
            @printf("  %-40s %12d  (NEW)\n", k, cur); continue
        end
        b = pinned[k]; ratio = b == 0 ? Inf : cur / b
        marker = ratio > 1.0 + TOLERANCE ? "FAIL" : "ok  "
        @printf("  %-40s %12d  (baseline %12d, ratio %.3f) %s\n", k, cur, b, ratio, marker)
        ratio > 1.0 + TOLERANCE && push!(failures,
            @sprintf("%s grew %.1f%% over baseline", k, 100 * (ratio - 1)))
    end
    if isempty(failures)
        println("All allocations within ", round(Int, 100 * TOLERANCE), "% of baseline.")
        exit(0)
    else
        println(stderr, "\nAllocation regression(s):")
        foreach(f -> println(stderr, "  ", f), failures)
        println(stderr, "\nIf intentional, regenerate baseline with --baseline.")
        exit(1)
    end
else
    println(stderr, "unknown mode: $mode")
    exit(2)
end

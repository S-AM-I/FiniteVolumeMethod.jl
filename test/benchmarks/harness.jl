# test/benchmarks/harness.jl — Published-benchmark caching harness (v3.1 Agent E)
#
# The benchmarks in this directory reproduce published reference data
# (Ghia, Moser, De Vahl Davis, Martin-Moyce, Sod). Each one is slow
# enough that re-running on every `using Pkg; Pkg.test()` would be
# prohibitive. Instead we:
#
#   1. Gate the whole suite on `ENV["FVM_RUN_BENCHMARKS"] == "true"`.
#      When unset/false, `@benchmark_testset` records a skipped summary
#      and returns without executing the body.
#
#   2. Within the gated path, hash the source files listed in
#      `sources_of(:tag)` (the solver files that actually affect the
#      benchmark). Compare against `test/benchmarks/.cache/<name>.sha`.
#      If the hash matches, the benchmark is a cached pass and returns
#      without executing the body. On mismatch, run the benchmark and
#      re-write the cache on success.
#
#   3. Exposes two macros:
#
#        @benchmark_testset "ghia_re400" [sources = :incompressible] begin
#            result = solve_problem(...)
#            @benchmark_assert isapprox(u_center, u_ref; rtol = 0.05)
#        end
#
#      `@benchmark_assert` is a thin wrapper over `@test` so assertions
#      still surface in the standard Test summary when the benchmark
#      runs.
#
# Cache file format (newline-separated):
#   line 1: SHA-256 of the concatenated source files
#   line 2: UTC ISO 8601 timestamp of the passing run
#   line 3: Julia version string
#   line 4: git short SHA if available, else "unknown"
#
# The cache lives under `.cache/` which is gitignored via
# `test/benchmarks/.gitignore` — benchmark provenance is local, not
# committed.

using SHA
using Dates: now, UTC, DateTime
using Test

const BENCHMARK_CACHE_DIR = joinpath(@__DIR__, ".cache")
const BENCHMARK_REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))

# Per-process benchmark status registry. Every `@benchmark_testset`
# records exactly one terminal status per benchmark name:
#
#   "passed"   — the body ran and every physics assertion passed
#   "failed"   — the body ran and at least one assertion failed/errored
#   "deferred" — the body bailed out via `mark_deferred_compute`
#                (recorded as `@test_broken`, never as a pass)
#   "cached"   — source-hash cache hit; physics assertions NOT re-executed
#   "skipped"  — FVM_RUN_BENCHMARKS unset
#
# `write_benchmark_summary` serialises this to a machine-readable TOML
# file so CI can assert that the advertised number of benchmarks
# actually executed their physics assertions (status == "passed").
# Guarded so repeated `include("harness.jl")` from each benchmark file
# does not reset the registry mid-suite.
if !@isdefined(BENCHMARK_RESULTS)
    const BENCHMARK_RESULTS = Dict{String, String}()
end

"""
    benchmarks_enabled() -> Bool

Check the `FVM_RUN_BENCHMARKS` environment flag. Benchmarks only run
when this is explicitly set to `"true"` (case-insensitive). Default
is `false` to keep `Pkg.test()` fast.
"""
function benchmarks_enabled()
    v = get(ENV, "FVM_RUN_BENCHMARKS", "false")
    return lowercase(strip(v)) in ("true", "1", "yes", "on")
end

"""
    sources_of(tag::Symbol) -> Vector{String}

Map a benchmark source-tag to the list of repo-relative file paths
whose content determines whether the benchmark needs to re-run. Each
benchmark declares which tag(s) it depends on. When any listed file's
content changes, the cache invalidates.

Tags correspond to solver phases in `src/` — benchmarks dependent on
a solver only invalidate when that solver's source changes, not when
unrelated files are touched.
"""
function sources_of(tag::Symbol)
    if tag === :incompressible
        return [
            "src/incompressible/simple.jl",
            "src/incompressible/momentum.jl",
            "src/incompressible/pressure.jl",
            "src/incompressible/correction.jl",
            "src/incompressible/residuals.jl",
            "src/collocated/gradient.jl",
            "src/collocated/laplacian.jl",
            "src/collocated/interpolation.jl",
        ]
    elseif tag === :turbulence
        return [
            "src/turbulence/solvers.jl",
            "src/turbulence/k_epsilon_rans.jl",
            "src/turbulence/wall_functions.jl",
            "src/turbulence/interface.jl",
            "src/incompressible/simple.jl",
        ]
    elseif tag === :thermal
        return [
            "src/thermal/solvers.jl",
            "src/thermal/energy_equation.jl",
            "src/thermal/buoyancy.jl",
            "src/incompressible/simple.jl",
            "src/collocated/gradient.jl",
            "src/collocated/laplacian.jl",
        ]
    elseif tag === :multiphase
        return [
            "src/multiphase/solvers.jl",
            "src/multiphase/alpha_transport.jl",
            "src/multiphase/boundedness.jl",
            "src/multiphase/mixture.jl",
            "src/collocated/gradient.jl",
            "src/collocated/interpolation.jl",
        ]
    elseif tag === :hyperbolic
        return [
            "src/hyperbolic/hyperbolic_solve.jl",
            "src/hyperbolic/hyperbolic_problem.jl",
            "src/hyperbolic/hllc_solver.jl",
            "src/hyperbolic/reconstruction.jl",
            "src/hyperbolic/euler.jl",
            "src/core/cache.jl",
        ]
    else
        error("Unknown benchmark source tag: $tag")
    end
end

"""
    _resolve_sources(tags) -> Vector{String}

Flatten a (possibly vector) source tag(s) into the deduplicated list
of existing file paths (repo-relative). Missing files are skipped with
a warning — this keeps the harness robust across branches where some
source files are temporarily absent.
"""
function _resolve_sources(tags)
    tag_list = tags isa Symbol ? [tags] : collect(tags)
    paths = String[]
    for tag in tag_list
        append!(paths, sources_of(tag))
    end
    unique!(paths)
    existing = String[]
    for p in paths
        full = joinpath(BENCHMARK_REPO_ROOT, p)
        if isfile(full)
            push!(existing, full)
        else
            @warn "Benchmark source file missing; excluding from hash" path = p
        end
    end
    return existing
end

"""
    _hash_sources(paths) -> String

SHA-256 of the concatenated file contents, with each file's relative
path embedded as a header so that renaming a file invalidates the
cache even if the content is identical.
"""
function _hash_sources(paths::Vector{String})
    ctx = SHA2_256_CTX()
    for p in paths
        rel = relpath(p, BENCHMARK_REPO_ROOT)
        SHA.update!(ctx, Vector{UInt8}("path=$rel\n"))
        open(p, "r") do io
            SHA.update!(ctx, read(io))
        end
        SHA.update!(ctx, Vector{UInt8}("\n---\n"))
    end
    return bytes2hex(SHA.digest!(ctx))
end

"""
    _cache_path(name) -> String
"""
_cache_path(name::AbstractString) = joinpath(BENCHMARK_CACHE_DIR, name * ".sha")

"""
    _cache_hit(name, expected_hash) -> Bool

Return `true` iff the cache file exists and its first line matches
`expected_hash`. Any file-system error or mismatched hash returns
`false` so the benchmark re-runs.
"""
function _cache_hit(name::AbstractString, expected_hash::AbstractString)
    path = _cache_path(name)
    isfile(path) || return false
    try
        open(path, "r") do io
            cached = strip(readline(io))
            return cached == expected_hash
        end
    catch
        return false
    end
end

"""
    _cache_write(name, hash)

Record a passing benchmark run: store the hash, timestamp, Julia
version, and git short-SHA (if available) for future forensics.
"""
function _cache_write(name::AbstractString, hash::AbstractString)
    isdir(BENCHMARK_CACHE_DIR) || mkpath(BENCHMARK_CACHE_DIR)
    git_sha = try
        readchomp(`git -C $(BENCHMARK_REPO_ROOT) rev-parse --short HEAD`)
    catch
        "unknown"
    end
    timestamp = string(now(UTC))
    open(_cache_path(name), "w") do io
        println(io, hash)
        println(io, timestamp)
        println(io, string(VERSION))
        println(io, git_sha)
    end
    return nothing
end

"""
    @benchmark_testset name [sources = :tag] body

Wrap a benchmark in three layers of gating:

  1. `ENV["FVM_RUN_BENCHMARKS"]` disabled ⇒ skipped-summary, no
     assertions counted.
  2. Source hash matches last-pass cache ⇒ single passing `@test true`
     recording "cached pass (hash HHHH)".
  3. Otherwise run the body inside `@testset`. On success (all
     assertions pass) refresh the cache so the next invocation hits
     the fast path.

The `sources` keyword defaults to `:incompressible` — override per
benchmark, e.g. `sources = :thermal` for the Rayleigh-Bénard case.

Assertions inside the body should be written with `@benchmark_assert`
(currently an alias for `@test`) so future extensions — e.g.
recording reference-value-with-measured-value provenance — can
intercept them without changing every benchmark.
"""
macro benchmark_testset(name, args...)
    # Parse keyword args. Valid form:
    #   @benchmark_testset "x" [sources = :incompressible] body
    # Trailing block is required; everything before it is kwargs.
    body = args[end]
    kwargs = args[1:(end - 1)]

    sources_expr = :(:incompressible)
    for kw in kwargs
        if kw isa Expr && kw.head === :(=) && kw.args[1] === :sources
            sources_expr = kw.args[2]
        else
            error("@benchmark_testset: unexpected kwarg $(kw)")
        end
    end

    # Wrap the body in a closure so `return` inside the user-facing
    # benchmark (used to bail out of a deferred-compute case) exits
    # the closure, not the enclosing file's module-level scope. The
    # closure is called from inside the `@testset` so assertions still
    # land in the right testset.
    return quote
        local _bench_name = $(esc(name))
        if !benchmarks_enabled()
            @info "benchmark skipped (FVM_RUN_BENCHMARKS unset)" name = _bench_name
            BENCHMARK_RESULTS[_bench_name] = "skipped"
        else
            local _sources_tag = $(esc(sources_expr))
            local _paths = _resolve_sources(_sources_tag)
            local _hash = _hash_sources(_paths)
            if _cache_hit(_bench_name, _hash)
                @info "benchmark cached pass" name = _bench_name hash = _hash[1:12]
                BENCHMARK_RESULTS[_bench_name] = "cached"
                @testset "$(_bench_name) [cached]" begin
                    @test true
                end
            else
                @info "benchmark running" name = _bench_name hash = _hash[1:12]
                delete!(BENCHMARK_RESULTS, _bench_name)
                local _body_fn = () -> $(esc(body))
                local _ts = @testset "$(_bench_name)" begin
                    _body_fn()
                end
                # Refresh cache only if every assertion passed AND the
                # benchmark was not deferred. A deferred benchmark never
                # executed its physics assertions, so it must not become
                # a cached "pass" on the next invocation.
                local _failed = _ts.anynonpass
                local _deferred = get(BENCHMARK_RESULTS, _bench_name, "") == "deferred"
                if _deferred
                    @warn "benchmark deferred; cache not refreshed" name = _bench_name
                elseif !_failed
                    BENCHMARK_RESULTS[_bench_name] = "passed"
                    _cache_write(_bench_name, _hash)
                    @info "benchmark cache refreshed" name = _bench_name
                else
                    BENCHMARK_RESULTS[_bench_name] = "failed"
                    @warn "benchmark failed; cache not refreshed" name = _bench_name
                end
            end
        end
    end
end

"""
    @benchmark_assert expr

Inside a `@benchmark_testset` body. Currently an alias for `@test`,
but reserved for future use (e.g. emitting a JSON record of the
measured vs reference value pair for the validation ledger).
"""
macro benchmark_assert(expr)
    return quote
        @test $(esc(expr))
    end
end

"""
    mark_deferred_compute(name, reason)

Called from inside a `@benchmark_testset` body when the underlying
solver can't reach the published tolerance on user compute within
the per-benchmark wall-clock budget. Records a **broken** test
(`@test_broken false`) so the deferred benchmark is visible in the
Test summary as broken — never as a pass — and marks the benchmark
`"deferred"` in `BENCHMARK_RESULTS` so `write_benchmark_summary` /
the CI gate can count it separately from executed benchmarks.

Example:
    if !converged
        mark_deferred_compute("ghia_re400",
            "SIMPLE plateau on 64x64 > 30 min M3 Max")
        return
    end
"""
function mark_deferred_compute(name::AbstractString, reason::AbstractString)
    @warn "benchmark DEFERRED_COMPUTE (recorded as broken, not passing)" name = name reason = reason
    BENCHMARK_RESULTS[name] = "deferred"
    @test_broken false
    return nothing
end

"""
    write_benchmark_summary(path) -> NamedTuple

Serialise `BENCHMARK_RESULTS` to `path` as a small TOML file with a
`[counts]` table (`passed`, `failed`, `deferred`, `cached`, `skipped`)
and a `[results]` table mapping benchmark name → status. Returns the
counts as a NamedTuple so callers (the CI published-benchmarks job)
can assert on them, e.g.

    counts = write_benchmark_summary("benchmark_summary.toml")
    counts.passed >= 5 || error("only \$(counts.passed) benchmarks executed physics assertions")
"""
function write_benchmark_summary(path::AbstractString)
    statuses = ("passed", "failed", "deferred", "cached", "skipped")
    counts = NamedTuple{Symbol.(Tuple(statuses))}(
        Tuple(count(==(s), values(BENCHMARK_RESULTS)) for s in statuses)
    )
    open(path, "w") do io
        println(io, "# Auto-generated by test/benchmarks/harness.jl")
        println(io, "[counts]")
        for s in statuses
            println(io, "$(s) = $(getproperty(counts, Symbol(s)))")
        end
        println(io)
        println(io, "[results]")
        for name in sort!(collect(keys(BENCHMARK_RESULTS)))
            println(io, "$(name) = \"$(BENCHMARK_RESULTS[name])\"")
        end
    end
    return counts
end

export @benchmark_testset, @benchmark_assert,
    benchmarks_enabled, mark_deferred_compute,
    write_benchmark_summary, BENCHMARK_RESULTS,
    sources_of

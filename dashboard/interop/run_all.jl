# ============================================================
# run_all.jl — Master script for interop examples
#
# Usage:
#   cd interop/
#   julia --project=. run_all.jl
#
# Runs examples 01-08 sequentially (09 is a live WebSocket demo
# that requires manual dashboard interaction).
# ============================================================

using Pkg
Pkg.instantiate()

const EXAMPLES_DIR = joinpath(@__DIR__, "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "output")

# Ensure output directory exists
mkpath(OUTPUT_DIR)

# Scripts to run (09 excluded — requires manual WebSocket interaction)
const SCRIPTS = [
    "01_sod_1d.jl",
    "02_euler_2d.jl",
    "03_shallow_water_1d.jl",
    "04_reactive_euler_1d.jl",
    "05_mhd_2d.jl",
    "06_euler_3d.jl",
    "07_convergence_study.jl",
    "08_parabolic_diffusion.jl",
]

# Expected output files
const EXPECTED_OUTPUTS = [
    "01_sod_1d.fvm-session.json",
    "02_euler_2d.fvm-session.json",
    "03_shallow_water_1d.fvm-session.json",
    "04_reactive_euler_1d.fvm-session.json",
    "05_mhd_2d.fvm-session.json",
    "06_euler_3d.fvm-session.json",
    "07_convergence_study.fvm-session.json",
    "08_parabolic_diffusion.fvm-session.json",
]

println("="^60)
println("  FVM Interop — Running all examples")
println("="^60)

results = Dict{String, Symbol}()

for script in SCRIPTS
    path = joinpath(EXAMPLES_DIR, script)
    println("\n▶ Running $script ...")
    try
        include(path)
        results[script] = :ok
        println("  ✓ $script completed")
    catch e
        results[script] = :failed
        println("  ✗ $script FAILED: $e")
    end
end

# Validate outputs
println("\n" * "="^60)
println("  Output Validation")
println("="^60)

using JSON3

all_valid = true
for outfile in EXPECTED_OUTPUTS
    path = joinpath(OUTPUT_DIR, outfile)
    if !isfile(path)
        println("  ✗ MISSING: $outfile")
        all_valid = false
        continue
    end

    # Validate JSON structure
    try
        data = open(path, "r") do io
            JSON3.read(io, Dict)
        end

        has_variables = haskey(data, "variables")
        has_mesh = haskey(data, "mesh")
        has_snapshots = haskey(data, "snapshots")

        if has_variables && has_mesh && has_snapshots
            n_snaps = length(data["snapshots"])
            println("  ✓ $outfile — $(n_snaps) snapshots, valid structure")
        else
            missing_keys = String[]
            has_variables || push!(missing_keys, "variables")
            has_mesh || push!(missing_keys, "mesh")
            has_snapshots || push!(missing_keys, "snapshots")
            println("  ✗ $outfile — missing keys: $(join(missing_keys, ", "))")
            all_valid = false
        end
    catch e
        println("  ✗ $outfile — invalid JSON: $e")
        all_valid = false
    end
end

# Summary
println("\n" * "="^60)
println("  Summary")
println("="^60)
n_ok = count(v -> v == :ok, values(results))
n_fail = count(v -> v == :failed, values(results))
println("  Scripts: $n_ok passed, $n_fail failed out of $(length(SCRIPTS))")
println("  Outputs: $(all_valid ? "all valid" : "some invalid — see above")")
println("  WebSocket demo (09) must be run manually.")
println("="^60)

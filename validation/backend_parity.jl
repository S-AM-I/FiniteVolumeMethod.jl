module RepoBackendParity

using CUDA
using FiniteVolumeMethod
using StaticArrays

function run_suite()
    return [_cuda_hyperbolic_2d_parity()]
end

function summarize_suite(results = run_suite())
    counts = Dict(:pass => 0, :fail => 0, :not_run => 0)
    for result in results
        counts[result.status] = get(counts, result.status, 0) + 1
    end
    overall_status = counts[:fail] > 0 ? :fail : (counts[:pass] > 0 ? :pass : :not_run)
    return (
        status = overall_status,
        counts = counts,
        results = results,
    )
end

function _cuda_hyperbolic_2d_parity()
    if !CUDA.functional()
        return (
            id = "cuda_hyperbolic_2d_parity",
            feature = :hyperbolic,
            status = :not_run,
            backend = "cuda",
            rationale = "CUDA.jl is loaded but not functional on this machine.",
            metrics = Dict{String, Float64}(),
            failures = String[],
        )
    end

    prob = _supported_cuda_problem()
    coords_cpu, U_cpu, t_cpu = FiniteVolumeMethod.solve_hyperbolic(prob; method = :ssprk3, backend = CPUBackend())
    coords_gpu, U_gpu, t_gpu = FiniteVolumeMethod.solve_hyperbolic(prob; method = :ssprk3, backend = CUDASolverBackend())

    max_abs_diff = maximum(maximum(abs.(U_cpu[ix, iy] - U_gpu[ix, iy])) for iy in axes(U_cpu, 2), ix in axes(U_cpu, 1))
    time_diff = abs(t_cpu - t_gpu)
    failures = String[]
    time_diff < 1.0e-12 || push!(failures, "final-time difference $time_diff exceeds tolerance 1e-12")
    max_abs_diff < 5.0e-9 || push!(failures, "maximum conserved-state difference $max_abs_diff exceeds tolerance 5e-9")
    coords_cpu == coords_gpu || push!(failures, "CPU and CUDA coordinate grids differ")

    return (
        id = "cuda_hyperbolic_2d_parity",
        feature = :hyperbolic,
        status = isempty(failures) ? :pass : :fail,
        backend = "cuda",
        rationale = "2D Euler hyperbolic solve on the supported CUDA extension path matches the CPU reference solve.",
        metrics = Dict(
            "max_abs_diff" => max_abs_diff,
            "final_time_diff" => time_diff,
            "nx" => Float64(prob.mesh.nx),
            "ny" => Float64(prob.mesh.ny),
        ),
        failures = failures,
    )
end

function _supported_cuda_problem()
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 24, 24)

    function ic(x, y)
        if x < 0.5 && y < 0.5
            return SVector(1.0, 0.1, 0.0, 1.0)
        elseif x >= 0.5 && y < 0.5
            return SVector(0.5313, 0.8276, 0.0, 0.4)
        elseif x < 0.5 && y >= 0.5
            return SVector(0.8, 0.0, 0.0, 0.7)
        else
            return SVector(0.5313, 0.0, 0.7276, 0.4)
        end
    end

    bc = TransmissiveBC()
    return HyperbolicProblem2D(
        law,
        mesh,
        HLLCSolver(),
        CellCenteredMUSCL(),
        bc,
        bc,
        bc,
        bc,
        ic;
        cfl = 0.35,
        final_time = 0.01,
    )
end

end

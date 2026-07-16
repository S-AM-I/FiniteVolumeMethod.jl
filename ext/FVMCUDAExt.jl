module FVMCUDAExt

using CUDA: CUDA, CuArray, @cuda, blockDim, blockIdx, threadIdx
using StaticArrays: SVector

using FiniteVolumeMethod

const FVM = FiniteVolumeMethod

FVM.to_backend(x::AbstractArray, backend::FVM.CUDASolverBackend) = CuArray(x)
# CUDA 6 no longer re-exports GPUArraysCore.AbstractGPUArray; dispatch on the
# concrete CuArray (consistent with the fold/unfold methods below).
FVM.to_host(x::CUDA.CuArray) = Array(x)

function FVM.supports_backend(prob::FVM.HyperbolicProblem2D, ::FVM.CUDASolverBackend)
    return prob.law isa FVM.EulerEquations{2} &&
        prob.riemann_solver isa Union{FVM.LaxFriedrichsSolver, FVM.HLLSolver, FVM.HLLCSolver} &&
        prob.reconstruction isa Union{FVM.NoReconstruction, FVM.CellCenteredMUSCL} &&
        _supported_bc_2d(prob.bc_left, prob.bc_right, prob.bc_bottom, prob.bc_top)
end

function FVM.backend_summary(backend::FVM.CUDASolverBackend)
    status = CUDA.functional() ? "ready" : "unavailable"
    return isnothing(backend.device) ?
        "CUDA backend ($status, default device)" :
        "CUDA backend ($status, device $(backend.device))"
end

@inline _supported_bc_2d(bcs...) = all(_supported_bc_2d, bcs)
@inline _supported_bc_2d(
    ::Union{
        FVM.TransmissiveBC,
        FVM.ReflectiveBC,
        FVM.InflowBC,
        FVM.DirichletHyperbolicBC,
        FVM.PeriodicHyperbolicBC,
    }
) = true
@inline _supported_bc_2d(::Any) = false

function FVM._solve_hyperbolic(
        prob::FVM.HyperbolicProblem2D, backend::FVM.CUDASolverBackend;
        method::Symbol = :ssprk3,
        parallel::Bool = false,
        callback::Union{Nothing, Function} = nothing,
        return_device_state::Bool = false,
    )
    parallel && error("The CUDA backend ignores `parallel`; select the backend instead of CPU threading.")
    callback === nothing || error("Callbacks are not supported on the CUDA backend yet.")
    CUDA.functional() || error("CUDA.jl is loaded but not functional on this machine.")
    FVM.supports_backend(prob, backend) || FVM._unsupported_backend("solve_hyperbolic(::HyperbolicProblem2D)", backend)

    if !isnothing(backend.device)
        CUDA.device!(backend.device)
    end

    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    FT = typeof(mesh.dx)

    U = CuArray(FVM.initialize_2d(prob))
    dU = similar(U)
    U1 = similar(U)
    U2 = similar(U)
    speed = CUDA.zeros(FT, nx * ny)
    zero_state = zero(eltype(U))

    _fill_array!(dU, zero_state)
    _fill_array!(U1, zero_state)
    _fill_array!(U2, zero_state)

    t = prob.initial_time

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = _compute_dt_cuda(prob, U, speed, t)
            dt <= zero(dt) && break
            _rhs_cuda!(dU, U, prob, t)
            _axpy_interior!(U, dU, dt, nx, ny)
            t += dt
        end
    elseif method == :ssprk3
        while t < prob.final_time - eps(typeof(t))
            dt = _compute_dt_cuda(prob, U, speed, t)
            dt <= zero(dt) && break

            _rhs_cuda!(dU, U, prob, t)
            _ssprk_stage1!(U1, U, dU, dt, nx, ny)

            _rhs_cuda!(dU, U1, prob, t + dt)
            _ssprk_stage2!(U2, U, U1, dU, dt, nx, ny)

            _rhs_cuda!(dU, U2, prob, t + 0.5 * dt)
            _ssprk_stage3!(U, U2, dU, dt, nx, ny)

            t += dt
        end
    else
        error("Unknown time integration method: $method. Use :euler or :ssprk3.")
    end

    coords = FVM._cell_center_coords_2d(mesh)
    U_interior = view(U, 3:(nx + 2), 3:(ny + 2))
    return coords, return_device_state ? U_interior : Array(U_interior), t
end

function _fill_array!(A, value)
    threads = 256
    blocks = cld(length(A), threads)
    @cuda threads = threads blocks = blocks fill_kernel!(A, value)
    return A
end

function _compute_dt_cuda(prob, U, speed, t)
    nx, ny = prob.mesh.nx, prob.mesh.ny
    threads = 256
    blocks = cld(length(speed), threads)
    @cuda threads = threads blocks = blocks speed_kernel!(speed, prob, U, nx, ny)
    max_speed = maximum(speed)
    dt = prob.cfl / max_speed
    if t + dt > prob.final_time
        dt = prob.final_time - t
    end
    return dt
end

function _rhs_cuda!(dU, U, prob, t)
    _apply_boundary_conditions_cuda!(U, prob, t)
    nx, ny = prob.mesh.nx, prob.mesh.ny
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    @cuda threads = threads blocks = blocks rhs_kernel_2d!(dU, U, prob, nx, ny)
    return dU
end

function _apply_boundary_conditions_cuda!(U, prob, t)
    nx, ny = prob.mesh.nx, prob.mesh.ny
    if prob.bc_left isa FVM.PeriodicHyperbolicBC && prob.bc_right isa FVM.PeriodicHyperbolicBC
        threads = 256
        blocks = cld(ny + 4, threads)
        @cuda threads = threads blocks = blocks periodic_x_kernel!(U, nx, ny)
    else
        _launch_vertical_bc!(left_bc_kernel!, U, prob.bc_left, prob.law, nx, ny, true)
        _launch_vertical_bc!(right_bc_kernel!, U, prob.bc_right, prob.law, nx, ny, false)
    end

    if prob.bc_bottom isa FVM.PeriodicHyperbolicBC && prob.bc_top isa FVM.PeriodicHyperbolicBC
        threads = 256
        blocks = cld(nx + 4, threads)
        @cuda threads = threads blocks = blocks periodic_y_kernel!(U, nx, ny)
    else
        _launch_horizontal_bc!(bottom_bc_kernel!, U, prob.bc_bottom, prob.law, nx, ny, true)
        _launch_horizontal_bc!(top_bc_kernel!, U, prob.bc_top, prob.law, nx, ny, false)
    end
    return U
end

function _launch_vertical_bc!(kernel, U, bc, law, nx, ny, is_left)
    threads = 256
    blocks = cld(ny + 4, threads)
    @cuda threads = threads blocks = blocks kernel(U, bc, law, nx, ny, is_left)
    return nothing
end

function _launch_horizontal_bc!(kernel, U, bc, law, nx, ny, is_bottom)
    threads = 256
    blocks = cld(nx + 4, threads)
    @cuda threads = threads blocks = blocks kernel(U, bc, law, nx, ny, is_bottom)
    return nothing
end

function fill_kernel!(A, value)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx <= length(A) || return
    @inbounds A[idx] = value
    return
end

function speed_kernel!(speed, prob, U, nx, ny)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx <= length(speed) || return
    ix = mod1(idx, nx)
    iy = fld(idx - 1, nx) + 1
    @inbounds begin
        w = FVM.conserved_to_primitive(prob.law, U[ix + 2, iy + 2])
        λx = FVM.max_wave_speed(prob.law, w, 1)
        λy = FVM.max_wave_speed(prob.law, w, 2)
        speed[idx] = λx / prob.mesh.dx + λy / prob.mesh.dy
    end
    return
end

function rhs_kernel_2d!(dU, U, prob, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    (1 <= ix <= nx && 1 <= iy <= ny) || return

    ii = ix + 2
    jj = iy + 2

    @inbounds begin
        wL_x, wR_x = FVM._reconstruct_face_2d(prob.reconstruction, prob.law, U, ii - 1, ii, jj, 1, nx)
        wL_xp, wR_xp = FVM._reconstruct_face_2d(prob.reconstruction, prob.law, U, ii, ii + 1, jj, 1, nx)
        fx_l = FVM.solve_riemann(prob.riemann_solver, prob.law, wL_x, wR_x, 1)
        fx_r = FVM.solve_riemann(prob.riemann_solver, prob.law, wL_xp, wR_xp, 1)

        wL_y, wR_y = FVM._reconstruct_face_2d_y(prob.reconstruction, prob.law, U, ii, jj - 1, jj, ny)
        wL_yp, wR_yp = FVM._reconstruct_face_2d_y(prob.reconstruction, prob.law, U, ii, jj, jj + 1, ny)
        fy_b = FVM.solve_riemann(prob.riemann_solver, prob.law, wL_y, wR_y, 2)
        fy_t = FVM.solve_riemann(prob.riemann_solver, prob.law, wL_yp, wR_yp, 2)

        dU[ii, jj] = -((fx_r - fx_l) / prob.mesh.dx) - ((fy_t - fy_b) / prob.mesh.dy)
    end
    return
end

function _axpy_interior!(U, dU, dt, nx, ny)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    @cuda threads = threads blocks = blocks axpy_interior_kernel!(U, dU, dt, nx, ny)
    return U
end

function axpy_interior_kernel!(U, dU, dt, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    (1 <= ix <= nx && 1 <= iy <= ny) || return
    ii = ix + 2
    jj = iy + 2
    @inbounds U[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
    return
end

function _ssprk_stage1!(U1, U, dU, dt, nx, ny)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    @cuda threads = threads blocks = blocks ssprk_stage1_kernel!(U1, U, dU, dt, nx, ny)
    return U1
end

function _ssprk_stage2!(U2, U, U1, dU, dt, nx, ny)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    @cuda threads = threads blocks = blocks ssprk_stage2_kernel!(U2, U, U1, dU, dt, nx, ny)
    return U2
end

function _ssprk_stage3!(U, U2, dU, dt, nx, ny)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    @cuda threads = threads blocks = blocks ssprk_stage3_kernel!(U, U2, dU, dt, nx, ny)
    return U
end

function ssprk_stage1_kernel!(U1, U, dU, dt, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    (1 <= ix <= nx && 1 <= iy <= ny) || return
    ii = ix + 2
    jj = iy + 2
    @inbounds U1[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
    return
end

function ssprk_stage2_kernel!(U2, U, U1, dU, dt, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    (1 <= ix <= nx && 1 <= iy <= ny) || return
    ii = ix + 2
    jj = iy + 2
    @inbounds U2[ii, jj] = 0.75 * U[ii, jj] + 0.25 * (U1[ii, jj] + dt * dU[ii, jj])
    return
end

function ssprk_stage3_kernel!(U, U2, dU, dt, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    (1 <= ix <= nx && 1 <= iy <= ny) || return
    ii = ix + 2
    jj = iy + 2
    @inbounds U[ii, jj] = (1 / 3) * U[ii, jj] + (2 / 3) * (U2[ii, jj] + dt * dU[ii, jj])
    return
end

function left_bc_kernel!(U, bc, law, nx, ny, ::Bool)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j <= ny + 4 || return
    @inbounds _apply_vertical_bc!(U, bc, law, nx, ny, j, true)
    return
end

function right_bc_kernel!(U, bc, law, nx, ny, ::Bool)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j <= ny + 4 || return
    @inbounds _apply_vertical_bc!(U, bc, law, nx, ny, j, false)
    return
end

function bottom_bc_kernel!(U, bc, law, nx, ny, ::Bool)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= nx + 4 || return
    @inbounds _apply_horizontal_bc!(U, bc, law, nx, ny, i, true)
    return
end

function top_bc_kernel!(U, bc, law, nx, ny, ::Bool)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= nx + 4 || return
    @inbounds _apply_horizontal_bc!(U, bc, law, nx, ny, i, false)
    return
end

@inline function _apply_vertical_bc!(U, ::FVM.TransmissiveBC, law, nx, ny, j, is_left)
    if is_left
        U[2, j] = U[3, j]
        U[1, j] = U[3, j]
    else
        U[nx + 3, j] = U[nx + 2, j]
        U[nx + 4, j] = U[nx + 2, j]
    end
    return
end

@inline function _apply_vertical_bc!(U, bc::Union{FVM.InflowBC, FVM.DirichletHyperbolicBC}, law, nx, ny, j, is_left)
    u_bc = FVM.primitive_to_conserved(law, bc.state)
    if is_left
        U[2, j] = u_bc
        U[1, j] = u_bc
    else
        U[nx + 3, j] = u_bc
        U[nx + 4, j] = u_bc
    end
    return
end

@inline function _apply_vertical_bc!(U, ::FVM.ReflectiveBC, law::FVM.EulerEquations{2}, nx, ny, j, is_left)
    if is_left
        w1 = FVM.conserved_to_primitive(law, U[3, j])
        w2 = FVM.conserved_to_primitive(law, U[4, j])
        U[2, j] = FVM.primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4]))
        U[1, j] = FVM.primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4]))
    else
        w1 = FVM.conserved_to_primitive(law, U[nx + 2, j])
        w2 = FVM.conserved_to_primitive(law, U[nx + 1, j])
        U[nx + 3, j] = FVM.primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4]))
        U[nx + 4, j] = FVM.primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4]))
    end
    return
end

@inline function _apply_horizontal_bc!(U, ::FVM.TransmissiveBC, law, nx, ny, i, is_bottom)
    if is_bottom
        U[i, 2] = U[i, 3]
        U[i, 1] = U[i, 3]
    else
        U[i, ny + 3] = U[i, ny + 2]
        U[i, ny + 4] = U[i, ny + 2]
    end
    return
end

@inline function _apply_horizontal_bc!(U, bc::Union{FVM.InflowBC, FVM.DirichletHyperbolicBC}, law, nx, ny, i, is_bottom)
    u_bc = FVM.primitive_to_conserved(law, bc.state)
    if is_bottom
        U[i, 2] = u_bc
        U[i, 1] = u_bc
    else
        U[i, ny + 3] = u_bc
        U[i, ny + 4] = u_bc
    end
    return
end

@inline function _apply_horizontal_bc!(U, ::FVM.ReflectiveBC, law::FVM.EulerEquations{2}, nx, ny, i, is_bottom)
    if is_bottom
        w1 = FVM.conserved_to_primitive(law, U[i, 3])
        w2 = FVM.conserved_to_primitive(law, U[i, 4])
        U[i, 2] = FVM.primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4]))
        U[i, 1] = FVM.primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4]))
    else
        w1 = FVM.conserved_to_primitive(law, U[i, ny + 2])
        w2 = FVM.conserved_to_primitive(law, U[i, ny + 1])
        U[i, ny + 3] = FVM.primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4]))
        U[i, ny + 4] = FVM.primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4]))
    end
    return
end

function periodic_x_kernel!(U, nx, ny)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j <= ny + 4 || return
    @inbounds begin
        U[2, j] = U[nx + 2, j]
        U[1, j] = U[nx + 1, j]
        U[nx + 3, j] = U[3, j]
        U[nx + 4, j] = U[4, j]
    end
    return
end

function periodic_y_kernel!(U, nx, ny)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= nx + 4 || return
    @inbounds begin
        U[i, 2] = U[i, ny + 2]
        U[i, 1] = U[i, ny + 1]
        U[i, ny + 3] = U[i, 3]
        U[i, ny + 4] = U[i, 4]
    end
    return
end

# ============================================================
# Semidiscrete CUDA support: build_cache for GPU
# ============================================================

"""
    FVM.build_cache(prob::FVM.HyperbolicProblem2D, backend::FVM.CUDASolverBackend)

Build a 2D semidiscrete cache with CuArray storage for GPU execution.

Only supported for 2D Euler with compatible Riemann solvers and BCs.
"""
function FVM.build_cache(prob::FVM.HyperbolicProblem2D, backend::FVM.CUDASolverBackend)
    CUDA.functional() || error("CUDA.jl is loaded but not functional on this machine.")
    FVM.supports_backend(prob, backend) || FVM._unsupported_backend("build_cache(::HyperbolicProblem2D)", backend)

    if !isnothing(backend.device)
        CUDA.device!(backend.device)
    end

    nx, ny = prob.mesh.nx, prob.mesh.ny
    N = FVM.nvariables(prob.law)
    ng = 2
    FT = typeof(prob.mesh.dx)

    padded_U = CUDA.zeros(SVector{N, FT}, nx + 2 * ng, ny + 2 * ng)
    padded_dU = CUDA.zeros(SVector{N, FT}, nx + 2 * ng, ny + 2 * ng)

    return FVM.HyperbolicCache2D{N, FT, typeof(prob)}(prob, padded_U, padded_dU, nx, ny, ng)
end

function FVM.unfold_to_padded!(cache::FVM.HyperbolicCache2D{N, FT, <:Any}, u::CUDA.CuArray) where {N, FT}
    ng = cache.ng
    nx, ny = cache.nx, cache.ny
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    @cuda threads = threads blocks = blocks _unfold_kernel_2d!(cache.padded_U, u, nx, ny, ng, Val(N))
    return nothing
end

function FVM.fold_from_padded!(du::CUDA.CuArray, cache::FVM.HyperbolicCache2D{N, FT, <:Any}) where {N, FT}
    ng = cache.ng
    nx, ny = cache.nx, cache.ny
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    @cuda threads = threads blocks = blocks _fold_kernel_2d!(du, cache.padded_dU, nx, ny, ng, Val(N))
    return nothing
end

function _unfold_kernel_2d!(padded_U, u, nx, ny, ng, ::Val{N}) where {N}
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    (1 <= ix <= nx && 1 <= iy <= ny) || return
    flat_idx = (iy - 1) * nx + ix
    base = (flat_idx - 1) * N
    sv = SVector{N}(ntuple(k -> @inbounds(u[base + k]), Val(N)))
    @inbounds padded_U[ix + ng, iy + ng] = sv
    return
end

function _fold_kernel_2d!(du, padded_dU, nx, ny, ng, ::Val{N}) where {N}
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    (1 <= ix <= nx && 1 <= iy <= ny) || return
    flat_idx = (iy - 1) * nx + ix
    base = (flat_idx - 1) * N
    sv = @inbounds padded_dU[ix + ng, iy + ng]
    for k in 1:N
        @inbounds du[base + k] = sv[k]
    end
    return
end

end

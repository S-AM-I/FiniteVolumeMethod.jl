# ============================================================
# Semidiscrete Cache Types
# ============================================================
#
# Pre-allocated workspace for the ODE RHS evaluation. Ghost cells
# live in the cache (not in the ODE state vector). The ODE state
# contains only interior cell values (plus face-B for MHD/CT).

"""
    AbstractSemidiscreteCache

Abstract supertype for all semidiscrete caches used as ODE parameters.
"""
abstract type AbstractSemidiscreteCache end

# ============================================================
# 1D Cache
# ============================================================

"""
    HyperbolicCache1D{N, FT, Prob}

Pre-allocated workspace for a 1D hyperbolic problem.

`padded_U` and `padded_dU` have length `nc + 2*ng` where `ng = 2`
(matching the standard ghost cell padding used by `initialize_1d`).
"""
struct HyperbolicCache1D{N, FT, Prob} <: AbstractSemidiscreteCache
    prob::Prob
    padded_U::Vector{SVector{N, FT}}
    padded_dU::Vector{SVector{N, FT}}
    nc::Int
    ng::Int
end

# ============================================================
# 2D Cache
# ============================================================

"""
    HyperbolicCache2D{N, FT, Prob}

Pre-allocated workspace for a 2D hyperbolic problem.

`padded_U` and `padded_dU` have size `(nx + 2*ng) x (ny + 2*ng)`.
"""
struct HyperbolicCache2D{N, FT, Prob} <: AbstractSemidiscreteCache
    prob::Prob
    padded_U::Matrix{SVector{N, FT}}
    padded_dU::Matrix{SVector{N, FT}}
    nx::Int
    ny::Int
    ng::Int
end

# ============================================================
# 3D Cache
# ============================================================

"""
    HyperbolicCache3D{N, FT, Prob}

Pre-allocated workspace for a 3D hyperbolic problem.
"""
struct HyperbolicCache3D{N, FT, Prob} <: AbstractSemidiscreteCache
    prob::Prob
    padded_U::Array{SVector{N, FT}, 3}
    padded_dU::Array{SVector{N, FT}, 3}
    nx::Int
    ny::Int
    nz::Int
    ng::Int
end

# ============================================================
# Unstructured Cache
# ============================================================

"""
    UnstructuredCache{N, FT, Prob}

Pre-allocated workspace for an unstructured hyperbolic problem.
No ghost cells; edge-based flux computation operates directly on cell values.
"""
struct UnstructuredCache{N, FT, Prob} <: AbstractSemidiscreteCache
    prob::Prob
    U::Vector{SVector{N, FT}}
    dU::Vector{SVector{N, FT}}
    ntri::Int
end

# ============================================================
# MHD/CT Cache (2D)
# ============================================================

"""
    MHDCTCache2D{N, FT, Prob}

Pre-allocated workspace for a 2D MHD problem with constrained transport.

Extends `HyperbolicCache2D` with face flux storage for EMF computation.
The augmented ODE state appends face-centered B to the cell-centered
conserved variables.

State layout: `[cell_conserved (nx*ny*N) | Bx_face ((nx+1)*ny) | By_face (nx*(ny+1))]`
"""
struct MHDCTCache2D{N, FT, Prob} <: AbstractSemidiscreteCache
    prob::Prob
    padded_U::Matrix{SVector{N, FT}}
    padded_dU::Matrix{SVector{N, FT}}
    nx::Int
    ny::Int
    ng::Int
    Fx_all::Matrix{SVector{N, FT}}
    Fy_all::Matrix{SVector{N, FT}}
    emf_z::Matrix{FT}
    n_cell_vars::Int
    n_bx_face::Int
    n_by_face::Int
end

# ============================================================
# GRMHD/CT Cache (2D)
# ============================================================

"""
    GRMHDCTCache2D{N, FT, Prob, MD, FD}

Pre-allocated workspace for a 2D GRMHD problem with constrained transport.
Extends `MHDCTCache2D` with precomputed metric data.
"""
struct GRMHDCTCache2D{N, FT, Prob, MD, FD} <: AbstractSemidiscreteCache
    prob::Prob
    padded_U::Matrix{SVector{N, FT}}
    padded_dU::Matrix{SVector{N, FT}}
    nx::Int
    ny::Int
    ng::Int
    Fx_all::Matrix{SVector{N, FT}}
    Fy_all::Matrix{SVector{N, FT}}
    emf_z::Matrix{FT}
    n_cell_vars::Int
    n_bx_face::Int
    n_by_face::Int
    metric_data::MD
    face_data::FD
end

# ============================================================
# AMR Cache
# ============================================================

"""
    AMRCache{N, FT, Grid}

Pre-allocated workspace for an AMR problem. Flattens all active block
interiors into a single state vector for SciML integration.
"""
struct AMRCache{N, FT, Grid} <: AbstractSemidiscreteCache
    grid::Grid
    block_offsets::Vector{Int}
    block_ids::Vector{Int}
    total_cells::Int
    per_block_padded::Dict{Int, Matrix{SVector{N, FT}}}
    per_block_dU::Dict{Int, Matrix{SVector{N, FT}}}
    law_ref::Any
    riemann_solver_ref::Any
    reconstruction_ref::Any
    cfl::FT
    initial_time::FT
    final_time::FT
end

# ============================================================
# Constructors: build_cache
# ============================================================

function _determine_ft(prob::HyperbolicProblem)
    x0 = cell_center(prob.mesh, 1)
    w0 = prob.initial_condition(x0)
    u0 = primitive_to_conserved(prob.law, w0)
    return eltype(u0)
end

function _determine_ft_2d(prob)
    x0, y0 = cell_center(prob.mesh, 1)
    w0 = prob.initial_condition(x0, y0)
    u0 = primitive_to_conserved(prob.law, w0)
    return eltype(u0)
end

function _determine_ft_3d(prob::HyperbolicProblem3D)
    x0, y0, z0 = cell_center(prob.mesh, 1)
    w0 = prob.initial_condition(x0, y0, z0)
    u0 = primitive_to_conserved(prob.law, w0)
    return eltype(u0)
end

"""
    build_cache(prob, backend::AbstractBackend = CPUBackend())

Build a pre-allocated workspace cache for the given problem.
Dispatches on problem type to create the appropriate cache.
"""
function build_cache(prob::HyperbolicProblem, backend::AbstractBackend = CPUBackend())
    _cpu_backend_only("build_cache(::HyperbolicProblem)", backend)
    nc = ncells(prob.mesh)
    N = nvariables(prob.law)
    ng = 2
    FT = _determine_ft(prob)

    padded_U = Vector{SVector{N, FT}}(undef, nc + 2 * ng)
    padded_dU = Vector{SVector{N, FT}}(undef, nc + 2 * ng)
    zero_state = zero(SVector{N, FT})
    for i in eachindex(padded_U)
        padded_U[i] = zero_state
        padded_dU[i] = zero_state
    end

    return HyperbolicCache1D{N, FT, typeof(prob)}(prob, padded_U, padded_dU, nc, ng)
end

function build_cache(prob::HyperbolicProblem2D, backend::AbstractBackend = CPUBackend())
    _cpu_backend_only("build_cache(::HyperbolicProblem2D)", backend)
    nx, ny = prob.mesh.nx, prob.mesh.ny
    N = nvariables(prob.law)
    ng = 2
    FT = _determine_ft_2d(prob)

    padded_U = Matrix{SVector{N, FT}}(undef, nx + 2 * ng, ny + 2 * ng)
    padded_dU = Matrix{SVector{N, FT}}(undef, nx + 2 * ng, ny + 2 * ng)
    zero_state = zero(SVector{N, FT})
    for j in axes(padded_U, 2), i in axes(padded_U, 1)
        padded_U[i, j] = zero_state
        padded_dU[i, j] = zero_state
    end

    return HyperbolicCache2D{N, FT, typeof(prob)}(prob, padded_U, padded_dU, nx, ny, ng)
end

function build_cache(prob::HyperbolicProblem3D, backend::AbstractBackend = CPUBackend())
    _cpu_backend_only("build_cache(::HyperbolicProblem3D)", backend)
    nx, ny, nz = prob.mesh.nx, prob.mesh.ny, prob.mesh.nz
    N = nvariables(prob.law)
    ng = 2
    FT = _determine_ft_3d(prob)

    padded_U = Array{SVector{N, FT}, 3}(undef, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng)
    padded_dU = Array{SVector{N, FT}, 3}(undef, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng)
    zero_state = zero(SVector{N, FT})
    for k in axes(padded_U, 3), j in axes(padded_U, 2), i in axes(padded_U, 1)
        padded_U[i, j, k] = zero_state
        padded_dU[i, j, k] = zero_state
    end

    return HyperbolicCache3D{N, FT, typeof(prob)}(prob, padded_U, padded_dU, nx, ny, nz, ng)
end

function build_cache(prob::UnstructuredHyperbolicProblem, backend::AbstractBackend = CPUBackend())
    _cpu_backend_only("build_cache(::UnstructuredHyperbolicProblem)", backend)
    mesh = prob.mesh
    N = nvariables(prob.law)

    x0, y0 = mesh.tri_centroids[1]
    w0 = prob.initial_condition(x0, y0)
    u0 = primitive_to_conserved(prob.law, w0)
    FT = eltype(u0)

    ntri = mesh.ntri
    U = Vector{SVector{N, FT}}(undef, ntri)
    dU = Vector{SVector{N, FT}}(undef, ntri)
    zero_state = zero(SVector{N, FT})
    for i in 1:ntri
        U[i] = zero_state
        dU[i] = zero_state
    end

    return UnstructuredCache{N, FT, typeof(prob)}(prob, U, dU, ntri)
end

function build_mhd_ct_cache(prob::HyperbolicProblem2D, backend::AbstractBackend = CPUBackend())
    _cpu_backend_only("build_mhd_ct_cache", backend)
    nx, ny = prob.mesh.nx, prob.mesh.ny
    N = nvariables(prob.law)
    ng = 2
    FT = _determine_ft_2d(prob)

    padded_U = Matrix{SVector{N, FT}}(undef, nx + 2 * ng, ny + 2 * ng)
    padded_dU = Matrix{SVector{N, FT}}(undef, nx + 2 * ng, ny + 2 * ng)
    zero_state = zero(SVector{N, FT})
    for j in axes(padded_U, 2), i in axes(padded_U, 1)
        padded_U[i, j] = zero_state
        padded_dU[i, j] = zero_state
    end

    zero_flux = zero(SVector{N, FT})
    Fx_all = fill(zero_flux, nx + 1, ny + 2)
    Fy_all = fill(zero_flux, nx + 2, ny + 1)
    emf_z = zeros(FT, nx + 1, ny + 1)

    n_cell_vars = nx * ny * N
    n_bx_face = (nx + 1) * ny
    n_by_face = nx * (ny + 1)

    return MHDCTCache2D{N, FT, typeof(prob)}(
        prob, padded_U, padded_dU, nx, ny, ng,
        Fx_all, Fy_all, emf_z, n_cell_vars, n_bx_face, n_by_face
    )
end

function build_grmhd_ct_cache(prob::HyperbolicProblem2D{<:GRMHDEquations{2}}, backend::AbstractBackend = CPUBackend())
    _cpu_backend_only("build_grmhd_ct_cache", backend)
    nx, ny = prob.mesh.nx, prob.mesh.ny
    N = nvariables(prob.law)
    ng = 2
    FT = _determine_ft_2d(prob)

    padded_U = Matrix{SVector{N, FT}}(undef, nx + 2 * ng, ny + 2 * ng)
    padded_dU = Matrix{SVector{N, FT}}(undef, nx + 2 * ng, ny + 2 * ng)
    zero_state = zero(SVector{N, FT})
    for j in axes(padded_U, 2), i in axes(padded_U, 1)
        padded_U[i, j] = zero_state
        padded_dU[i, j] = zero_state
    end

    zero_flux = zero(SVector{N, FT})
    Fx_all = fill(zero_flux, nx + 1, ny + 2)
    Fy_all = fill(zero_flux, nx + 2, ny + 1)
    emf_z = zeros(FT, nx + 1, ny + 1)

    n_cell_vars = nx * ny * N
    n_bx_face = (nx + 1) * ny
    n_by_face = nx * (ny + 1)

    md = precompute_metric(prob.law.metric, prob.mesh)
    fd = precompute_metric_at_faces(prob.law.metric, prob.mesh)

    return GRMHDCTCache2D{N, FT, typeof(prob), typeof(md), typeof(fd)}(
        prob, padded_U, padded_dU, nx, ny, ng,
        Fx_all, Fy_all, emf_z, n_cell_vars, n_bx_face, n_by_face,
        md, fd
    )
end

function build_amr_cache(prob::AMRProblem)
    grid = prob.grid
    law = grid.law
    N = nvariables(law)
    FT = Float64

    # Collect active blocks and compute offsets
    ids = sort!(collect(keys(filter(p -> p.second.active, grid.blocks))))
    offsets = Vector{Int}(undef, length(ids))
    cumulative = 0
    for (idx, bid) in enumerate(ids)
        offsets[idx] = cumulative
        block = grid.blocks[bid]
        cumulative += prod(block.dims)
    end

    per_block_padded = Dict{Int, Matrix{SVector{N, FT}}}()
    per_block_dU = Dict{Int, Matrix{SVector{N, FT}}}()
    zero_state = zero(SVector{N, FT})
    for bid in ids
        block = grid.blocks[bid]
        nx, ny = block.dims[1], block.dims[2]
        pad = Matrix{SVector{N, FT}}(undef, nx + 4, ny + 4)
        du_pad = Matrix{SVector{N, FT}}(undef, nx + 4, ny + 4)
        for j in axes(pad, 2), i in axes(pad, 1)
            pad[i, j] = zero_state
            du_pad[i, j] = zero_state
        end
        per_block_padded[bid] = pad
        per_block_dU[bid] = du_pad
    end

    return AMRCache{N, FT, typeof(grid)}(
        grid, offsets, ids, cumulative,
        per_block_padded, per_block_dU,
        law, prob.riemann_solver, prob.reconstruction,
        prob.cfl, prob.initial_time, prob.final_time
    )
end

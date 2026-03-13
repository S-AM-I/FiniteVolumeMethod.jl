# ============================================================
# Solution Accessors and Snapshots
# ============================================================
#
# Canonical accessors for extracting structured state from SciML
# solutions across node-based parabolic/FVM problems, cell-centered
# hyperbolic problems, constrained-transport MHD, and semidiscrete AMR.

abstract type AbstractFVMSolutionAccessor end

"""
    FVMSolutionAccessor

Accessor for node-based `FVMProblem` and `FVMSystem` solutions.
"""
struct FVMSolutionAccessor{M, V} <: AbstractFVMSolutionAccessor
    mesh::M
    variable_names::V
    layout::Symbol
end

function FVMSolutionAccessor(prob::FVMProblem)
    return FVMSolutionAccessor(prob.mesh, ["u"], :node_values)
end

function FVMSolutionAccessor(prob::FVMSystem{N}) where {N}
    return FVMSolutionAccessor(prob.mesh, ["u_$(i)" for i in 1:N], :node_system_values)
end

FVMSolutionAccessor(prob::SteadyFVMProblem) = FVMSolutionAccessor(prob.problem)
FVMSolutionAccessor(prob::AbstractFVMTemplate) = FVMSolutionAccessor(prob.mesh, ["u"], :node_values)

"""
    HyperbolicSolutionAccessor{N, Law, Mesh}

Accessor for extracting structured data from an ODE solution
produced by solving a hyperbolic `ODEProblem`.

# Usage
```julia
ode_prob = ODEProblem(prob)
sol = solve(ode_prob, SSPRK33(); adaptive = false, dt = 1e-3)
accessor = HyperbolicSolutionAccessor(prob)
U = get_conserved(accessor, sol, length(sol.t))
W = get_primitive(accessor, sol, length(sol.t))
```
"""
struct HyperbolicSolutionAccessor{N, Law, Mesh} <: AbstractFVMSolutionAccessor
    law::Law
    mesh::Mesh
    dims::NTuple  # () for 1D, (nx,ny) for 2D, etc.
end

function HyperbolicSolutionAccessor(prob::HyperbolicProblem)
    N = nvariables(prob.law)
    return HyperbolicSolutionAccessor{N, typeof(prob.law), typeof(prob.mesh)}(
        prob.law, prob.mesh, ()
    )
end

function HyperbolicSolutionAccessor(prob::HyperbolicProblem2D)
    N = nvariables(prob.law)
    return HyperbolicSolutionAccessor{N, typeof(prob.law), typeof(prob.mesh)}(
        prob.law, prob.mesh, (prob.mesh.nx, prob.mesh.ny)
    )
end

function HyperbolicSolutionAccessor(prob::HyperbolicProblem3D)
    N = nvariables(prob.law)
    return HyperbolicSolutionAccessor{N, typeof(prob.law), typeof(prob.mesh)}(
        prob.law, prob.mesh, (prob.mesh.nx, prob.mesh.ny, prob.mesh.nz)
    )
end

function HyperbolicSolutionAccessor(prob::UnstructuredHyperbolicProblem)
    N = nvariables(prob.law)
    return HyperbolicSolutionAccessor{N, typeof(prob.law), typeof(prob.mesh)}(
        prob.law, prob.mesh, (prob.mesh.ntri,)
    )
end

"""
    MHDSolutionAccessor{N, Law, Mesh}

Accessor for extracting structured data from an augmented-state MHD
ODE solution, including constrained transport face-B fields.
"""
struct MHDSolutionAccessor{N, Law, Mesh} <: AbstractFVMSolutionAccessor
    law::Law
    mesh::Mesh
    nx::Int
    ny::Int
    n_cell_vars::Int
    n_bx_face::Int
    n_by_face::Int
end

function MHDSolutionAccessor(prob::HyperbolicProblem2D)
    N = nvariables(prob.law)
    nx, ny = prob.mesh.nx, prob.mesh.ny
    n_cell_vars = nx * ny * N
    n_bx_face = (nx + 1) * ny
    n_by_face = nx * (ny + 1)
    return MHDSolutionAccessor{N, typeof(prob.law), typeof(prob.mesh)}(
        prob.law, prob.mesh, nx, ny, n_cell_vars, n_bx_face, n_by_face
    )
end

"""
    AMRODESolutionAccessor{N, Law, Grid}

Accessor for semidiscrete AMR `ODEProblem` solutions, decoding the
flattened SciML state back into per-block arrays.
"""
struct AMRODESolutionAccessor{N, Law, Grid} <: AbstractFVMSolutionAccessor
    law::Law
    grid::Grid
    block_offsets::Vector{Int}
    block_ids::Vector{Int}
end

function AMRODESolutionAccessor(cache::AMRCache{N}) where {N}
    return AMRODESolutionAccessor{N, typeof(cache.law_ref), typeof(cache.grid)}(
        cache.law_ref,
        cache.grid,
        copy(cache.block_offsets),
        copy(cache.block_ids),
    )
end

AMRODESolutionAccessor(prob::AMRProblem) = AMRODESolutionAccessor(build_amr_cache(prob))

_solution_time(sol, i) = sol.t[i]

"""
    solution_accessor(prob)

Return the canonical accessor for `prob`. This is the preferred
family-agnostic entry point for decoding solution state.
"""
solution_accessor(prob::Union{FVMProblem, FVMSystem, SteadyFVMProblem, AbstractFVMTemplate}) =
    FVMSolutionAccessor(prob)
solution_accessor(prob::HyperbolicProblem) = HyperbolicSolutionAccessor(prob)
solution_accessor(prob::HyperbolicProblem2D{<:IdealMHDEquations{2}}) = MHDSolutionAccessor(prob)
solution_accessor(prob::HyperbolicProblem2D{<:SRMHDEquations{2}}) = MHDSolutionAccessor(prob)
solution_accessor(prob::HyperbolicProblem2D{<:GRMHDEquations{2}}) = MHDSolutionAccessor(prob)
solution_accessor(prob::HyperbolicProblem2D) = HyperbolicSolutionAccessor(prob)
solution_accessor(prob::HyperbolicProblem3D) = HyperbolicSolutionAccessor(prob)
solution_accessor(prob::UnstructuredHyperbolicProblem) = HyperbolicSolutionAccessor(prob)
solution_accessor(prob::AMRProblem) = AMRODESolutionAccessor(prob)
solution_accessor(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:HyperbolicCache1D}) = HyperbolicSolutionAccessor(ode_prob.p.prob)
solution_accessor(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:HyperbolicCache2D}) = HyperbolicSolutionAccessor(ode_prob.p.prob)
solution_accessor(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:HyperbolicCache3D}) = HyperbolicSolutionAccessor(ode_prob.p.prob)
solution_accessor(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:UnstructuredCache}) = HyperbolicSolutionAccessor(ode_prob.p.prob)
solution_accessor(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:MHDCTCache2D}) = MHDSolutionAccessor(ode_prob.p.prob)
solution_accessor(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:GRMHDCTCache2D}) = MHDSolutionAccessor(ode_prob.p.prob)
solution_accessor(ode_prob::ODEProblem{<:Any, <:Any, <:Any, <:AMRCache}) = AMRODESolutionAccessor(ode_prob.p)

"""
    solution_variables(accessor)

Return the variable names associated with the accessor layout.
"""
solution_variables(accessor::FVMSolutionAccessor) = copy(accessor.variable_names)
solution_variables(accessor::HyperbolicSolutionAccessor) = variable_names(accessor.law)
solution_variables(accessor::MHDSolutionAccessor) = variable_names(accessor.law)
solution_variables(accessor::AMRODESolutionAccessor) = variable_names(accessor.law)

"""
    solution_state_layout(accessor)

Return a symbolic description of the state layout exposed by `accessor`.
"""
solution_state_layout(accessor::FVMSolutionAccessor) = accessor.layout
solution_state_layout(::HyperbolicSolutionAccessor) = :cell_centered_conserved
solution_state_layout(::MHDSolutionAccessor) = :cell_centered_conserved_with_ct
solution_state_layout(::AMRODESolutionAccessor) = :block_cell_centered_conserved

"""
    get_conserved(accessor::HyperbolicSolutionAccessor{N}, sol, i) where {N}

Get conserved variables at time index `i` as a vector of SVectors.
"""
function get_conserved(accessor::HyperbolicSolutionAccessor{N}, sol, i) where {N}
    FT = eltype(sol.u[i])
    return copy(reinterpret(SVector{N, FT}, copy(sol.u[i])))
end

"""
    get_primitive(accessor::HyperbolicSolutionAccessor{N}, sol, i) where {N}

Get primitive variables at time index `i` as a vector of SVectors.
"""
function get_primitive(accessor::HyperbolicSolutionAccessor{N}, sol, i) where {N}
    FT = eltype(sol.u[i])
    U = reinterpret(SVector{N, FT}, copy(sol.u[i]))
    return [conserved_to_primitive(accessor.law, u) for u in U]
end

"""
    get_coordinates(accessor::FVMSolutionAccessor)

Get the node coordinates corresponding to a node-based FVM solution.
"""
function get_coordinates(accessor::FVMSolutionAccessor)
    tri = accessor.mesh.triangulation
    return [get_point(accessor.mesh, i) for i in DelaunayTriangulation.each_point_index(tri)]
end

"""
    get_coordinates(accessor::HyperbolicSolutionAccessor)

Get cell center coordinates.
"""
function get_coordinates(accessor::HyperbolicSolutionAccessor{N, Law, Mesh}) where {N, Law, Mesh}
    mesh = accessor.mesh
    if isempty(accessor.dims)
        nc = ncells(mesh)
        return [cell_center(mesh, i) for i in 1:nc]
    elseif length(accessor.dims) == 2
        nx, ny = accessor.dims
        if hasproperty(mesh, :nx)
            return [(cell_center(mesh, cell_idx(mesh, ix, iy))) for ix in 1:nx, iy in 1:ny]
        else
            return mesh.tri_centroids
        end
    elseif length(accessor.dims) == 3
        nx, ny, nz = accessor.dims
        return [(cell_center(mesh, cell_idx_3d(mesh, ix, iy, iz))) for ix in 1:nx, iy in 1:ny, iz in 1:nz]
    else
        return mesh.tri_centroids
    end
end

"""
    get_conserved(accessor::MHDSolutionAccessor{N}, sol, i) where {N}

Get cell-centered conserved variables (excluding face-B) as an nx x ny matrix.
"""
function get_conserved(accessor::MHDSolutionAccessor{N}, sol, i) where {N}
    FT = eltype(sol.u[i])
    u = sol.u[i]
    cell_data = @view u[1:(accessor.n_cell_vars)]
    U_flat = reinterpret(SVector{N, FT}, copy(cell_data))
    nx, ny = accessor.nx, accessor.ny
    return reshape(U_flat, nx, ny)
end

"""
    get_primitive(accessor::MHDSolutionAccessor{N}, sol, i) where {N}

Get cell-centered primitive variables as an nx x ny matrix.
"""
function get_primitive(accessor::MHDSolutionAccessor{N}, sol, i) where {N}
    U = get_conserved(accessor, sol, i)
    return [conserved_to_primitive(accessor.law, U[ix, iy]) for ix in axes(U, 1), iy in axes(U, 2)]
end

"""
    get_ct_state(accessor::MHDSolutionAccessor, sol, i) -> CTData2D

Extract the constrained transport face-centered B fields from the
augmented state at time index `i`.
"""
function get_ct_state(accessor::MHDSolutionAccessor, sol, i)
    u = sol.u[i]
    nx, ny = accessor.nx, accessor.ny
    FT = eltype(u)

    bx_start = accessor.n_cell_vars + 1
    bx_end = accessor.n_cell_vars + accessor.n_bx_face
    Bx_face = reshape(copy(@view u[bx_start:bx_end]), nx + 1, ny)

    by_start = bx_end + 1
    by_end = bx_end + accessor.n_by_face
    By_face = reshape(copy(@view u[by_start:by_end]), nx, ny + 1)

    emf_z = zeros(FT, nx + 1, ny + 1)

    return CTData2D(Bx_face, By_face, emf_z)
end

"""
    get_coordinates(accessor::MHDSolutionAccessor)

Get cell center coordinates as an nx x ny matrix.
"""
function get_coordinates(accessor::MHDSolutionAccessor)
    mesh = accessor.mesh
    nx, ny = accessor.nx, accessor.ny
    return [(cell_center(mesh, cell_idx(mesh, ix, iy))) for ix in 1:nx, iy in 1:ny]
end

"""
    get_conserved(accessor::AMRODESolutionAccessor{N}, sol, i) where {N}

Decode the flattened semidiscrete AMR state into per-block conserved arrays.
"""
function get_conserved(accessor::AMRODESolutionAccessor{N}, sol, i) where {N}
    u = sol.u[i]
    FT = eltype(u)
    states = Dict{Int, Matrix{SVector{N, FT}}}()

    for (idx, bid) in enumerate(accessor.block_ids)
        offset = accessor.block_offsets[idx]
        block = accessor.grid.blocks[bid]
        nx, ny = block.dims[1], block.dims[2]
        ncells_block = nx * ny
        block_state = reinterpret(
            SVector{N, FT},
            copy(@view u[(offset * N + 1):((offset + ncells_block) * N)])
        )
        states[bid] = reshape(block_state, nx, ny)
    end

    return states
end

"""
    get_primitive(accessor::AMRODESolutionAccessor{N}, sol, i) where {N}

Decode the flattened semidiscrete AMR state into per-block primitive arrays.
"""
function get_primitive(accessor::AMRODESolutionAccessor{N}, sol, i) where {N}
    conserved = get_conserved(accessor, sol, i)
    primitives = Dict{Int, Any}()

    for (bid, block_state) in conserved
        primitives[bid] = [conserved_to_primitive(accessor.law, block_state[ix, iy]) for ix in axes(block_state, 1), iy in axes(block_state, 2)]
    end

    return primitives
end

"""
    get_coordinates(accessor::AMRODESolutionAccessor)

Get block-local cell center coordinates for the semidiscrete AMR state.
"""
function get_coordinates(accessor::AMRODESolutionAccessor)
    coords = Dict{Int, Any}()

    for bid in accessor.block_ids
        block = accessor.grid.blocks[bid]
        nx, ny = block.dims[1], block.dims[2]
        coords[bid] = [block_cell_center(block, ix, iy) for ix in 1:nx, iy in 1:ny]
    end

    return coords
end

"""
    solution_coordinates(accessor)

Return the physical coordinates associated with `accessor`.
"""
solution_coordinates(accessor::AbstractFVMSolutionAccessor) = get_coordinates(accessor)

"""
    solution_snapshot(accessor, sol, i)

Return a structured snapshot at time index `i` containing the raw SciML
state and decoded physical fields.
"""
function solution_snapshot(accessor::FVMSolutionAccessor, sol, i)
    values = copy(sol.u[i])
    return (
        time = _solution_time(sol, i),
        layout = solution_state_layout(accessor),
        variables = solution_variables(accessor),
        coordinates = solution_coordinates(accessor),
        raw_state = values,
        values = values,
    )
end

function solution_snapshot(accessor::HyperbolicSolutionAccessor, sol, i)
    return (
        time = _solution_time(sol, i),
        layout = solution_state_layout(accessor),
        variables = solution_variables(accessor),
        coordinates = solution_coordinates(accessor),
        raw_state = copy(sol.u[i]),
        conserved = get_conserved(accessor, sol, i),
        primitive = get_primitive(accessor, sol, i),
    )
end

function solution_snapshot(accessor::MHDSolutionAccessor, sol, i)
    return (
        time = _solution_time(sol, i),
        layout = solution_state_layout(accessor),
        variables = solution_variables(accessor),
        coordinates = solution_coordinates(accessor),
        raw_state = copy(sol.u[i]),
        conserved = get_conserved(accessor, sol, i),
        primitive = get_primitive(accessor, sol, i),
        ct_state = get_ct_state(accessor, sol, i),
    )
end

function solution_snapshot(accessor::AMRODESolutionAccessor, sol, i)
    return (
        time = _solution_time(sol, i),
        layout = solution_state_layout(accessor),
        variables = solution_variables(accessor),
        coordinates = solution_coordinates(accessor),
        raw_state = copy(sol.u[i]),
        conserved = get_conserved(accessor, sol, i),
        primitive = get_primitive(accessor, sol, i),
    )
end

solution_snapshot(prob, sol, i) = solution_snapshot(solution_accessor(prob), sol, i)

# ============================================================
# AMR Solution
# ============================================================

"""
    AMRSolution{Sols, Grid, FT}

Result type for AMR problems solved via segmented ODEProblem.

# Fields
- `segments`: Vector of ODE solutions, one per regrid interval.
- `grid`: Final AMR grid with solution data.
- `t_final`: Final simulation time.
"""
struct AMRSolution{Sols, Grid, FT}
    segments::Sols
    grid::Grid
    t_final::FT
end

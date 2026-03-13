# ============================================================
# Solution Accessors for Hyperbolic ODEProblem Solutions
# ============================================================
#
# Structured accessors that extract conserved/primitive variables,
# coordinates, and CT state from SciML ODE solutions.

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
struct HyperbolicSolutionAccessor{N, Law, Mesh}
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
    get_coordinates(accessor::HyperbolicSolutionAccessor)

Get cell center coordinates.
"""
function get_coordinates(accessor::HyperbolicSolutionAccessor{N, Law, Mesh}) where {N, Law, Mesh}
    mesh = accessor.mesh
    if isempty(accessor.dims)
        # 1D
        nc = ncells(mesh)
        return [cell_center(mesh, i) for i in 1:nc]
    elseif length(accessor.dims) == 2
        nx, ny = accessor.dims
        if hasproperty(mesh, :nx)
            return [(cell_center(mesh, cell_idx(mesh, ix, iy))) for ix in 1:nx, iy in 1:ny]
        else
            # Unstructured with ntri
            return mesh.tri_centroids
        end
    elseif length(accessor.dims) == 3
        nx, ny, nz = accessor.dims
        return [(cell_center(mesh, cell_idx_3d(mesh, ix, iy, iz))) for ix in 1:nx, iy in 1:ny, iz in 1:nz]
    else
        # Unstructured
        return mesh.tri_centroids
    end
end

# ============================================================
# MHD Solution Accessor
# ============================================================

"""
    MHDSolutionAccessor{N, Law, Mesh}

Accessor for extracting structured data from an augmented-state MHD
ODE solution, including constrained transport face-B fields.
"""
struct MHDSolutionAccessor{N, Law, Mesh}
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

    Bx_start = accessor.n_cell_vars + 1
    Bx_end = accessor.n_cell_vars + accessor.n_bx_face
    Bx_face = reshape(copy(@view u[Bx_start:Bx_end]), nx + 1, ny)

    By_start = Bx_end + 1
    By_end = Bx_end + accessor.n_by_face
    By_face = reshape(copy(@view u[By_start:By_end]), nx, ny + 1)

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

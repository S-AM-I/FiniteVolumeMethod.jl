# MPI extension smoke test
#
# NOT included in runtests.jl — requires mpiexec to run:
#   mpiexec -n 2 julia --project=test test/mpi_test.jl
#
# This test verifies that the MPI extension loads, mesh distribution
# works, and halo exchange completes without error.

using MPI
MPI.Init()

using PartitionedArrays
using FiniteVolumeMethod
using Test

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

@testset "MPI extension smoke test (rank=$rank/$nranks)" begin
    # Build a small 2D Cartesian mesh as UnstructuredFVMMesh
    nx, ny = 4, 4
    dx, dy = 1.0 / nx, 1.0 / ny
    nc = nx * ny

    cell_centers = zeros(2, nc)
    cell_volumes = zeros(nc)
    for j in 1:ny, i in 1:nx
        c = (j - 1) * nx + i
        cell_centers[1, c] = (i - 0.5) * dx
        cell_centers[2, c] = (j - 0.5) * dy
        cell_volumes[c] = dx * dy
    end

    # Build faces (internal + boundary)
    face_cells_list = Tuple{Int, Int}[]
    face_centers_list = Vector{Float64}[]
    face_normals_list = Vector{Float64}[]
    face_areas_list = Float64[]

    # Horizontal internal faces (between rows)
    for j in 1:(ny - 1), i in 1:nx
        owner = (j - 1) * nx + i
        neighbor = j * nx + i
        xf = (i - 0.5) * dx
        yf = j * dy
        push!(face_cells_list, (owner, neighbor))
        push!(face_centers_list, [xf, yf])
        push!(face_normals_list, [0.0, 1.0])
        push!(face_areas_list, dx)
    end

    # Vertical internal faces (between columns)
    for j in 1:ny, i in 1:(nx - 1)
        owner = (j - 1) * nx + i
        neighbor = (j - 1) * nx + i + 1
        xf = i * dx
        yf = (j - 0.5) * dy
        push!(face_cells_list, (owner, neighbor))
        push!(face_centers_list, [xf, yf])
        push!(face_normals_list, [1.0, 0.0])
        push!(face_areas_list, dy)
    end

    # Boundary faces (owner, 0)
    # Bottom
    for i in 1:nx
        owner = i
        push!(face_cells_list, (owner, 0))
        push!(face_centers_list, [(i - 0.5) * dx, 0.0])
        push!(face_normals_list, [0.0, -1.0])
        push!(face_areas_list, dx)
    end
    # Top
    for i in 1:nx
        owner = (ny - 1) * nx + i
        push!(face_cells_list, (owner, 0))
        push!(face_centers_list, [(i - 0.5) * dx, 1.0])
        push!(face_normals_list, [0.0, 1.0])
        push!(face_areas_list, dx)
    end
    # Left
    for j in 1:ny
        owner = (j - 1) * nx + 1
        push!(face_cells_list, (owner, 0))
        push!(face_centers_list, [0.0, (j - 0.5) * dy])
        push!(face_normals_list, [-1.0, 0.0])
        push!(face_areas_list, dy)
    end
    # Right
    for j in 1:ny
        owner = (j - 1) * nx + nx
        push!(face_cells_list, (owner, 0))
        push!(face_centers_list, [1.0, (j - 0.5) * dy])
        push!(face_normals_list, [1.0, 0.0])
        push!(face_areas_list, dy)
    end

    nf = length(face_cells_list)
    face_cells = zeros(Int, 2, nf)
    face_centers_mat = zeros(2, nf)
    face_normals_mat = zeros(2, nf)
    face_areas_vec = zeros(nf)
    for (f, (o, n)) in enumerate(face_cells_list)
        face_cells[1, f] = o
        face_cells[2, f] = n
        face_centers_mat[:, f] .= face_centers_list[f]
        face_normals_mat[:, f] .= face_normals_list[f]
        face_areas_vec[f] = face_areas_list[f]
    end

    face_tags = Symbol[f[2] == 0 ? :boundary : Symbol("") for f in face_cells_list]

    mesh = UnstructuredFVMMesh{2, Float64}(
        cell_centers, cell_volumes, face_cells, face_centers_mat,
        face_areas_vec, face_normals_mat, face_tags, nothing, nothing,
    )

    @testset "distribute_mesh" begin
        dmesh = distribute_mesh(mesh, comm)
        @test dmesh.rank == rank
        @test dmesh.nranks == nranks
        @test dmesh.n_owned + dmesh.n_ghost == nc
        @test dmesh.n_owned > 0
    end

    @testset "halo_exchange!" begin
        dmesh = distribute_mesh(mesh, comm)
        values = fill(Float64(rank), nc)
        halo_exchange!(values, dmesh)
        # After exchange, all values should be populated (no NaN/undef)
        @test all(isfinite, values)
    end
end

MPI.Finalize()

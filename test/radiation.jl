using FiniteVolumeMethod
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# -- Mesh builder (copied from incompressible.jl -- safe_include isolates modules) --

"""
    build_cartesian_unstructured_mesh(nx, ny, Lx, Ly)

Create an `UnstructuredFVMMesh{2, Float64}` from a regular Cartesian grid
with `nx x ny` cells on domain `[0, Lx] x [0, Ly]`.

Face normals follow the convention:
- Internal vertical faces (between i and i+1): normal `(1, 0)`, area `dy`
- Internal horizontal faces (between j and j+1): normal `(0, 1)`, area `dx`
- Boundary faces point outward from the domain

Boundary face tags: `:left`, `:right`, `:bottom`, `:top`.
Internal face tags: `:internal`.
"""
function build_cartesian_unstructured_mesh(
        nx::Int, ny::Int, Lx::Float64, Ly::Float64,
    )
    dx = Lx / nx
    dy = Ly / ny
    ncells = nx * ny

    # Cell indexing: (i, j) -> (j-1)*nx + i,  i in 1:nx, j in 1:ny
    cell_idx(i, j) = (j - 1) * nx + i

    # Cell centers and volumes
    cell_centers = zeros(Float64, 2, ncells)
    cell_volumes = zeros(Float64, ncells)
    for j in 1:ny, i in 1:nx
        c = cell_idx(i, j)
        cell_centers[1, c] = (i - 0.5) * dx
        cell_centers[2, c] = (j - 0.5) * dy
        cell_volumes[c] = dx * dy
    end

    # Build faces: internal vertical, internal horizontal, then boundary
    face_cells_list = Tuple{Int, Int}[]    # (owner, neighbour)
    face_normals_list = Tuple{Float64, Float64}[]
    face_areas_list = Float64[]
    face_centers_list = Tuple{Float64, Float64}[]
    face_tags_list = Symbol[]

    # Internal vertical faces: between (i, j) and (i+1, j)
    for j in 1:ny, i in 1:(nx - 1)
        P = cell_idx(i, j)
        N = cell_idx(i + 1, j)
        push!(face_cells_list, (P, N))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (i * dx, (j - 0.5) * dy))
        push!(face_tags_list, :internal)
    end

    # Internal horizontal faces: between (i, j) and (i, j+1)
    for j in 1:(ny - 1), i in 1:nx
        P = cell_idx(i, j)
        N = cell_idx(i, j + 1)
        push!(face_cells_list, (P, N))
        push!(face_normals_list, (0.0, 1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, j * dy))
        push!(face_tags_list, :internal)
    end

    # Boundary: left (i=1, normal=-x)
    for j in 1:ny
        P = cell_idx(1, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (-1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (0.0, (j - 0.5) * dy))
        push!(face_tags_list, :left)
    end

    # Boundary: right (i=nx, normal=+x)
    for j in 1:ny
        P = cell_idx(nx, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (Lx, (j - 0.5) * dy))
        push!(face_tags_list, :right)
    end

    # Boundary: bottom (j=1, normal=-y)
    for i in 1:nx
        P = cell_idx(i, 1)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (0.0, -1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, 0.0))
        push!(face_tags_list, :bottom)
    end

    # Boundary: top (j=ny, normal=+y)
    for i in 1:nx
        P = cell_idx(i, ny)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (0.0, 1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, Ly))
        push!(face_tags_list, :top)
    end

    nfaces = length(face_cells_list)

    # Pack into matrices
    face_cells = zeros(Int, 2, nfaces)
    face_normals = zeros(Float64, 2, nfaces)
    face_areas = zeros(Float64, nfaces)
    face_centers = zeros(Float64, 2, nfaces)
    for f in 1:nfaces
        face_cells[1, f] = face_cells_list[f][1]
        face_cells[2, f] = face_cells_list[f][2]
        face_normals[1, f] = face_normals_list[f][1]
        face_normals[2, f] = face_normals_list[f][2]
        face_areas[f] = face_areas_list[f]
        face_centers[1, f] = face_centers_list[f][1]
        face_centers[2, f] = face_centers_list[f][2]
    end

    # Build cell_faces connectivity
    cell_faces = [Int[] for _ in 1:ncells]
    for f in 1:nfaces
        P = face_cells[1, f]
        push!(cell_faces[P], f)
        N = face_cells[2, f]
        if N != 0
            push!(cell_faces[N], f)
        end
    end

    return FiniteVolumeMethod.UnstructuredFVMMesh{2, Float64}(
        cell_centers,
        cell_volumes,
        face_cells,
        face_centers,
        face_areas,
        face_normals,
        face_tags_list,
        nothing,       # face_velocity
        cell_faces,
    )
end

# -- Tests --------------------------------------------------------------------

@testset "Radiation (Phase 9)" begin

    # -- 1. P1Model construction -----------------------------------------------
    @testset "P1Model construction" begin
        # Default
        m = P1Model()
        @test m isa P1Model{Float64}
        @test m.a == 0.1

        # Custom
        m2 = P1Model(; a = 0.5)
        @test m2.a == 0.5
    end

    # -- 2. RadiationState construction ----------------------------------------
    @testset "RadiationState construction" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        rs = RadiationState(mesh; G_init = 100.0)
        @test rs isa RadiationState{Float64}
        @test length(rs.G.internal) == 9
        @test all(==(100.0), rs.G.internal)

        # Default (G_init = 0)
        rs0 = RadiationState(mesh)
        @test all(==(0.0), rs0.G.internal)
    end

    # -- 3. STEFAN_BOLTZMANN value ---------------------------------------------
    @testset "STEFAN_BOLTZMANN value" begin
        @test STEFAN_BOLTZMANN isa Float64
        @test STEFAN_BOLTZMANN ≈ 5.67e-8 atol = 1.0e-10
    end

    # -- 4. marshak_wall_bc ----------------------------------------------------
    @testset "marshak_wall_bc" begin
        rad = P1Model(; a = 0.5)
        T_wall = 1000.0
        bc = marshak_wall_bc(rad, T_wall)
        @test bc isa ParabolicRobin
        @test bc.a == 1.0
        @test bc.b ≈ 2.0 / (3.0 * 0.5)
        @test bc.c ≈ 4.0 * STEFAN_BOLTZMANN * 1000.0^4
    end

    # -- 5. radiation_inlet_bc -------------------------------------------------
    @testset "radiation_inlet_bc" begin
        T_inlet = 500.0
        bc = radiation_inlet_bc(T_inlet)
        @test bc isa ParabolicDirichlet
        @test bc.value ≈ 4.0 * STEFAN_BOLTZMANN * 500.0^4
    end

    # -- 6. assemble_p1! smoke -------------------------------------------------
    @testset "assemble_p1! smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        rad = P1Model(; a = 0.2)
        T_field = CollocatedScalarField(:T, mesh; value = 300.0)
        bcs_G = Dict{Symbol, AbstractBoundaryCondition}(
            :left => ParabolicDirichlet(0.0),
            :right => ParabolicDirichlet(0.0),
            :bottom => ParabolicDirichlet(0.0),
            :top => ParabolicDirichlet(0.0),
        )

        eq = CollocatedEquation(mesh)
        assemble_p1!(eq, rad, T_field, mesh, bcs_G)

        # Matrix should have nonzero entries
        @test nnz(eq.A) > 0
        # Diagonal should be positive (diffusion + absorption)
        for c in 1:nc
            @test eq.A[c, c] > 0
        end
        # RHS should be positive (emission from T=300K)
        @test all(>(0), eq.b)
    end

    # -- 7. solve_p1_radiation -------------------------------------------------
    @testset "solve_p1_radiation" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        rad = P1Model(; a = 1.0)
        T_val = 300.0
        T_field = CollocatedScalarField(:T, mesh; value = T_val)

        # Marshak BCs on all walls
        bcs_G = Dict{Symbol, AbstractBoundaryCondition}(
            :left => marshak_wall_bc(rad, T_val),
            :right => marshak_wall_bc(rad, T_val),
            :bottom => marshak_wall_bc(rad, T_val),
            :top => marshak_wall_bc(rad, T_val),
        )

        G = solve_p1_radiation(rad, T_field, mesh, bcs_G)

        @test G isa CollocatedScalarField{Float64}
        @test length(G.internal) == nc
        # G should be positive everywhere
        @test all(>(0), G.internal)
        # In equilibrium with uniform T and Marshak BCs, G should be near 4*sigma*T^4
        G_eq = 4.0 * STEFAN_BOLTZMANN * T_val^4
        for c in 1:nc
            @test G.internal[c] ≈ G_eq rtol = 0.2
        end
    end

    # -- 8. compute_radiation_source -------------------------------------------
    @testset "compute_radiation_source" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        rad = P1Model(; a = 0.5)
        T_val = 400.0
        sigma = STEFAN_BOLTZMANN

        T_field = CollocatedScalarField(:T, mesh; value = T_val)

        # Set G to equilibrium value: G = 4*sigma*T^4
        G_eq = 4.0 * sigma * T_val^4
        G_field = CollocatedScalarField(:G, mesh; value = G_eq)

        S_rad = compute_radiation_source(rad, G_field, T_field)

        @test length(S_rad) == nc
        # At equilibrium, S_rad = a*G - 4*a*sigma*T^4 = 0
        for c in 1:nc
            @test abs(S_rad[c]) < 1.0e-6
        end

        # Non-equilibrium: G much larger than equilibrium -> positive source
        G_high = CollocatedScalarField(:G, mesh; value = 2.0 * G_eq)
        S_high = compute_radiation_source(rad, G_high, T_field)
        @test all(>(0), S_high)

        # G = 0 -> net emission (negative source)
        G_zero = CollocatedScalarField(:G, mesh; value = 0.0)
        S_zero = compute_radiation_source(rad, G_zero, T_field)
        @test all(<(0), S_zero)
    end

    # -- 9. solve_simple_thermal_radiation smoke --------------------------------
    @testset "solve_simple_thermal_radiation smoke" begin
        mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE(; max_iterations = 5, tolerance = 1.0e-12)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        thermal_props = FluidThermalProperties{2}(; k = 0.6, Cp = 4000.0)
        rad_model = P1Model(; a = 0.1)

        bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
            :left => thermal_inlet_bc(350.0),
            :right => thermal_insulated_bc(),
            :bottom => thermal_inlet_bc(300.0),
            :top => thermal_inlet_bc(300.0),
        )

        bcs_G = Dict{Symbol, AbstractBoundaryCondition}(
            :left => radiation_inlet_bc(350.0),
            :right => marshak_wall_bc(rad_model, 300.0),
            :bottom => marshak_wall_bc(rad_model, 300.0),
            :top => marshak_wall_bc(rad_model, 300.0),
        )

        result, thermal_state, rad_state = solve_simple_thermal_radiation(
            prob, thermal_props, rad_model;
            bcs_T = bcs_T,
            bcs_G = bcs_G,
        )

        @test result isa SolveResult{2, Float64}
        @test thermal_state isa ThermalState{Float64}
        @test rad_state isa RadiationState{Float64}
        @test result.iterations == 5
        @test all(isfinite, thermal_state.T_field.internal)
        @test all(isfinite, rad_state.G.internal)
        @test all(>=(0), rad_state.G.internal)
    end

end

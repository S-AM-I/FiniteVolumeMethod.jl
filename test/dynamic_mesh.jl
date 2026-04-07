using FiniteVolumeMethod
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# ── Mesh builder (from test/incompressible.jl) ─────────────────────

function build_cartesian_unstructured_mesh(
        nx::Int, ny::Int, Lx::Float64, Ly::Float64,
    )
    dx = Lx / nx
    dy = Ly / ny
    ncells = nx * ny

    cell_idx(i, j) = (j - 1) * nx + i

    cell_centers = zeros(Float64, 2, ncells)
    cell_volumes = zeros(Float64, ncells)
    for j in 1:ny, i in 1:nx
        c = cell_idx(i, j)
        cell_centers[1, c] = (i - 0.5) * dx
        cell_centers[2, c] = (j - 0.5) * dy
        cell_volumes[c] = dx * dy
    end

    face_cells_list = Tuple{Int, Int}[]
    face_normals_list = Tuple{Float64, Float64}[]
    face_areas_list = Float64[]
    face_centers_list = Tuple{Float64, Float64}[]
    face_tags_list = Symbol[]

    for j in 1:ny, i in 1:(nx - 1)
        P = cell_idx(i, j)
        N = cell_idx(i + 1, j)
        push!(face_cells_list, (P, N))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (i * dx, (j - 0.5) * dy))
        push!(face_tags_list, :internal)
    end

    for j in 1:(ny - 1), i in 1:nx
        P = cell_idx(i, j)
        N = cell_idx(i, j + 1)
        push!(face_cells_list, (P, N))
        push!(face_normals_list, (0.0, 1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, j * dy))
        push!(face_tags_list, :internal)
    end

    for j in 1:ny
        P = cell_idx(1, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (-1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (0.0, (j - 0.5) * dy))
        push!(face_tags_list, :left)
    end

    for j in 1:ny
        P = cell_idx(nx, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (Lx, (j - 0.5) * dy))
        push!(face_tags_list, :right)
    end

    for i in 1:nx
        P = cell_idx(i, 1)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (0.0, -1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, 0.0))
        push!(face_tags_list, :bottom)
    end

    for i in 1:nx
        P = cell_idx(i, ny)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (0.0, 1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, Ly))
        push!(face_tags_list, :top)
    end

    nfaces = length(face_cells_list)

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

# ── Tests ──────────────────────────────────────────────────────────────

@testset "Dynamic/Moving Mesh (Phase 10)" begin

    # ── 1. SolidBodyMotion construction ───────────────────────────────
    @testset "SolidBodyMotion construction" begin
        func = t -> SVector(0.1 * t, 0.0)
        motion = SolidBodyMotion{2, Float64}(func)
        @test motion isa SolidBodyMotion{2, Float64}
        @test motion isa AbstractMotionSolver
        d = motion.displacement_func(1.0)
        @test d == SVector(0.1, 0.0)
        @test d isa SVector{2, Float64}
    end

    # ── 2. LaplacianMotion construction ───────────────────────────────
    @testset "LaplacianMotion construction" begin
        lm = LaplacianMotion()
        @test lm isa LaplacianMotion{Float64}
        @test lm isa AbstractMotionSolver
        @test lm.gamma == 1.0

        lm2 = LaplacianMotion(; gamma = 2.5)
        @test lm2.gamma == 2.5
    end

    # ── 3. MeshMotionState construction ───────────────────────────────
    @testset "MeshMotionState construction" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        ms = MeshMotionState(mesh)
        @test ms isa MeshMotionState{2, Float64}
        @test length(ms.displacement) == 16
        @test all(d -> d == zero(SVector{2, Float64}), ms.displacement)
        nf = size(mesh.face_cells, 2)
        @test length(ms.phi_mesh) == nf
        @test all(==(0.0), ms.phi_mesh)
        @test ms.V_old == mesh.cell_volumes
        @test ms.V_old !== mesh.cell_volumes  # should be a copy
    end

    # ── 4. compute_displacement! SolidBody ────────────────────────────
    @testset "compute_displacement! SolidBody" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        ms = MeshMotionState(mesh)
        func = t -> SVector(0.5 * t, -0.2 * t)
        motion = SolidBodyMotion{2, Float64}(func)

        compute_displacement!(ms, motion, mesh, 2.0)

        expected = SVector(1.0, -0.4)
        for c in 1:length(mesh.cell_volumes)
            @test ms.displacement[c] == expected
        end
    end

    # ── 5. ale_corrected_flux ─────────────────────────────────────────
    @testset "ale_corrected_flux" begin
        mesh = build_cartesian_unstructured_mesh(2, 2, 1.0, 1.0)
        phi = FaceFluxField(:phi, mesh)
        nf = size(mesh.face_cells, 2)

        # Set some flux values
        for f in 1:nf
            phi.values[f] = Float64(f)
        end

        phi_mesh = ones(Float64, nf) * 0.5

        phi_ale = ale_corrected_flux(phi, phi_mesh)
        @test phi_ale isa FaceFluxField{Float64}
        @test phi_ale.name == :phi_ale
        for f in 1:nf
            @test phi_ale.values[f] == Float64(f) - 0.5
        end

        # Mismatched lengths should error
        @test_throws ErrorException ale_corrected_flux(phi, zeros(Float64, nf + 1))
    end

    # ── 6. compute_mesh_flux! zero displacement ───────────────────────
    @testset "compute_mesh_flux! zero displacement" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        ms = MeshMotionState(mesh)

        # Zero displacement -> zero phi_mesh
        compute_mesh_flux!(ms, mesh, 0.1)

        @test all(v -> abs(v) < 1.0e-14, ms.phi_mesh)
    end

    # ── 7. solve_ale smoke test ───────────────────────────────────────
    @testset "solve_ale smoke test" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)

        # Solid body: small translation
        motion = SolidBodyMotion{2, Float64}(t -> SVector(0.01 * t, 0.0))

        bcs_U = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(),
            :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(),
            :top => FixedVelocityBC((1.0, 0.0)),
        )
        bcs_p = Dict{Symbol, AbstractBoundaryCondition}(
            :left => ParabolicNeumann(0.0),
            :right => ParabolicNeumann(0.0),
            :bottom => ParabolicNeumann(0.0),
            :top => ParabolicNeumann(0.0),
        )

        result = solve_ale(
            mesh, motion, bcs_U, bcs_p,
            (0.0, 0.02), 0.01;
            nu = 0.01,
            algorithm = PISO(; n_correctors = 1),
        )

        @test result isa SolveResult{2, Float64}
        @test result.iterations == 2
        @test result.converged == true
        @test haskey(result.residuals, :continuity)
        @test length(result.residuals[:continuity]) == 2
    end

    # ── 8. update_mesh! geometry update ───────────────────────────────
    @testset "update_mesh! geometry update" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        original_centers = copy(mesh.cell_centers)
        ms = MeshMotionState(mesh)

        # Uniform displacement: all cells shift by (0.1, 0.2)
        for c in 1:length(mesh.cell_volumes)
            ms.displacement[c] = SVector(0.1, 0.2)
        end

        update_mesh!(mesh, ms, 0.1)

        # Cell centers should be shifted
        for c in 1:length(mesh.cell_volumes)
            @test mesh.cell_centers[1, c] ≈ original_centers[1, c] + 0.1 atol = 1.0e-14
            @test mesh.cell_centers[2, c] ≈ original_centers[2, c] + 0.2 atol = 1.0e-14
        end

        # Uniform displacement preserves volumes (divergence-free)
        for c in 1:length(mesh.cell_volumes)
            @test mesh.cell_volumes[c] ≈ ms.V_old[c] atol = 1.0e-12
        end
    end
end

using FiniteVolumeMethod
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SciMLBase
import SciMLBase.SciMLStructures as SS

# ── Mesh builder (copied for safe_include isolation) ─────────────────

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
        nothing,
        cell_faces,
    )
end

# ── Helper: build a standard channel problem ─────────────────────────

function build_channel_problem(; max_iterations = 5, tolerance = 1.0e-10)
    mesh = build_cartesian_unstructured_mesh(8, 4, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => FixedVelocityBC((0.1, 0.0)),
        :right => FixedPressureBC(0.0),
        :bottom => NoSlipWallBC(),
        :top => NoSlipWallBC(),
    )
    algo = SIMPLE(; max_iterations = max_iterations, tolerance = tolerance)
    return IncompressibleProblem(mesh, bcs, algo; nu = 0.1)
end

# ── Tests ─────────────────────────────────────────────────────────────

@testset "Incompressible SciML Compliance" begin

    # ── 1. CommonSolve.solve dispatch (B2) ────────────────────────────
    @testset "CommonSolve.solve dispatch (B2)" begin
        prob = build_channel_problem()

        sol = solve(prob, SIMPLE(; max_iterations = 5, tolerance = 1.0e-10))
        @test sol isa IncompressibleSolution
        @test sol.iterations > 0
        @test sol.retcode in (:Success, :MaxIters)

        sol2 = solve(prob)
        @test sol2 isa IncompressibleSolution
        @test sol2.iterations > 0
        @test sol2.retcode in (:Success, :MaxIters)
    end

    # ── 2. Symbolic indexing (B3) ─────────────────────────────────────
    @testset "Symbolic indexing (B3)" begin
        prob = build_channel_problem()
        sol = solve(prob, SIMPLE(; max_iterations = 5, tolerance = 1.0e-10))
        ncells = length(prob.mesh.cell_volumes)
        nfaces = size(prob.mesh.face_cells, 2)

        U = sol[:U]
        @test U isa Vector{<:SVector}
        @test length(U) == ncells

        p = sol[:p]
        @test p isa Vector{Float64}
        @test length(p) == ncells

        ux = sol[:Ux]
        @test ux isa Vector{Float64}
        @test length(ux) == ncells

        uy = sol[:Uy]
        @test uy isa Vector{Float64}
        @test length(uy) == ncells

        phi = sol[:phi]
        @test phi isa AbstractVector
        @test length(phi) == nfaces

        @test keys(sol) == (:U, :p, :phi, :Ux, :Uy)

        @test_throws ErrorException sol[:nonexistent]
    end

    # ── 3. Solution properties ────────────────────────────────────────
    @testset "Solution properties" begin
        prob = build_channel_problem()
        sol = solve(prob, SIMPLE(; max_iterations = 5, tolerance = 1.0e-10))

        @test sol.converged isa Bool
        @test sol.iterations isa Int
        @test sol.iterations > 0
        @test sol.residuals isa Dict
        @test haskey(sol.residuals, :Ux)
        @test haskey(sol.residuals, :Uy)
        @test haskey(sol.residuals, :continuity)
        @test sol.retcode isa Symbol
    end

    # ── 4. remake (B1) ────────────────────────────────────────────────
    @testset "remake (B1)" begin
        prob = build_channel_problem()

        prob2 = remake(prob; nu = 0.05)
        @test prob2.nu == 0.05
        @test prob.nu == 0.1  # original unchanged

        prob3 = remake(prob; density = 2.0)
        @test prob3.density == 2.0
        @test prob.density == 1.0  # original unchanged

        new_algo = SIMPLE(; max_iterations = 10)
        prob4 = remake(prob; algorithm = new_algo)
        @test prob4.algorithm.max_iterations == 10
        @test prob.algorithm.max_iterations == 5  # original unchanged
    end

    # ── 5. SciMLStructures (B4) ───────────────────────────────────────
    @testset "SciMLStructures (B4)" begin
        prob = build_channel_problem()

        @test SS.isscimlstructure(prob)
        @test SS.hasportion(SS.Tunable(), prob)

        vals, repack, aliased = SS.canonicalize(SS.Tunable(), prob)
        @test !aliased

        @test vals[1] ≈ prob.nu
        @test vals[2] ≈ prob.density

        new_prob = repack([0.05, 2.0, vals[3:end]...])
        @test new_prob.nu ≈ 0.05
        @test new_prob.density ≈ 2.0

        replaced = SS.replace(SS.Tunable(), prob, [0.05, 2.0, vals[3:end]...])
        @test replaced.nu ≈ 0.05
        @test replaced.density ≈ 2.0
    end

    # ── 6. New BC types (C1) ──────────────────────────────────────────
    @testset "New BC types (C1)" begin
        # ZeroGradientBC
        zg = ZeroGradientBC()
        @test FiniteVolumeMethod.expand_velocity_bc(zg, 1) isa ParabolicNeumann
        @test FiniteVolumeMethod.expand_velocity_bc(zg, 2) isa ParabolicNeumann
        @test FiniteVolumeMethod.expand_pressure_bc(zg) isa ParabolicNeumann

        # TotalPressureBC
        tp = TotalPressureBC(101325.0)
        @test FiniteVolumeMethod.expand_pressure_bc(tp) isa ParabolicDirichlet
        @test FiniteVolumeMethod.expand_pressure_bc(tp).value ≈ 101325.0
        @test FiniteVolumeMethod.expand_velocity_bc(tp, 1) isa ParabolicNeumann

        # SymmetryBC
        sym = SymmetryBC()
        @test FiniteVolumeMethod.expand_velocity_bc(sym, 1) isa ParabolicNeumann
        @test FiniteVolumeMethod.expand_velocity_bc(sym, 2) isa ParabolicNeumann
        @test FiniteVolumeMethod.expand_pressure_bc(sym) isa ParabolicNeumann

        # FlowRateInletBC
        fr = FlowRateInletBC((0.5, 0.0))
        vel_bc = FiniteVolumeMethod.expand_velocity_bc(fr, 1)
        @test vel_bc isa ParabolicDirichlet
        @test vel_bc.value ≈ 0.5
        vel_bc_y = FiniteVolumeMethod.expand_velocity_bc(fr, 2)
        @test vel_bc_y.value ≈ 0.0
        @test FiniteVolumeMethod.expand_pressure_bc(fr) isa ParabolicNeumann

        # TimeDependentVelocityBC
        td = TimeDependentVelocityBC{2, Float64}(t -> SVector(t, 0.0))
        vel_td = FiniteVolumeMethod.expand_velocity_bc(td, 1)
        @test vel_td isa ParabolicDirichlet
        @test vel_td.value ≈ 0.0  # t_ref = 0
        vel_td_y = FiniteVolumeMethod.expand_velocity_bc(td, 2)
        @test vel_td_y.value ≈ 0.0
        @test FiniteVolumeMethod.expand_pressure_bc(td) isa ParabolicNeumann
    end
end

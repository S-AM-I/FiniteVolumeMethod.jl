using FiniteVolumeMethod
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# ── Mesh builder ─────────────────────────────────────────────────────

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

# ── Tests ──────────────────────────────────────────────────────────────

@testset "Incompressible Navier-Stokes" begin

    # ── 1. Type construction ───────────────────────────────────────────
    @testset "Type construction" begin
        s = SIMPLE()
        @test s isa SIMPLE{Float64}
        @test s.alpha_U == 0.7
        @test s.alpha_p == 0.3
        @test s.max_iterations == 1000
        @test s.tolerance == 1.0e-6

        p = PISO()
        @test p isa PISO{Float64}
        @test p.n_correctors == 2

        pm = PIMPLE()
        @test pm isa PIMPLE{Float64}
        @test pm.n_outer == 2
        @test pm.n_correctors == 1
        @test pm.alpha_U == 0.7
        @test pm.alpha_p == 0.3

        # IncompressibleState construction
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        state = IncompressibleState(mesh)
        @test length(state.U.internal) == 9
        @test length(state.p.internal) == 9
        @test all(==(1.0), state.A_P)
        @test all(==(zero(SVector{2, Float64})), state.H_U)
    end

    # ── 2. BC expansion ───────────────────────────────────────────────
    @testset "BC expansion" begin
        # FixedVelocityBC -> Dirichlet on velocity, Neumann on pressure
        fv = FixedVelocityBC((1.0, 2.0))
        @test fv.value == SVector(1.0, 2.0)
        vel_bc_x = FiniteVolumeMethod.expand_velocity_bc(fv, 1)
        vel_bc_y = FiniteVolumeMethod.expand_velocity_bc(fv, 2)
        @test vel_bc_x isa ParabolicDirichlet
        @test vel_bc_x.value == 1.0
        @test vel_bc_y.value == 2.0
        p_bc = FiniteVolumeMethod.expand_pressure_bc(fv)
        @test p_bc isa ParabolicNeumann
        @test p_bc.value == 0.0

        # FixedPressureBC -> Neumann on velocity, Dirichlet on pressure
        fp = FixedPressureBC(5.0)
        vel_bc = FiniteVolumeMethod.expand_velocity_bc(fp, 1)
        @test vel_bc isa ParabolicNeumann
        p_bc2 = FiniteVolumeMethod.expand_pressure_bc(fp)
        @test p_bc2 isa ParabolicDirichlet
        @test p_bc2.value == 5.0

        # NoSlipWallBC -> Dirichlet(0) on velocity, Neumann on pressure
        ns = NoSlipWallBC()
        @test FiniteVolumeMethod.expand_velocity_bc(ns, 1) isa ParabolicDirichlet
        @test FiniteVolumeMethod.expand_velocity_bc(ns, 1).value == 0.0
        @test FiniteVolumeMethod.expand_pressure_bc(ns) isa ParabolicNeumann

        # SlipWallBC -> Neumann on velocity, Neumann on pressure
        sw = SlipWallBC()
        @test FiniteVolumeMethod.expand_velocity_bc(sw, 1) isa ParabolicNeumann
        @test FiniteVolumeMethod.expand_pressure_bc(sw) isa ParabolicNeumann
    end

    # ── 3. Momentum assembly smoke test ────────────────────────────────
    @testset "Momentum assembly smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE()
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.01)
        state = IncompressibleState(mesh)
        FiniteVolumeMethod.update_boundary_velocity!(state, bcs, mesh)

        eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq, state, prob, 1)

        # A should be nonzero (has diffusion contributions at minimum)
        @test nnz(eq.A) > 0
        # Diagonal should be positive (diffusion adds positive diagonal)
        nc = length(mesh.cell_volumes)
        for c in 1:nc
            @test eq.A[c, c] > 0
        end
    end

    # ── 4. Pressure equation smoke test ────────────────────────────────
    @testset "Pressure assembly smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE()
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.01)
        state = IncompressibleState(mesh)
        FiniteVolumeMethod.update_boundary_velocity!(state, bcs, mesh)
        FiniteVolumeMethod.update_boundary_pressure!(state, bcs, mesh)

        # Need momentum operators for pressure assembly
        eqs = FiniteVolumeMethod.CollocatedEquation{Float64}[]
        for d in 1:2
            eq = FiniteVolumeMethod.CollocatedEquation(mesh)
            FiniteVolumeMethod.assemble_momentum!(eq, state, prob, d)
            push!(eqs, eq)
        end
        FiniteVolumeMethod.extract_momentum_operators!(state, eqs, mesh)

        p_eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_pressure!(p_eq, state, prob)

        @test nnz(p_eq.A) > 0
        nc = length(mesh.cell_volumes)
        # Pressure Laplacian should have positive diagonal
        for c in 1:nc
            @test p_eq.A[c, c] >= 0
        end
    end

    # ── 5. Pressure reference fix ──────────────────────────────────────
    @testset "Pressure reference fix" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        nc = length(mesh.cell_volumes)

        # Put some values in the matrix first
        for c in 1:nc
            eq.A[c, c] = 10.0
        end
        eq.b[1] = 5.0

        FiniteVolumeMethod.fix_pressure_reference!(eq, 1, 0.0)

        # Row 1 should be zeroed except diagonal = 1
        @test eq.A[1, 1] == 1.0
        for j in 2:nc
            @test eq.A[1, j] == 0.0
        end
        @test eq.b[1] == 0.0
        # Other rows should be untouched
        @test eq.A[2, 2] == 10.0
    end

    # ── 6. SIMPLE convergence ─────────────────────────────────────────
    @testset "SIMPLE convergence" begin
        mesh = build_cartesian_unstructured_mesh(8, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE(; max_iterations = 100, tolerance = 1.0e-6)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        result = solve_simple(prob)

        @test result isa SolveResult{2, Float64}
        @test result.iterations > 0

        # Residual histories should be populated and finite
        @test haskey(result.residuals, :Ux)
        @test haskey(result.residuals, :Uy)
        @test haskey(result.residuals, :continuity)
        @test all(isfinite, result.residuals[:Ux])
        @test all(isfinite, result.residuals[:continuity])

        # Ux residual should decrease from first to last (solver is stable)
        @test result.residuals[:Ux][end] < result.residuals[:Ux][1]

        # Velocity should be positive-x at interior
        u_x = [result.state.U.internal[c][1] for c in eachindex(result.state.U.internal)]
        @test all(u_x .>= -0.01)
    end

    # ── 7. PISO transient smoke test ───────────────────────────────────
    @testset "PISO transient smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = PISO(; n_correctors = 2)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        result = solve_incompressible(prob, (0.0, 0.02), 0.01)

        @test result isa SolveResult{2, Float64}
        @test result.converged  # transient always returns true
        @test result.iterations == 2
        @test length(result.residuals[:continuity]) == 2
    end

    # ── 8. PIMPLE transient smoke test ─────────────────────────────────
    @testset "PIMPLE transient smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = PIMPLE(; n_outer = 2, n_correctors = 1)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        result = solve_incompressible(prob, (0.0, 0.02), 0.01)

        @test result isa SolveResult{2, Float64}
        @test result.converged
        @test result.iterations == 2
        @test length(result.residuals[:continuity]) == 2
    end

    # ── 9. Residual computation ────────────────────────────────────────
    @testset "Residual computation" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        state = IncompressibleState(mesh)
        # Zero fluxes should give zero continuity residual
        r = FiniteVolumeMethod.continuity_residual(state, mesh)
        @test r == 0.0

        # Nonzero flux should give nonzero residual
        nf = size(mesh.face_cells, 2)
        for f in 1:nf
            state.phi.values[f] = 0.01
        end
        r2 = FiniteVolumeMethod.continuity_residual(state, mesh)
        @test r2 > 0.0
    end
end

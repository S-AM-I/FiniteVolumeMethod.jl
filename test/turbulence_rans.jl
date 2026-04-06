using FiniteVolumeMethod
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# ── Mesh builder (copied from incompressible.jl — safe_include isolates modules) ──

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

@testset "RANS Turbulence Models" begin

    # ── 1. Type construction ──────────────────────────────────────────
    @testset "Type construction" begin
        # StandardKEpsilon (existing type)
        ke = StandardKEpsilon()
        @test ke.C_mu == 0.09
        @test ke.sigma_k == 1.0
        @test ke.sigma_epsilon == 1.3
        @test n_turbulence_fields(ke) == 2
        @test turbulence_field_names(ke) == (:k, :epsilon)

        # KOmega
        kw = KOmega()
        @test kw isa KOmega{Float64}
        @test kw.beta_star == 0.09
        @test kw.alpha == 5.0 / 9.0
        @test kw.beta == 3.0 / 40.0
        @test kw.sigma_k == 0.5
        @test kw.sigma_omega == 0.5
        @test n_turbulence_fields(kw) == 2
        @test turbulence_field_names(kw) == (:k, :omega)

        # KOmegaSSTModel
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        sst = KOmegaSSTModel(mesh, [:bottom, :top])
        @test sst isa KOmegaSSTModel{Float64}
        @test length(sst.d_wall) == 16
        @test all(isfinite, sst.d_wall)
        @test all(>(0), sst.d_wall)
        @test n_turbulence_fields(sst) == 2
        @test turbulence_field_names(sst) == (:k, :omega)

        # SpalartAllmaras
        sa = SpalartAllmaras(mesh, [:bottom, :top])
        @test sa isa SpalartAllmaras{Float64}
        @test sa.cb1 == 0.1355
        @test sa.cv1 == 7.1
        @test length(sa.d_wall) == 16
        @test n_turbulence_fields(sa) == 1
        @test turbulence_field_names(sa) == (:nu_tilde,)
    end

    # ── 2. RANSTurbulenceState construction ───────────────────────────
    @testset "RANSTurbulenceState construction" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        # k-ε state
        ke = StandardKEpsilon()
        ts_ke = RANSTurbulenceState(ke, mesh)
        @test haskey(ts_ke.fields, :k)
        @test haskey(ts_ke.fields, :epsilon)
        @test length(ts_ke.fields[:k].internal) == nc
        @test length(ts_ke.nu_t) == nc
        @test all(==(1.0e-6), ts_ke.fields[:k].internal)

        # k-ω state with custom initial values
        kw = KOmega()
        ts_kw = RANSTurbulenceState(kw, mesh; k = 0.01, omega = 100.0)
        @test haskey(ts_kw.fields, :k)
        @test haskey(ts_kw.fields, :omega)
        @test all(==(0.01), ts_kw.fields[:k].internal)
        @test all(==(100.0), ts_kw.fields[:omega].internal)

        # SA state
        sa = SpalartAllmaras(mesh, [:bottom])
        ts_sa = RANSTurbulenceState(sa, mesh; nu_tilde = 1.0e-4)
        @test haskey(ts_sa.fields, :nu_tilde)
        @test length(ts_sa.fields) == 1
        @test all(==(1.0e-4), ts_sa.fields[:nu_tilde].internal)
    end

    # ── 3. compute_nu_eff ─────────────────────────────────────────────
    @testset "compute_nu_eff" begin
        nu = 1.0e-3
        nu_t = [0.01, 0.02, 0.05, 0.1]
        nu_eff = compute_nu_eff(nu, nu_t)
        @test length(nu_eff) == 4
        for c in 1:4
            @test nu_eff[c] ≈ nu + nu_t[c]
        end
    end

    # ── 4. compute_strain_rate ────────────────────────────────────────
    @testset "compute_strain_rate" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        # Uniform velocity field U = (1, 0) → zero strain rate
        U_uniform = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
        S_uniform = compute_strain_rate(U_uniform, mesh)
        @test length(S_uniform) == nc
        @test all(isfinite, S_uniform)
        @test all(s -> s < 0.1, S_uniform)

        # Linear velocity field: Ux = y → dudy = 1, nonzero strain rate
        U_linear = CollocatedVectorField(:U, mesh)
        for c in 1:nc
            y_c = mesh.cell_centers[2, c]
            U_linear.internal[c] = SVector(y_c, 0.0)
        end
        # Set boundary values too (needed for gradient computation)
        for (i, f) in enumerate(U_linear.boundary_face_indices)
            y_f = mesh.face_centers[2, f]
            U_linear.boundary[i] = SVector(y_f, 0.0)
        end
        S_linear = compute_strain_rate(U_linear, mesh)
        @test all(isfinite, S_linear)
        @test all(>(0), S_linear)
    end

    # ── 5. compute_wall_distance ──────────────────────────────────────
    @testset "compute_wall_distance" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        d_wall = compute_wall_distance(mesh, [:bottom])

        # Bottom row cells (j=1): centers at y = 0.125
        # Top row cells (j=4): centers at y = 0.875
        # Distance to bottom face (y=0) should be smaller for bottom row
        nx = 4
        for i in 1:nx
            bottom_cell = i           # j=1
            top_cell = (4 - 1) * nx + i  # j=4
            @test d_wall[bottom_cell] < d_wall[top_cell]
        end
        @test all(isfinite, d_wall)
        @test all(>(0), d_wall)
    end

    # ── 6. k-ε turbulent_viscosity! ──────────────────────────────────
    @testset "k-ε turbulent_viscosity!" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        ke = StandardKEpsilon()

        ts = RANSTurbulenceState(ke, mesh; k = 0.1, epsilon = 0.01)
        nu_t = zeros(Float64, nc)
        turbulent_viscosity!(nu_t, ke, ts, mesh)

        expected = ke.C_mu * 0.1^2 / 0.01
        for c in 1:nc
            @test nu_t[c] ≈ expected
        end
    end

    # ── 7. k-ε solve_turbulence! smoke ────────────────────────────────
    @testset "k-ε solve_turbulence! smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        ke = StandardKEpsilon()

        ts = RANSTurbulenceState(ke, mesh; k = 1.0e-4, epsilon = 1.0e-5)
        turbulent_viscosity!(ts.nu_t, ke, ts, mesh)

        U = CollocatedVectorField(:U, mesh; value = SVector(0.1, 0.0))
        # Set boundary values for gradient computation
        for (i, _) in enumerate(U.boundary_face_indices)
            U.boundary[i] = SVector(0.1, 0.0)
        end
        phi = FaceFluxField(:phi, mesh; value = 0.0)

        bcs_turb = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
            :k => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(1.0e-4),
                :right => ParabolicNeumann(0.0),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
            :epsilon => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(1.0e-5),
                :right => ParabolicNeumann(0.0),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
        )

        solve_turbulence!(ts, ke, U, phi, 1.0e-3, mesh, bcs_turb)

        @test all(isfinite, ts.fields[:k].internal)
        @test all(isfinite, ts.fields[:epsilon].internal)
        @test all(>(0), ts.fields[:k].internal)
        @test all(>(0), ts.fields[:epsilon].internal)
    end

    # ── 8. solve_simple_turbulent smoke ───────────────────────────────
    @testset "solve_simple_turbulent smoke" begin
        mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE(; max_iterations = 5, tolerance = 1.0e-12)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.01)

        ke = StandardKEpsilon()
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
            :k => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(1.0e-4),
                :right => ParabolicNeumann(0.0),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
            :epsilon => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(1.0e-5),
                :right => ParabolicNeumann(0.0),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
        )
        result, turb_state = solve_simple_turbulent(
            prob, ke; turb_bcs = turb_bcs,
        )

        @test result isa SolveResult{2, Float64}
        @test turb_state isa RANSTurbulenceState{Float64}
        @test result.iterations == 5
        @test all(isfinite, turb_state.fields[:k].internal)
        @test all(isfinite, turb_state.fields[:epsilon].internal)
        @test all(isfinite, turb_state.nu_t)
    end

    # ── 9. KOmega turbulent_viscosity! ────────────────────────────────
    @testset "KOmega turbulent_viscosity!" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        kw = KOmega()

        ts = RANSTurbulenceState(kw, mesh; k = 0.05, omega = 50.0)
        nu_t = zeros(Float64, nc)
        turbulent_viscosity!(nu_t, kw, ts, mesh)

        expected = 0.05 / 50.0
        for c in 1:nc
            @test nu_t[c] ≈ expected
        end
    end

    # ── 10. turbulence_inlet_bc ───────────────────────────────────────
    @testset "turbulence_inlet_bc" begin
        U_mag = 1.0
        intensity = 0.05
        length_scale = 0.01

        # k-ε
        ke = StandardKEpsilon()
        bc_ke = turbulence_inlet_bc(ke, U_mag, intensity, length_scale)
        @test haskey(bc_ke, :k)
        @test haskey(bc_ke, :epsilon)
        @test bc_ke[:k] isa ParabolicDirichlet
        @test bc_ke[:epsilon] isa ParabolicDirichlet
        @test bc_ke[:k].value > 0
        @test bc_ke[:epsilon].value > 0

        # k-ω
        kw = KOmega()
        bc_kw = turbulence_inlet_bc(kw, U_mag, intensity, length_scale)
        @test haskey(bc_kw, :k)
        @test haskey(bc_kw, :omega)
        @test bc_kw[:k] isa ParabolicDirichlet
        @test bc_kw[:omega] isa ParabolicDirichlet
        @test bc_kw[:k].value > 0
        @test bc_kw[:omega].value > 0

        # SST
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        sst = KOmegaSSTModel(mesh, [:bottom])
        bc_sst = turbulence_inlet_bc(sst, U_mag, intensity, length_scale)
        @test haskey(bc_sst, :k)
        @test haskey(bc_sst, :omega)
        @test bc_sst[:k] isa ParabolicDirichlet
        @test bc_sst[:omega] isa ParabolicDirichlet

        # SA
        sa = SpalartAllmaras(mesh, [:bottom])
        bc_sa = turbulence_inlet_bc(sa, U_mag, intensity, length_scale)
        @test haskey(bc_sa, :nu_tilde)
        @test bc_sa[:nu_tilde] isa ParabolicDirichlet
        @test bc_sa[:nu_tilde].value > 0
    end

    # ── 11. turbulence_wall_bc ────────────────────────────────────────
    @testset "turbulence_wall_bc" begin
        # k-ε
        ke = StandardKEpsilon()
        wbc_ke = turbulence_wall_bc(ke)
        @test haskey(wbc_ke, :k)
        @test haskey(wbc_ke, :epsilon)
        @test wbc_ke[:k] isa ParabolicNeumann
        @test wbc_ke[:epsilon] isa ParabolicNeumann

        # k-ω
        kw = KOmega()
        wbc_kw = turbulence_wall_bc(kw)
        @test haskey(wbc_kw, :k)
        @test haskey(wbc_kw, :omega)
        @test wbc_kw[:k] isa ParabolicNeumann
        @test wbc_kw[:omega] isa ParabolicNeumann

        # SST (dispatches via Union{KOmega, KOmegaSSTModel})
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        sst = KOmegaSSTModel(mesh, [:bottom])
        wbc_sst = turbulence_wall_bc(sst)
        @test haskey(wbc_sst, :k)
        @test haskey(wbc_sst, :omega)

        # SA
        sa = SpalartAllmaras(mesh, [:bottom])
        wbc_sa = turbulence_wall_bc(sa)
        @test haskey(wbc_sa, :nu_tilde)
        @test wbc_sa[:nu_tilde] isa ParabolicDirichlet
        @test wbc_sa[:nu_tilde].value == 0.0
    end
end

using FiniteVolumeMethod
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# -- Mesh builder (copied from radiation.jl -- safe_include isolates modules) --

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

    # Internal vertical faces
    for j in 1:ny, i in 1:(nx - 1)
        P = cell_idx(i, j)
        N = cell_idx(i + 1, j)
        push!(face_cells_list, (P, N))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (i * dx, (j - 0.5) * dy))
        push!(face_tags_list, :internal)
    end

    # Internal horizontal faces
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

# ===========================================================================
# 1. FvDOM Radiation
# ===========================================================================

@testset "fvDOM Radiation" begin

    # -- 1a. FvDOMModel construction: S2 has 4 directions in 2D ----------------
    @testset "FvDOMModel construction (2D)" begin
        model = FvDOMModel(; a = 0.2, Dim = 2)
        @test model isa FvDOMModel{2, Float64}
        @test model.a == 0.2
        @test length(model.directions) == 4
        @test length(model.weights) == 4
        # All weights equal pi/2
        @test all(w -> w ≈ pi / 2, model.weights)
        # Directions are unit vectors
        for d in model.directions
            @test norm(d) ≈ 1.0 atol = 1.0e-14
        end
    end

    @testset "FvDOMModel construction (3D)" begin
        model = FvDOMModel(; a = 0.5, Dim = 3)
        @test model isa FvDOMModel{3, Float64}
        @test length(model.directions) == 8
        @test length(model.weights) == 8
        @test all(w -> w ≈ pi / 2, model.weights)
        for d in model.directions
            @test norm(d) ≈ 1.0 atol = 1.0e-14
        end
    end

    # -- 1b. solve_fvdom_radiation smoke: uniform T, returns positive G --------
    @testset "solve_fvdom_radiation smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        model = FvDOMModel(; a = 1.0, Dim = 2)
        T_val = 300.0
        T_field = CollocatedScalarField(:T, mesh; value = T_val)

        # Dirichlet BCs: intensity = sigma*T^4/pi on all walls
        I_bc_val = STEFAN_BOLTZMANN * T_val^4 / pi
        bcs_G = Dict{Symbol, AbstractBoundaryCondition}(
            :left => ParabolicDirichlet(I_bc_val),
            :right => ParabolicDirichlet(I_bc_val),
            :bottom => ParabolicDirichlet(I_bc_val),
            :top => ParabolicDirichlet(I_bc_val),
        )

        rad_state = solve_fvdom_radiation(model, T_field, mesh, bcs_G)

        @test rad_state isa RadiationState{Float64}
        @test length(rad_state.G.internal) == nc
        # G should be positive everywhere
        @test all(>(0), rad_state.G.internal)
        # G should be finite
        @test all(isfinite, rad_state.G.internal)
    end

    # -- 1c. compute_radiation_source (fvDOM dispatch) -------------------------
    @testset "compute_radiation_source (fvDOM)" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        model = FvDOMModel(; a = 0.5, Dim = 2)
        T_val = 400.0
        sigma = STEFAN_BOLTZMANN

        T_field = CollocatedScalarField(:T, mesh; value = T_val)
        G_eq = 4.0 * sigma * T_val^4
        G_field = CollocatedScalarField(:G, mesh; value = G_eq)

        S_rad = compute_radiation_source(model, G_field, T_field)
        @test length(S_rad) == nc
        # At equilibrium: S_rad ≈ 0
        for c in 1:nc
            @test abs(S_rad[c]) < 1.0e-6
        end
    end
end

# ===========================================================================
# 2. Spray Breakup Models
# ===========================================================================

@testset "Spray Breakup Models" begin

    # -- 2a. TABBreakup: We < crit -> no breakup, We > crit -> breakup --------
    @testset "TABBreakup threshold" begin
        tab = TABBreakup(; We_crit = 12.0)
        @test tab isa TABBreakup{Float64}
        @test tab.We_crit == 12.0
        @test tab.C_b == 0.5

        @test !should_breakup(tab, 10.0)
        @test !should_breakup(tab, 12.0)  # not strictly greater
        @test should_breakup(tab, 13.0)
        @test should_breakup(tab, 100.0)
    end

    # -- 2b. KHRTBreakup construction -----------------------------------------
    @testset "KHRTBreakup construction" begin
        khrt = KHRTBreakup()
        @test khrt isa KHRTBreakup{Float64}
        @test khrt.B0 == 0.61
        @test khrt.B1 == 10.0

        khrt2 = KHRTBreakup(; B0 = 0.5, B1 = 20.0)
        @test khrt2.B0 == 0.5
        @test khrt2.B1 == 20.0
    end

    # -- 2c. weber_number known values -----------------------------------------
    @testset "weber_number" begin
        # SVector version
        U_rel = SVector(10.0, 0.0)
        d = 0.001
        rho_f = 1.2
        sigma = 0.072
        We = weber_number(U_rel, d, rho_f, sigma)
        expected = 1.2 * 100.0 * 0.001 / 0.072  # 1.666...
        @test We ≈ expected

        # Scalar version
        We_scalar = weber_number(10.0, d, rho_f, sigma)
        @test We_scalar ≈ expected
    end

    # -- 2d. breakup_diameter --------------------------------------------------
    @testset "breakup_diameter" begin
        tab = TABBreakup(; We_crit = 12.0)
        d_parent = 0.001
        We = 48.0  # 4x critical
        d_child = breakup_diameter(tab, d_parent, We)
        @test d_child ≈ d_parent * (12.0 / 48.0)^(1.0 / 3.0)
        @test d_child < d_parent
    end

    # -- 2e. apply_breakup! ---------------------------------------------------
    @testset "apply_breakup!" begin
        tracker = ParticleTracker{2, Float64}()
        pos = SVector(0.5, 0.5)
        vel = SVector(50.0, 0.0)  # high velocity for breakup
        inject_particles!(tracker, [pos])

        # Set up particle properties
        p = tracker.particles[1]
        p.velocity = vel
        set_particle_properties!(p; diameter = 0.001, density = 1000.0)

        n_before = length(tracker.particles)

        tab = TABBreakup(; We_crit = 12.0)
        rho_f = 1.2
        sigma_st = 0.072

        # Check We: 1.2 * 2500 * 0.001 / 0.072 = 41.67 > 12
        apply_breakup!(tracker, tab, rho_f, sigma_st)

        # A new particle should have been added
        @test length(tracker.particles) > n_before
        # Both parent and child should have reduced diameter
        @test tracker.particles[1].properties[:diameter] < 0.001
    end
end

# ===========================================================================
# 3. EDC Combustion
# ===========================================================================

@testset "EDC Combustion" begin

    # -- 3a. EddyDissipationConcept construction (already exists) ---------------
    @testset "EddyDissipationConcept construction" begin
        edc = EddyDissipationConcept()
        @test edc isa EddyDissipationConcept{Float64}
        @test edc.C_gamma ≈ 2.1377
        @test edc.C_tau ≈ 0.4082
    end

    # -- 3b. compute_edc_reaction_rates with known k/eps -----------------------
    @testset "compute_edc_reaction_rates with turbulence" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        edc = EddyDissipationConcept()
        props = CombustionProperties()
        species = SpeciesState(mesh, props; fuel = 0.05, oxidizer = 0.23, product = 0.0)

        k_field = fill(1.0, nc)
        eps_field = fill(10.0, nc)
        density = 1.2
        nu = 1.5e-5

        rates = compute_edc_reaction_rates(
            edc, species, props, k_field, eps_field, density, nu, mesh,
        )

        @test rates isa NTuple{3, Vector{Float64}}
        @test length(rates[1]) == nc

        # Fuel should be consumed (negative rate)
        @test all(<=(0), rates[1])
        # Oxidizer should be consumed (negative, same sign as fuel * s)
        @test all(<=(0), rates[2])
        # Product should be produced (positive)
        @test all(>=(0), rates[3])

        # Stoichiometric consistency: omega_ox = s * omega_fuel
        s = props.stoich_ratio
        for c in 1:nc
            @test rates[2][c] ≈ s * rates[1][c]
        end
    end

    # -- 3c. EDC fallback (no turbulence) --------------------------------------
    @testset "compute_edc_reaction_rates fallback" begin
        mesh = build_cartesian_unstructured_mesh(2, 2, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        edc = EddyDissipationConcept()
        props = CombustionProperties()
        species = SpeciesState(mesh, props; fuel = 0.05, oxidizer = 0.23)

        rates = compute_edc_reaction_rates(
            edc, species, props, nothing, nothing, 1.2, 1.5e-5, mesh,
        )

        @test rates isa NTuple{3, Vector{Float64}}
        # Should still produce valid rates via fallback
        @test all(isfinite, rates[1])
        @test all(<=(0), rates[1])
    end
end

# ===========================================================================
# 4. Cyclic BC Assembly
# ===========================================================================

@testset "Cyclic BC Assembly" begin

    # -- 4a. match_cyclic_faces: simple 4x4 mesh, match :left to :right --------
    @testset "match_cyclic_faces" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)

        pairs = match_cyclic_faces(mesh, :left, :right)

        @test length(pairs) == 4  # 4 rows of cells = 4 face pairs

        # Each pair should have valid face indices
        nf = size(mesh.face_cells, 2)
        for (f1, f2) in pairs
            @test 1 <= f1 <= nf
            @test 1 <= f2 <= nf
            # f1 should be on :left, f2 on :right
            @test mesh.face_cells[2, f1] == 0  # boundary
            @test mesh.face_cells[2, f2] == 0  # boundary
        end

        # Face centers should match in y-coordinate (after accounting for offset)
        for (f1, f2) in pairs
            y1 = mesh.face_centers[2, f1]
            y2 = mesh.face_centers[2, f2]
            @test y1 ≈ y2 atol = 1.0e-10
        end
    end

    # -- 4b. match_cyclic_faces: bottom to top ---------------------------------
    @testset "match_cyclic_faces bottom-top" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        pairs = match_cyclic_faces(mesh, :bottom, :top)
        @test length(pairs) == 4

        # x-coordinates should match
        for (f1, f2) in pairs
            x1 = mesh.face_centers[1, f1]
            x2 = mesh.face_centers[1, f2]
            @test x1 ≈ x2 atol = 1.0e-10
        end
    end

    # -- 4c. apply_cyclic_bc! modifies equation --------------------------------
    @testset "apply_cyclic_bc!" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        pairs = match_cyclic_faces(mesh, :left, :right)

        eq = CollocatedEquation(mesh)
        field = CollocatedScalarField(:phi, mesh; value = 1.0)

        # Record sparsity before
        nnz_before = nnz(eq.A)

        apply_cyclic_bc!(eq, field, mesh, pairs)

        # Equation should have new off-diagonal entries
        @test nnz(eq.A) > nnz_before

        # For each pair, the coupled cells should have nonzero cross-entries
        for (f1, f2) in pairs
            c1 = mesh.face_cells[1, f1]
            c2 = mesh.face_cells[1, f2]
            @test eq.A[c1, c2] != 0.0
            @test eq.A[c2, c1] != 0.0
            # Diagonal should also be increased
            @test eq.A[c1, c1] > 0.0
            @test eq.A[c2, c2] > 0.0
        end
    end
end

# ===========================================================================
# 5. Crank-Nicolson Temporal Discretization
# ===========================================================================

@testset "Crank-Nicolson ddt" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    # -- 5a. assemble_ddt_crank_nicolson! matches Euler for the temporal part --
    @testset "C-N ddt matches Euler temporal contribution" begin
        phi_old = rand(nc)
        rho = 1.2
        dt = 0.01

        eq_euler = CollocatedEquation(mesh)
        assemble_ddt_euler!(eq_euler, rho, phi_old, mesh, dt)

        eq_cn = CollocatedEquation(mesh)
        assemble_ddt_crank_nicolson!(eq_cn, rho, phi_old, mesh, dt)

        # Temporal mass-matrix part is identical
        @test eq_cn.A ≈ eq_euler.A
        @test eq_cn.b ≈ eq_euler.b
    end

    # -- 5b. Diagonal and RHS values are correct --
    @testset "C-N ddt diagonal and RHS values" begin
        phi_old = ones(nc)
        rho = 2.0
        dt = 0.05

        eq = CollocatedEquation(mesh)
        assemble_ddt_crank_nicolson!(eq, rho, phi_old, mesh, dt)

        for c in 1:nc
            expected_coeff = rho * mesh.cell_volumes[c] / dt
            @test eq.A[c, c] ≈ expected_coeff
            @test eq.b[c] ≈ expected_coeff * phi_old[c]
        end
    end

    # -- 5c. Per-cell density vector works --
    @testset "C-N ddt with per-cell density" begin
        phi_old = rand(nc)
        rho_field = rand(nc) .+ 0.5  # positive densities
        dt = 0.02

        eq = CollocatedEquation(mesh)
        assemble_ddt_crank_nicolson!(eq, rho_field, phi_old, mesh, dt)

        for c in 1:nc
            expected_coeff = rho_field[c] * mesh.cell_volumes[c] / dt
            @test eq.A[c, c] ≈ expected_coeff
            @test eq.b[c] ≈ expected_coeff * phi_old[c]
        end
    end

    # -- 5d. Unified assemble_ddt! dispatches to C-N --
    @testset "assemble_ddt! dispatches TIME_CRANK_NICOLSON" begin
        phi_old = rand(nc)
        rho = 1.0
        dt = 0.01

        eq_dispatch = CollocatedEquation(mesh)
        assemble_ddt!(eq_dispatch, rho, phi_old, mesh, dt; scheme = TIME_CRANK_NICOLSON)

        eq_direct = CollocatedEquation(mesh)
        assemble_ddt_crank_nicolson!(eq_direct, rho, phi_old, mesh, dt)

        @test eq_dispatch.A ≈ eq_direct.A
        @test eq_dispatch.b ≈ eq_direct.b
    end
end

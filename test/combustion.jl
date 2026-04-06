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

Boundary face tags: `:left`, `:right`, `:bottom`, `:top`.
Internal face tags: `:internal`.
"""
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

    # Boundary: left
    for j in 1:ny
        P = cell_idx(1, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (-1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (0.0, (j - 0.5) * dy))
        push!(face_tags_list, :left)
    end

    # Boundary: right
    for j in 1:ny
        P = cell_idx(nx, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (Lx, (j - 0.5) * dy))
        push!(face_tags_list, :right)
    end

    # Boundary: bottom
    for i in 1:nx
        P = cell_idx(i, 1)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (0.0, -1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, 0.0))
        push!(face_tags_list, :bottom)
    end

    # Boundary: top
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

# -- Tests --

@testset "Combustion & Species Transport (Phase 8)" begin

    # -- 1. CombustionProperties defaults --------------------------------
    @testset "CombustionProperties defaults" begin
        cp = CombustionProperties()
        @test cp isa CombustionProperties{3, Float64}
        @test cp.species_names == (:fuel, :oxidizer, :product)
        @test cp.molecular_weights == (16.0, 32.0, 44.0)
        @test cp.diffusivities == (2.0e-5, 2.0e-5, 2.0e-5)
        @test cp.Sc_t == 0.7
        @test cp.stoich_ratio == 4.0
        @test cp.heat_of_combustion == 50.0e6
    end

    # -- 2. SpeciesState construction -------------------------------------
    @testset "SpeciesState construction" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        cp = CombustionProperties()
        ss = SpeciesState(mesh, cp; fuel = 1.0, oxidizer = 0.233)
        @test ss isa SpeciesState{3, Float64}
        @test length(ss.Y) == 3
        @test ss.Y[1].name == :fuel
        @test all(==(1.0), ss.Y[1].internal)
        @test ss.Y[2].name == :oxidizer
        @test all(==(0.233), ss.Y[2].internal)
        @test ss.Y[3].name == :product
        @test all(==(0.0), ss.Y[3].internal)
    end

    # -- 3. EddyDissipationModel defaults ---------------------------------
    @testset "EddyDissipationModel defaults" begin
        edm = EddyDissipationModel()
        @test edm isa EddyDissipationModel{Float64}
        @test edm.A_edm == 4.0
        @test edm.B_edm == 0.5
    end

    # -- 4. assemble_species! smoke ---------------------------------------
    @testset "assemble_species! smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        Y_field = CollocatedScalarField(:fuel, mesh; value = 0.5)
        phi = FaceFluxField(:phi, mesh; value = 0.01)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => ParabolicDirichlet(1.0),
            :right => ParabolicNeumann(0.0),
            :bottom => ParabolicNeumann(0.0),
            :top => ParabolicNeumann(0.0),
        )

        eq = CollocatedEquation(mesh)
        assemble_species!(eq, Y_field, phi, 1.0e-4, mesh, bcs)

        # Matrix should have nonzero entries after assembly
        @test nnz(eq.A) > 0
    end

    # -- 5. compute_edm_reaction_rates ------------------------------------
    @testset "compute_edm_reaction_rates" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        cp = CombustionProperties()
        edm = EddyDissipationModel()

        # Set up species: fuel=1.0, oxidizer=0.233, product=0.0
        ss = SpeciesState(mesh, cp; fuel = 1.0, oxidizer = 0.233)

        # With turbulence fields
        k_field = fill(1.0, nc)
        eps_field = fill(0.1, nc)
        density = 1.2

        omega = compute_edm_reaction_rates(
            edm, ss, cp, k_field, eps_field, density, mesh,
        )

        @test length(omega) == 3
        @test length(omega[1]) == nc
        # Fuel consumed (negative)
        @test all(<(0), omega[1])
        # Oxidizer consumed (negative)
        @test all(<(0), omega[2])
        # Product formed (positive)
        @test all(>(0), omega[3])
        # All rates finite
        @test all(isfinite, omega[1])
        @test all(isfinite, omega[2])
        @test all(isfinite, omega[3])

        # Without turbulence (fallback mixing time)
        omega_fb = compute_edm_reaction_rates(
            edm, ss, cp, nothing, nothing, density, mesh,
        )
        @test all(isfinite, omega_fb[1])
        @test all(<(0), omega_fb[1])
    end

    # -- 6. compute_heat_release ------------------------------------------
    @testset "compute_heat_release" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        cp = CombustionProperties()
        edm = EddyDissipationModel()
        ss = SpeciesState(mesh, cp; fuel = 1.0, oxidizer = 0.233)
        density = 1.2

        omega = compute_edm_reaction_rates(
            edm, ss, cp, nothing, nothing, density, mesh,
        )

        S_h = compute_heat_release(omega, cp)
        @test length(S_h) == nc
        # Heat release should be positive (exothermic, ω_fuel < 0, ΔH > 0)
        @test all(>(0), S_h)
        @test all(isfinite, S_h)
    end

    # -- 7. solve_species! smoke ------------------------------------------
    @testset "solve_species! smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        cp = CombustionProperties()
        ss = SpeciesState(mesh, cp; fuel = 0.5, oxidizer = 0.2)
        phi = FaceFluxField(:phi, mesh; value = 0.01)

        # Zero reaction rates
        reaction_rates = ntuple(_ -> zeros(Float64, nc), Val(3))
        density = 1.0

        bcs_species = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
            :fuel => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(1.0),
                :right => ParabolicNeumann(0.0),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
            :oxidizer => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(0.0),
                :right => ParabolicNeumann(0.0),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
            :product => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(0.0),
                :right => ParabolicNeumann(0.0),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
        )

        solve_species!(
            ss, phi, cp, reaction_rates,
            nothing, density, mesh, bcs_species,
        )

        for i in 1:3
            @test all(isfinite, ss.Y[i].internal)
            @test all(y -> 0.0 <= y <= 1.0, ss.Y[i].internal)
        end
    end

    # -- 8. solve_simple_reacting smoke -----------------------------------
    @testset "solve_simple_reacting smoke" begin
        mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE(; max_iterations = 3, tolerance = 1.0e-12)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        thermal_props = FluidThermalProperties{2}(; k = 0.6, Cp = 4000.0)
        cp = CombustionProperties()
        edm = EddyDissipationModel()

        bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
            :left => thermal_inlet_bc(350.0),
            :right => thermal_insulated_bc(),
            :bottom => thermal_inlet_bc(300.0),
            :top => thermal_inlet_bc(300.0),
        )

        bcs_species = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
            :fuel => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(1.0),
                :right => ParabolicNeumann(0.0),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
            :oxidizer => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(0.0),
                :right => ParabolicDirichlet(0.233),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
            :product => Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(0.0),
                :right => ParabolicNeumann(0.0),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            ),
        )

        result, thermal_state, species_state = solve_simple_reacting(
            prob, thermal_props, cp, edm;
            bcs_T = bcs_T,
            bcs_species = bcs_species,
            Y_init = Dict(:fuel => 0.5, :oxidizer => 0.1),
        )

        @test result isa SolveResult{2, Float64}
        @test thermal_state isa ThermalState{Float64}
        @test species_state isa SpeciesState{3, Float64}
        @test result.iterations == 3
        @test all(isfinite, thermal_state.T_field.internal)
        for i in 1:3
            @test all(isfinite, species_state.Y[i].internal)
            @test all(y -> 0.0 <= y <= 1.0, species_state.Y[i].internal)
        end
    end

end

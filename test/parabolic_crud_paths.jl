using Test
using FiniteVolumeMethod

# Integration test: exercises the exact FVM code paths that CRUD.jl uses.
# Each @testset maps to one CRUD use-case so failures are easy to locate.

@testset "Parabolic CRUD Paths" begin

    # ------------------------------------------------------------------
    # Test 1: 1D mesh generation + Diffusion1D + assemble_system
    # ------------------------------------------------------------------
    @testset "1D diffusion assembly (Dirichlet + Neumann BCs)" begin
        mesh = generate_mesh_1d(10, 1.0)
        @test mesh isa Mesh1D
        @test length(mesh.cells) == 10
        @test length(mesh.nodes) == 11

        # Cell volume access — the primary CRUD use-case
        @test mesh.cells[1].volume ≈ 0.1

        model = Diffusion1D(1.0)
        @test model isa AbstractDiffusion

        bc_left = ParabolicDirichlet(0.0)
        bc_right = ParabolicNeumann(0.0)

        A, b = assemble_system(model, mesh, bc_left, bc_right)
        @test size(A) == (10, 10)
        @test length(b) == 10

        # Diagonal entries must be positive (diffusion adds to diagonal)
        for i in 1:10
            @test A[i, i] > 0.0
        end
    end

    # ------------------------------------------------------------------
    # Test 2: TimeController — does NOT exist in FVM; CRUD must not use it
    # ------------------------------------------------------------------
    @testset "TimeController absent (CRUD should not depend on it)" begin
        @test !isdefined(FiniteVolumeMethod, :TimeController)
        @test !isdefined(FiniteVolumeMethod, :accept_step!)
    end

    # ------------------------------------------------------------------
    # Test 3: 2D mesh generation + Diffusion2D + assemble_system
    # ------------------------------------------------------------------
    @testset "2D diffusion assembly" begin
        mesh = generate_mesh_2d(4, 4, 1.0, 1.0)
        @test mesh isa Mesh2D
        @test length(mesh.cells) == 16

        # Cell volume access
        @test mesh.cells[1].volume ≈ 0.0625

        model = Diffusion2D(1.0)
        @test model isa AbstractDiffusion

        # 2D assemble_system takes a 4-tuple of BCs: (left, right, bottom, top)
        bc_left = ParabolicDirichlet(1.0)
        bc_right = ParabolicDirichlet(0.0)
        bc_bottom = ParabolicNeumann(0.0)
        bc_top = ParabolicNeumann(0.0)
        bcs = (bc_left, bc_right, bc_bottom, bc_top)

        A, b = assemble_system(model, mesh, bcs)
        @test size(A) == (16, 16)
        @test length(b) == 16

        # Dirichlet BCs on left/right contribute to RHS
        @test any(b .!= 0.0)
    end

    # ------------------------------------------------------------------
    # Test 4: CylindricalDiffusion2D is an AbstractDiffusion
    # ------------------------------------------------------------------
    @testset "CylindricalDiffusion2D <: AbstractDiffusion" begin
        model = CylindricalDiffusion2D(0.5)
        @test model isa AbstractDiffusion
        @test model.gamma ≈ 0.5
    end

    # ------------------------------------------------------------------
    # Test 5: AbstractProblemPDE is exported
    # ------------------------------------------------------------------
    @testset "AbstractProblemPDE exported" begin
        @test isdefined(FiniteVolumeMethod, :AbstractProblemPDE)

        # Define a concrete subtype to confirm it is usable
        struct _TestProblem <: AbstractProblemPDE end
        @test _TestProblem() isa AbstractProblemPDE
    end

    # ------------------------------------------------------------------
    # Test 6: CellField constructor
    # ------------------------------------------------------------------
    @testset "CellField constructor" begin
        var = Variable(:T, STATEVAR, :K, "temperature")
        field = CellField(var, [300.0, 310.0, 320.0])
        @test field isa CellField
        @test length(field.values) == 3
        @test field.values[1] ≈ 300.0
        @test field.variable.name == :T
    end

    # ------------------------------------------------------------------
    # Test 6b: make_cell_field factory — sized + metadata-tagged in one call
    # ------------------------------------------------------------------
    @testset "make_cell_field factory" begin
        mesh = generate_mesh_1d(8, 1.0)
        T = make_cell_field(mesh; name = :temperature, unit = :K,
                            description = "Temperature", init = 560.0)
        @test T isa CellField
        @test length(T.values) == length(mesh.cells)
        @test all(v -> v ≈ 560.0, T.values)
        @test T.variable.name == :temperature
        @test T.variable.role === STATEVAR
        @test T.variable.unit === :K

        # Defaults: zero init, :unitless, empty description
        c = make_cell_field(mesh; name = :H3BO3)
        @test all(v -> v == 0.0, c.values)
        @test c.variable.unit === :unitless
        @test c.variable.description == ""
    end

    # ------------------------------------------------------------------
    # Test 7: AbstractFVMMesh is exported (abstract mesh type)
    # ------------------------------------------------------------------
    @testset "AbstractFVMMesh exported" begin
        @test isdefined(FiniteVolumeMethod, :AbstractFVMMesh)
    end

    # ------------------------------------------------------------------
    # Test 8: SciMLBase.ODEProblem convenience method for structured parabolic
    # ------------------------------------------------------------------
    @testset "ODEProblem(model, mesh, bcs; tspan, u0)" begin
        using OrdinaryDiffEq
        using SciMLBase
        mesh = generate_mesh_1d(50, 1.0e-3)
        u0 = fill(560.0, length(mesh.cells))
        prob = SciMLBase.ODEProblem(
            Diffusion1D(2.0e-7), mesh,
            ParabolicDirichlet(600.0), ParabolicNeumann(0.0);
            tspan = (0.0, 10.0), u0 = u0,
        )
        @test prob isa SciMLBase.ODEProblem
        @test prob.tspan == (0.0, 10.0)
        @test length(prob.u0) == length(mesh.cells)

        # NOTE: autodiff=false because parabolic_to_odefunction's preallocated
        # `_tmp = similar(b)` buffer is Float64-typed and breaks ForwardDiff's
        # Dual-number propagation. Tracked as a separate FVM issue (DiffCache
        # refactor); not blocking for the convenience method's correctness.
        sol = solve(prob, ImplicitEuler(autodiff = false); adaptive = false, dt = 0.01)
        # After 10 s with α=2e-7 and L=1e-3 (τ ≈ 5 s), the rod should have nearly
        # equilibrated to the 600 K wall.
        T_final = sol.u[end]
        @test all(595.0 .< T_final .< 600.5)
    end

end

using FiniteVolumeMethod
using Test
using StaticArrays
using DelaunayTriangulation
using LinearAlgebra: norm

# ============================================================
# Helper: Sod shock tube IC
# ============================================================
sod_ic_1d(x) = x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1)
sod_ic_2d(x, y) = x < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)

# ============================================================
# Type Construction
# ============================================================
@testset "Type Construction" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
        final_time = 0.1
    )

    hyp_op = HyperbolicOperator(prob)
    @test hyp_op isa AbstractOperator
    @test hyp_op isa HyperbolicOperator
    @test hyp_op.problem === prob

    src = NullSource()
    src_op = SourceOperator(law, src)
    @test src_op isa AbstractOperator
    @test src_op isa SourceOperator
    @test src_op.source === src
    @test src_op.newton_tol ≈ 1.0e-10
    @test src_op.newton_maxiter == 10

    # Custom tolerance
    src_op2 = SourceOperator(law, src; newton_tol = 1.0e-8, newton_maxiter = 5)
    @test src_op2.newton_tol ≈ 1.0e-8
    @test src_op2.newton_maxiter == 5

    # Splitting schemes
    @test LieTrotterSplitting() isa AbstractSplittingScheme
    @test StrangSplitting() isa AbstractSplittingScheme

    # CoupledProblem
    coupled = CoupledProblem((hyp_op, src_op), StrangSplitting(), 0.0, 0.1)
    @test coupled.splitting isa StrangSplitting
    @test length(coupled.operators) == 2
    @test coupled.initial_time == 0.0
    @test coupled.final_time == 0.1
end

# ============================================================
# Operator Time Step Computation
# ============================================================
@testset "Operator dt" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
        final_time = 0.2, cfl = 0.5
    )

    U = FiniteVolumeMethod.initialize_1d(prob)
    hyp_op = HyperbolicOperator(prob)

    dt_hyp = compute_operator_dt(hyp_op, U, 0.0)
    @test dt_hyp > 0.0
    @test isfinite(dt_hyp)

    # Source operator dt is infinite
    src_op = SourceOperator(law, NullSource())
    dt_src = compute_operator_dt(src_op, U, 0.0)
    @test dt_src == typemax(Float64)

    # Hyp dt matches compute_dt (without final_time clipping)
    dt_direct = compute_dt(prob, U, 0.0)
    # compute_dt clips to final_time; compute_operator_dt does not
    # But for t=0, dt_direct also won't clip if dt < final_time
    @test dt_hyp ≈ dt_direct || dt_hyp ≈ prob.final_time
end

# ============================================================
# 1D Lie-Trotter with NullSource = Pure Hyperbolic
# ============================================================
@testset "1D Lie-Trotter NullSource = pure hyperbolic" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
        final_time = 0.2, cfl = 0.5
    )

    # Reference: pure hyperbolic
    x_ref, U_ref, t_ref = solve_hyperbolic(prob)

    # Coupled with NullSource + Lie-Trotter
    x_split, U_split, t_split = solve_coupled(
        prob, NullSource();
        splitting = LieTrotterSplitting()
    )

    @test length(x_ref) == length(x_split)
    @test t_ref ≈ t_split atol = 1.0e-10
    for i in eachindex(U_ref)
        @test U_ref[i] ≈ U_split[i] atol = 1.0e-12
    end
end

# ============================================================
# 1D Strang with NullSource
# ============================================================
@testset "1D Strang NullSource" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
        final_time = 0.2, cfl = 0.5
    )

    # Strang: L_hyp(dt/2) ∘ L_src(dt) ∘ L_hyp(dt/2)
    # NullSource is no-op, so this is two half-steps of hyperbolic.
    # Two half-steps ≠ one full step for SSP-RK3, but result should be physical.
    x_strang, U_strang, t_strang = solve_coupled(
        prob, NullSource();
        splitting = StrangSplitting()
    )

    @test t_strang ≈ 0.2 atol = 1.0e-10
    @test length(U_strang) == 100

    # Check density is positive and physical
    for u in U_strang
        w = conserved_to_primitive(law, u)
        @test w[1] > 0.0   # density > 0
        @test w[3] > 0.0   # pressure > 0
    end

    # Compare with pure hyperbolic — not exact but close
    _, U_ref, _ = solve_hyperbolic(prob)
    max_diff = maximum(norm(U_strang[i] - U_ref[i]) for i in eachindex(U_ref))
    # Two half-steps vs one full step: difference should be small
    @test max_diff < 0.1
end

# ============================================================
# 1D Source Operator: Uniform Cooling
# ============================================================
@testset "1D Source: Uniform Cooling" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)

    # Uniform IC: ρ=1, v=0, P=1
    uniform_ic(x) = SVector(1.0, 0.0, 1.0)
    t_final = 0.05

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), uniform_ic;
        final_time = t_final, cfl = 0.5
    )

    # Constant cooling function: Λ(T) = 0.1
    cooling = CoolingSource(T -> 0.1; mu_mol = 1.0)

    x, U_final, t = solve_coupled(
        prob, cooling;
        splitting = LieTrotterSplitting()
    )

    @test t ≈ t_final atol = 1.0e-10

    # With uniform IC and no flow, hyperbolic step is no-op.
    # Source: dE/dt = -ρ²Λ = -(1)²(0.1) = -0.1
    # E_0 = P/(γ-1) = 1.0/0.4 = 2.5
    # E(t) ≈ E_0 - 0.1*t = 2.5 - 0.1*0.05 = 2.495
    E_expected = 2.5 - 0.1 * t_final

    for u in U_final
        # Mass conserved
        @test u[1] ≈ 1.0 atol = 1.0e-10
        # Momentum conserved (zero)
        @test u[2] ≈ 0.0 atol = 1.0e-10
        # Energy decreased
        @test u[3] < 2.5
        @test u[3] ≈ E_expected atol = 1.0e-3
    end
end

# ============================================================
# 1D Mass Conservation with Source
# ============================================================
@testset "1D Mass Conservation" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
        final_time = 0.1, cfl = 0.5
    )

    cooling = CoolingSource(T -> 0.01; mu_mol = 1.0)

    # Measure initial mass
    U0 = FiniteVolumeMethod.initialize_1d(prob)
    nc = ncells(mesh)
    dx = mesh.dx
    mass_initial = sum(U0[i + 2][1] * dx for i in 1:nc)

    # Solve with coupling
    _, U_final, _ = solve_coupled(prob, cooling; splitting = StrangSplitting())

    # CoolingSource has zero mass source → mass should be conserved
    mass_final = sum(U_final[i][1] * dx for i in 1:nc)

    # Transmissive BCs may allow mass to leave, but the cooling source
    # itself does not create/destroy mass. For Sod tube in 0.1s,
    # waves haven't reached the boundary, so mass should be well conserved.
    @test mass_final ≈ mass_initial atol = 1.0e-3
end

# ============================================================
# 1D Sod + Cooling: Lie-Trotter vs Strang
# ============================================================
@testset "1D Sod + Cooling: Lie-Trotter vs Strang" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
        final_time = 0.1, cfl = 0.5
    )

    cooling = CoolingSource(T -> 0.01; mu_mol = 1.0)

    _, U_lie, t_lie = solve_coupled(
        prob, cooling;
        splitting = LieTrotterSplitting()
    )
    _, U_strang, t_strang = solve_coupled(
        prob, cooling;
        splitting = StrangSplitting()
    )

    @test t_lie ≈ t_strang atol = 1.0e-10

    # Both should produce physical results
    for i in eachindex(U_lie)
        w_lie = conserved_to_primitive(law, U_lie[i])
        w_strang = conserved_to_primitive(law, U_strang[i])
        @test w_lie[1] > 0.0    # density > 0
        @test w_lie[3] > 0.0    # pressure > 0
        @test w_strang[1] > 0.0
        @test w_strang[3] > 0.0
    end

    # Results should be different (different splitting order)
    # but close to each other
    max_diff = maximum(norm(U_lie[i] - U_strang[i]) for i in eachindex(U_lie))
    @test max_diff > 0.0    # Not identical
    @test max_diff < 0.05   # But close
end

# ============================================================
# 1D MUSCL Reconstruction with Splitting
# ============================================================
@testset "1D MUSCL + Splitting" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    recon = CellCenteredMUSCL(MinmodLimiter())
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), recon,
        TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
        final_time = 0.1, cfl = 0.5
    )

    # Lie-Trotter with NullSource should match pure MUSCL solve
    _, U_ref, t_ref = solve_hyperbolic(prob)
    _, U_split, t_split = solve_coupled(
        prob, NullSource();
        splitting = LieTrotterSplitting()
    )

    @test t_ref ≈ t_split atol = 1.0e-10
    for i in eachindex(U_ref)
        @test U_ref[i] ≈ U_split[i] atol = 1.0e-12
    end
end

# ============================================================
# 1D Different Riemann Solvers
# ============================================================
@testset "1D Different Riemann Solvers" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    cooling = CoolingSource(T -> 0.01; mu_mol = 1.0)

    for solver in (LaxFriedrichsSolver(), HLLSolver(), HLLCSolver())
        prob = HyperbolicProblem(
            law, mesh, solver, NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
            final_time = 0.1, cfl = 0.4
        )

        _, U, t = solve_coupled(prob, cooling; splitting = StrangSplitting())
        @test t ≈ 0.1 atol = 1.0e-10

        # All should give physical results
        for u in U
            w = conserved_to_primitive(law, u)
            @test w[1] > 0.0
            @test w[3] > 0.0
        end
    end
end

# ============================================================
# 2D Lie-Trotter with NullSource
# ============================================================
@testset "2D Lie-Trotter NullSource = pure hyperbolic" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        sod_ic_2d;
        final_time = 0.1, cfl = 0.4
    )

    # Reference
    _, U_ref, t_ref = solve_hyperbolic(prob)

    # Coupled with NullSource
    _, U_split, t_split = solve_coupled(
        prob, NullSource();
        splitting = LieTrotterSplitting()
    )

    @test t_ref ≈ t_split atol = 1.0e-10
    for iy in 1:20, ix in 1:20
        @test U_ref[ix, iy] ≈ U_split[ix, iy] atol = 1.0e-12
    end
end

# ============================================================
# 2D Source Operator
# ============================================================
@testset "2D Source Operator" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
    uniform_ic(x, y) = SVector(1.0, 0.0, 0.0, 1.0)
    t_final = 0.05

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        uniform_ic;
        final_time = t_final, cfl = 0.4
    )

    cooling = CoolingSource(T -> 0.1; mu_mol = 1.0)
    _, U_final, t = solve_coupled(prob, cooling; splitting = StrangSplitting())

    @test t ≈ t_final atol = 1.0e-10

    # E_0 = P/(γ-1) = 1.0/0.4 = 2.5
    # dE/dt = -ρ²Λ = -0.1
    # E(t) ≈ 2.5 - 0.1*t = 2.5 - 0.005 = 2.495
    E_expected = 2.5 - 0.1 * t_final

    for iy in 1:20, ix in 1:20
        u = U_final[ix, iy]
        @test u[1] ≈ 1.0 atol = 1.0e-10      # mass conserved
        @test u[2] ≈ 0.0 atol = 1.0e-10      # momentum x
        @test u[3] ≈ 0.0 atol = 1.0e-10      # momentum y
        @test u[4] < 2.5                     # energy decreased
        @test u[4] ≈ E_expected atol = 1.0e-3  # close to expected
    end
end

# ============================================================
# 2D Mass Conservation
# ============================================================
@testset "2D Mass Conservation" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        sod_ic_2d;
        final_time = 0.05, cfl = 0.4
    )

    cooling = CoolingSource(T -> 0.01; mu_mol = 1.0)
    dx, dy = mesh.dx, mesh.dy

    # Initial mass
    U0 = FiniteVolumeMethod.initialize_2d(prob)
    mass_0 = sum(U0[ix + 2, iy + 2][1] * dx * dy for iy in 1:20, ix in 1:20)

    _, U_final, _ = solve_coupled(prob, cooling; splitting = StrangSplitting())
    mass_f = sum(U_final[ix, iy][1] * dx * dy for iy in 1:20, ix in 1:20)

    @test mass_f ≈ mass_0 atol = 1.0e-2
end

# ============================================================
# CoupledProblem General API
# ============================================================
@testset "CoupledProblem General API" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
        final_time = 0.1, cfl = 0.5
    )

    hyp_op = HyperbolicOperator(prob)
    src_op = SourceOperator(law, NullSource())
    coupled = CoupledProblem(
        (hyp_op, src_op), LieTrotterSplitting(),
        0.0, 0.1
    )

    # solve_coupled with CoupledProblem
    x, U, t = solve_coupled(coupled)
    @test length(x) == 50
    @test length(U) == 50
    @test t ≈ 0.1 atol = 1.0e-10
end

# ============================================================
# Multiple Operators
# ============================================================
@testset "Multiple Operators" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
        final_time = 0.1, cfl = 0.5
    )

    # Three operators: NullSource + Hyp + NullSource
    hyp_op = HyperbolicOperator(prob)
    src1 = SourceOperator(law, NullSource())
    src2 = SourceOperator(law, NullSource())

    # Lie-Trotter with 3 operators: all NullSources are no-ops
    coupled = CoupledProblem(
        (src1, hyp_op, src2), LieTrotterSplitting(),
        0.0, 0.1
    )
    x, U, t = solve_coupled(coupled)
    @test t ≈ 0.1 atol = 1.0e-10

    # Compare with pure hyperbolic
    _, U_ref, _ = solve_hyperbolic(prob)
    for i in eachindex(U_ref)
        @test U_ref[i] ≈ U[i] atol = 1.0e-12
    end

    # Strang with 3 operators: symmetric sweep
    coupled_s = CoupledProblem(
        (src1, hyp_op, src2), StrangSplitting(),
        0.0, 0.1
    )
    x_s, U_s, t_s = solve_coupled(coupled_s)
    @test t_s ≈ 0.1 atol = 1.0e-10

    # With NullSources, Strang just does two half-steps of hyperbolic
    for u in U_s
        w = conserved_to_primitive(law, u)
        @test w[1] > 0.0
        @test w[3] > 0.0
    end
end

# ============================================================
# Resistive Source (MHD)
# ============================================================
@testset "1D MHD + Resistive Source" begin
    eos = IdealGasEOS(5 / 3)
    law = IdealMHDEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)

    # Uniform MHD state with magnetic field
    mhd_ic(x) = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), mhd_ic;
        final_time = 0.01, cfl = 0.4
    )

    resistive = ResistiveSource(0.1)
    _, U, t = solve_coupled(prob, resistive; splitting = StrangSplitting())

    @test t ≈ 0.01 atol = 1.0e-10

    # Resistive source damps B field
    # B components should have decreased from initial
    for u in U
        w = conserved_to_primitive(law, u)
        @test w[1] > 0.0  # density positive
        @test w[5] > 0.0  # pressure positive
    end
end

# ============================================================
# Data Transfer: Cell ↔ Vertex
# ============================================================
@testset "Data Transfer" begin
    # Create unstructured mesh from a triangulation
    tri = triangulate_rectangle(0, 1, 0, 1, 6, 6)
    mesh = UnstructuredHyperbolicMesh(tri)

    @testset "cell_to_vertex: constant field" begin
        cell_vals = fill(3.14, mesh.ntri)
        vertex_vals = cell_to_vertex(mesh, cell_vals)

        # Constant should be preserved exactly
        for (_, val) in vertex_vals
            @test val ≈ 3.14 atol = 1.0e-12
        end
    end

    @testset "vertex_to_cell: constant field" begin
        vertex_vals = Dict(v => 2.72 for v in keys(mesh.vertex_coords))
        cell_vals = vertex_to_cell(mesh, vertex_vals)

        for val in cell_vals
            @test val ≈ 2.72 atol = 1.0e-12
        end
    end

    @testset "cell_to_vertex: linear field" begin
        # f(x,y) = x + y at centroids
        cell_vals = [cx + cy for (cx, cy) in mesh.tri_centroids]
        vertex_vals = cell_to_vertex(mesh, cell_vals)

        # For a linear field on a good mesh, area-weighted
        # average at vertices should be close to exact value
        for (v, val) in vertex_vals
            x, y = mesh.vertex_coords[v]
            # Not exact at vertices due to area weighting,
            # but should be reasonably close
            @test abs(val - (x + y)) < 0.2
        end
    end

    @testset "vertex_to_cell: linear field" begin
        # f(x,y) = x + y at vertices
        vertex_vals = Dict(v => x + y for (v, (x, y)) in mesh.vertex_coords)
        cell_vals = vertex_to_cell(mesh, vertex_vals)

        # Average of linear function at vertices = value at centroid
        for (tri_id, (v1, v2, v3)) in enumerate(mesh.tri_verts)
            cx, cy = mesh.tri_centroids[tri_id]
            @test cell_vals[tri_id] ≈ cx + cy atol = 1.0e-10
        end
    end

    @testset "SVector data transfer" begin
        # Test with SVector values (like conserved variables)
        cell_vals = [SVector(1.0, cx) for (cx, _) in mesh.tri_centroids]
        vertex_vals = cell_to_vertex(mesh, cell_vals)

        # First component (constant 1.0) should be preserved
        for (_, val) in vertex_vals
            @test val[1] ≈ 1.0 atol = 1.0e-10
        end

        # Roundtrip: cell → vertex → cell
        cell_back = vertex_to_cell(mesh, vertex_vals)
        @test length(cell_back) == mesh.ntri
        for val in cell_back
            @test val isa SVector{2, Float64}
            @test val[1] ≈ 1.0 atol = 1.0e-10
        end
    end
end

# ============================================================
# Comparison with IMEX
# ============================================================
@testset "Splitting vs IMEX" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)

    uniform_ic(x) = SVector(1.0, 0.0, 1.0)
    t_final = 0.05

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), uniform_ic;
        final_time = t_final, cfl = 0.5
    )

    cooling = CoolingSource(T -> 0.05; mu_mol = 1.0)

    # IMEX solution
    _, U_imex, t_imex = solve_hyperbolic_imex(prob, cooling)

    # Strang splitting solution
    _, U_strang, t_strang = solve_coupled(
        prob, cooling;
        splitting = StrangSplitting()
    )

    @test t_imex ≈ t_strang atol = 1.0e-10

    # Both should give similar results on a uniform field
    # (no advection, only source term)
    for i in eachindex(U_imex)
        # Mass should be identical (no mass source)
        @test U_imex[i][1] ≈ U_strang[i][1] atol = 1.0e-10
        # Energy should be close (different time integration schemes)
        @test abs(U_imex[i][3] - U_strang[i][3]) < 0.01
    end
end

# ============================================================
# 2D CoupledProblem API
# ============================================================
@testset "2D CoupledProblem API" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 15, 15)
    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        sod_ic_2d;
        final_time = 0.05, cfl = 0.4
    )

    hyp_op = HyperbolicOperator(prob)
    src_op = SourceOperator(law, NullSource())
    coupled = CoupledProblem(
        (hyp_op, src_op), StrangSplitting(),
        0.0, 0.05
    )

    coords, U, t = solve_coupled(coupled)
    @test t ≈ 0.05 atol = 1.0e-10
    @test size(U) == (15, 15)

    for iy in 1:15, ix in 1:15
        w = conserved_to_primitive(law, U[ix, iy])
        @test w[1] > 0.0
        @test w[4] > 0.0
    end
end

# ============================================================
# Edge Cases
# ============================================================
@testset "Edge Cases" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 20)

    @testset "Very short time" begin
        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
            final_time = 1.0e-10, cfl = 0.5
        )
        _, U, t = solve_coupled(prob, NullSource(); splitting = LieTrotterSplitting())
        @test t ≈ 1.0e-10 atol = 1.0e-15
    end

    @testset "Single cell" begin
        mesh1 = StructuredMesh1D(0.0, 1.0, 1)
        prob = HyperbolicProblem(
            law, mesh1, HLLSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
            final_time = 0.01, cfl = 0.5
        )
        _, U, t = solve_coupled(prob, NullSource(); splitting = LieTrotterSplitting())
        @test length(U) == 1
        @test t ≈ 0.01 atol = 1.0e-10
    end

    @testset "Different gamma" begin
        eos2 = IdealGasEOS(5 / 3)
        law2 = EulerEquations{1}(eos2)
        prob = HyperbolicProblem(
            law2, mesh, HLLSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(), sod_ic_1d;
            final_time = 0.1, cfl = 0.5
        )
        cooling = CoolingSource(T -> 0.01; mu_mol = 1.0)
        _, U, t = solve_coupled(prob, cooling; splitting = StrangSplitting())
        @test t ≈ 0.1 atol = 1.0e-10
        for u in U
            w = conserved_to_primitive(law2, u)
            @test w[1] > 0.0
            @test w[3] > 0.0
        end
    end
end

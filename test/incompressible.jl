using FiniteVolumeMethod
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# ── Mesh builder (shared helper) ─────────────────────────────────────
include("TestHelpers.jl")

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

        # Normalization: the normalized residual is raw / Σ|phi_f|
        r_raw = FiniteVolumeMethod.continuity_residual(state, mesh; normalize = false)
        @test r2 ≈ r_raw / (0.01 * nf)
    end

    # ── 10. Transient ddt uses old-time snapshot (kinematic) ──────────
    @testset "Transient ddt: old-time snapshot + unit density" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
        )
        prob = IncompressibleProblem(mesh, bcs, PISO(); nu = 0.1)
        state = IncompressibleState(mesh)
        nc = length(mesh.cell_volumes)
        dt = 0.05

        # Set U = u1, snapshot, then move U to u2 (simulating an outer
        # iterate).  The ddt RHS must reference u1, NOT u2.
        u1 = [SVector(0.3 + 0.01 * c, -0.2) for c in 1:nc]
        copyto!(state.U.internal, u1)
        FiniteVolumeMethod._snapshot_old_time!(state)
        u2 = [SVector(9.9, 9.9) for _ in 1:nc]
        copyto!(state.U.internal, u2)

        eq_t = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq_t, state, prob, 1; dt = dt)
        eq_s = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq_s, state, prob, 1)

        for c in 1:nc
            V_c = mesh.cell_volumes[c]
            ddt_rhs = eq_t.b[c] - eq_s.b[c]
            @test ddt_rhs ≈ V_c / dt * u1[c][1] atol = 1.0e-12
        end

        # Kinematic form: density must NOT scale the ddt term
        prob_heavy = IncompressibleProblem(mesh, bcs, PISO(); nu = 0.1, density = 1000.0)
        eq_h = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq_h, state, prob_heavy, 1; dt = dt)
        @test eq_h.A.nzval ≈ eq_t.A.nzval atol = 1.0e-12
        @test eq_h.b ≈ eq_t.b atol = 1.0e-12
    end

    # ── 11. PIMPLE outer-iteration consistency (Stokes-like decay) ────
    @testset "PIMPLE n_outer consistency for transient decay" begin
        # A decaying vortex in a closed box: outer iterations must
        # converge to the SAME backward-Euler step, not re-discretize
        # the ddt against the previous iterate (which drives the
        # solution toward steady state within one dt).
        mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
        )
        nc = length(mesh.cell_volumes)
        U0 = Vector{SVector{2, Float64}}(undef, nc)
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            # Small divergence-free vortex (Taylor-Green-like)
            U0[c] = 1.0e-3 * SVector(
                sin(π * x) * cos(π * y),
                -cos(π * x) * sin(π * y),
            )
        end

        function run_pimple(n_outer)
            algo = PIMPLE(; n_outer = n_outer, n_correctors = 2, alpha_U = 0.7, alpha_p = 0.3)
            prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.05)
            result = solve_incompressible(prob, (0.0, 0.05), 0.01; U0 = copy(U0))
            return result.state.U.internal
        end

        U_1 = run_pimple(1)
        U_3 = run_pimple(3)

        n1 = maximum(norm.(U_1))
        n3 = maximum(norm.(U_3))
        @test n1 > 0 && n3 > 0
        # Both must show comparable decay: relative difference small.
        # (Before the U_old fix, n_outer = 3 decayed ~3x faster.)
        @test isapprox(n3, n1; rtol = 0.05)

        # And the flow must actually DECAY relative to the IC
        @test n3 < 1.0e-3
    end

    # ── 12. Symmetry/slip boundaries do not leak mass ──────────────────
    @testset "SymmetryBC / SlipWallBC zero boundary flux" begin
        mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => SymmetryBC(),
            :right => SlipWallBC(),
            :bottom => NoSlipWallBC(),
            :top => FixedVelocityBC((0.5, 0.0)),
        )
        algo = SIMPLE(; max_iterations = 30, tolerance = 1.0e-10)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.05)
        result = solve_simple(prob)
        state = result.state

        # Wall-normal boundary velocity must be exactly projected out on
        # symmetry/slip patches → boundary face flux ~ 0 (machine eps)
        nf = size(mesh.face_cells, 2)
        total_boundary_flux = 0.0
        for (i, f) in enumerate(state.U.boundary_face_indices)
            tag = FiniteVolumeMethod._face_tag(mesh, f)
            S_f = FiniteVolumeMethod.face_normal_area(mesh, f)
            flux = dot(state.U.boundary[i], S_f)
            if tag === :left || tag === :right
                @test abs(flux) < 1.0e-14
            end
            total_boundary_flux += flux
        end
        # Closed-box global mass balance across ALL boundaries
        @test abs(total_boundary_flux) < 1.0e-12
    end

    # ── 13. Density invariance of the kinematic solver ─────────────────
    @testset "Velocity independent of density at fixed nu" begin
        mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(), :top => FixedVelocityBC((1.0, 0.0)),
        )
        algo = SIMPLE(; max_iterations = 40, tolerance = 1.0e-12)

        prob_1 = IncompressibleProblem(mesh, bcs, algo; nu = 0.02, density = 1.0)
        prob_1000 = IncompressibleProblem(mesh, bcs, algo; nu = 0.02, density = 1000.0)
        r1 = solve_simple(prob_1)
        r1000 = solve_simple(prob_1000)

        for c in eachindex(r1.state.U.internal)
            @test r1.state.U.internal[c] ≈ r1000.state.U.internal[c] rtol = 1.0e-12
        end

        # Transient path too (ddt used to be scaled by rho)
        piso_1 = IncompressibleProblem(mesh, bcs, PISO(); nu = 0.02, density = 1.0)
        piso_1000 = IncompressibleProblem(mesh, bcs, PISO(); nu = 0.02, density = 1000.0)
        t1 = solve_incompressible(piso_1, (0.0, 0.02), 0.01)
        t1000 = solve_incompressible(piso_1000, (0.0, 0.02), 0.01)
        for c in eachindex(t1.state.U.internal)
            @test t1.state.U.internal[c] ≈ t1000.state.U.internal[c] rtol = 1.0e-12
        end
    end

    # ── 14. Symmetric pressure reference elimination ───────────────────
    @testset "fix_pressure_reference! keeps matrix symmetric" begin
        mesh = build_cartesian_unstructured_mesh(5, 5, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
        )
        prob = IncompressibleProblem(mesh, bcs, SIMPLE(); nu = 0.1)
        state = IncompressibleState(mesh)
        nc = length(mesh.cell_volumes)
        for c in 1:nc
            state.A_P[c] = 1.0 + 0.5 * sin(1.7 * c)  # spatially varying
        end

        p_eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_pressure!(p_eq, state, prob)
        # Pure-Neumann Laplacian is symmetric before the fix
        @test norm(p_eq.A - p_eq.A') < 1.0e-12

        FiniteVolumeMethod.fix_pressure_reference!(p_eq, 1, 0.0)
        # Symmetric elimination must PRESERVE symmetry (row-only zeroing
        # used to break it, ruling out CG/AMG)
        @test norm(p_eq.A - p_eq.A') < 1.0e-12
        @test p_eq.A[1, 1] == 1.0
        @test p_eq.b[1] == 0.0

        # Nonzero reference value moves column entries to the RHS
        p_eq2 = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_pressure!(p_eq2, state, prob)
        A_before = copy(p_eq2.A)
        b_before = copy(p_eq2.b)
        FiniteVolumeMethod.fix_pressure_reference!(p_eq2, 1, 2.5)
        for i in 2:nc
            @test p_eq2.b[i] ≈ b_before[i] - A_before[i, 1] * 2.5 atol = 1.0e-12
        end
    end

    # ── 15. Time-dependent velocity BCs enter matrix + boundary ───────
    @testset "TimeDependentVelocityBC / SpatialVelocityBC honored" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        lid = TimeDependentVelocityBC{2, Float64}(t -> SVector(sin(2π * t), 0.0))
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(), :top => lid,
        )
        prob = IncompressibleProblem(mesh, bcs, PISO(); nu = 0.1)
        state = IncompressibleState(mesh)

        # Matrix RHS must change with time (was frozen at t = 0 before)
        eq_t0 = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq_t0, state, prob, 1; t = 0.0)
        eq_t1 = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq_t1, state, prob, 1; t = 0.25)
        @test !(eq_t0.b ≈ eq_t1.b)

        # Boundary velocity update must evaluate the BC at time t
        FiniteVolumeMethod.update_boundary_velocity!(state, bcs, mesh; t = 0.25)
        top_vals = [
            state.U.boundary[i][1]
                for (i, f) in enumerate(state.U.boundary_face_indices)
                if FiniteVolumeMethod._face_tag(mesh, f) === :top
        ]
        @test all(v -> isapprox(v, 1.0; atol = 1.0e-12), top_vals)

        # The transient solution must respond to the oscillating lid
        result = solve_incompressible(prob, (0.0, 0.25), 0.05)
        u_top = maximum(u[1] for u in result.state.U.internal)
        @test u_top > 1.0e-3  # lid at sin(π/2) = 1 has dragged the fluid

        # SpatialVelocityBC: true face values enter the RHS (not a
        # Dirichlet(0) placeholder)
        sbc = FiniteVolumeMethod.SpatialVelocityBC(
            x -> SVector(x[1] * (1.0 - x[1]), 0.0), Val(2), Float64,
        )
        bcs_s = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(), :top => sbc,
        )
        prob_s = IncompressibleProblem(mesh, bcs_s, SIMPLE(); nu = 0.1)
        eq_s = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq_s, IncompressibleState(mesh), prob_s, 1)
        bcs_zero = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
        )
        prob_z = IncompressibleProblem(mesh, bcs_zero, SIMPLE(); nu = 0.1)
        eq_z = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq_z, IncompressibleState(mesh), prob_z, 1)
        @test !(eq_s.b ≈ eq_z.b)
    end

    # ── 16. Convection honors nonzero Neumann boundary values ──────────
    @testset "Convection BC: nonzero Neumann not treated as zero" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        phi_flux = FiniteVolumeMethod.FaceFluxField(:phi, mesh; value = 0.0)
        nf = size(mesh.face_cells, 2)
        for f in 1:nf
            # Uniform positive x-flux
            S_f = FiniteVolumeMethod.face_normal_area(mesh, f)
            phi_flux.values[f] = S_f[1]
        end

        function assemble_neumann(gval)
            eq = FiniteVolumeMethod.CollocatedEquation(mesh)
            bcs = Dict{Symbol, AbstractBoundaryCondition}(
                :left => ParabolicDirichlet(1.0),
                :right => ParabolicNeumann(gval),
                :bottom => ParabolicNeumann(0.0),
                :top => ParabolicNeumann(0.0),
            )
            FiniteVolumeMethod.assemble_convection!(eq, phi_flux, mesh, bcs)
            return eq
        end

        eq0 = assemble_neumann(0.0)
        eq2 = assemble_neumann(2.0)
        @test eq0.A.nzval ≈ eq2.A.nzval atol = 1.0e-14   # implicit part equal
        @test !(eq0.b ≈ eq2.b)                            # explicit part differs
        # The difference is -F_f * g * d_n on right-boundary owner cells
        db = eq2.b .- eq0.b
        @test any(x -> abs(x) > 1.0e-12, db)
        @test all(x -> x <= 1.0e-14, db)  # outflow with g > 0 lowers b
    end

    # ── 17. PISO type flexibility ──────────────────────────────────────
    @testset "PISO{T} constructor" begin
        p32 = PISO{Float32}(; n_correctors = 3)
        @test p32 isa PISO{Float32}
        @test p32.n_correctors == 3
        @test PISO() isa PISO{Float64}
    end

    # ── 18. Snapshots, converged flag, and U0/p0 initial conditions ───
    @testset "SolveResult snapshots + initial conditions" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
        )
        prob = IncompressibleProblem(mesh, bcs, PISO(); nu = 0.05)
        nc = length(mesh.cell_volumes)

        U0 = [SVector(1.0e-3, 0.0) for _ in 1:nc]
        result = solve_incompressible(
            prob, (0.0, 0.04), 0.01;
            save_every = 2, U0 = U0, p0 = fill(0.5, nc),
        )
        # 4 steps, save_every = 2 → 2 snapshots (was collected + discarded)
        @test result.iterations == 4
        @test length(result.snapshots) == 2
        @test result.converged  # finite residuals
        # Snapshots are independent copies
        @test result.snapshots[1].U.internal !== result.state.U.internal
        # The IC decayed but is still visible in the first snapshot
        @test maximum(u[1] for u in result.snapshots[1].U.internal) > 0.0

        # Steady results carry an empty snapshot vector
        sprob = IncompressibleProblem(
            mesh, bcs, SIMPLE(; max_iterations = 3); nu = 0.05,
        )
        sres = solve_simple(sprob)
        @test isempty(sres.snapshots)
    end
end

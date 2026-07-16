using FiniteVolumeMethod
using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK: Euler
using StaticArrays
using Test

# ============================================================
# Constants for 2D Euler AMR tests
# ============================================================

const SD_EOS = IdealGasEOS(1.4)
const SD_LAW = EulerEquations{2}(SD_EOS)
const SD_NVAR = 4  # 2D Euler: [rho, rho*vx, rho*vy, E]

# ============================================================
# 1. AMR ODEProblem construction
# ============================================================

@testset "AMR ODEProblem construction" begin
    # Set up a single-block AMR grid
    criterion = GradientRefinement(; refine_threshold = 0.5, coarsen_threshold = 0.05)
    block_size = (8, 8)
    max_levels = 2
    domain_lo = (0.0, 0.0)
    domain_hi = (1.0, 1.0)

    grid = AMRGrid(SD_LAW, criterion, block_size, max_levels, domain_lo, domain_hi, Val(SD_NVAR))

    # Fill the single root block with a uniform state
    gamma = 1.4
    rho = 1.0
    P = 1.0
    E = P / (gamma - 1)
    state = SVector(rho, 0.0, 0.0, E)
    root = grid.blocks[1]
    for j in 1:8, i in 1:8
        root.U[i, j] = state
    end

    # Build AMRProblem
    solver = LaxFriedrichsSolver()
    recon = NoReconstruction()
    bcs = (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC())
    amr_prob = AMRProblem(grid, solver, recon, bcs; final_time = 0.1, cfl = 0.4)

    # Build ODEProblem
    ode_prob = ODEProblem(amr_prob)

    # Verify it is an ODEProblem
    @test ode_prob isa ODEProblem

    # u0 length should equal total_cells * NVAR
    # Single block with 8x8 = 64 cells, 4 variables each => 256 entries
    expected_len = 8 * 8 * SD_NVAR
    @test length(ode_prob.u0) == expected_len

    # Parameter should be an AMRCache
    @test ode_prob.p isa AMRCache

    # Tspan should match
    @test ode_prob.tspan == (0.0, 0.1)
end

# ============================================================
# 2. AMR cache round-trip
# ============================================================

@testset "AMR cache round-trip" begin
    criterion = GradientRefinement(; refine_threshold = 0.5, coarsen_threshold = 0.05)
    block_size = (4, 4)
    max_levels = 2
    domain_lo = (0.0, 0.0)
    domain_hi = (1.0, 1.0)

    grid = AMRGrid(SD_LAW, criterion, block_size, max_levels, domain_lo, domain_hi, Val(SD_NVAR))

    # Fill root block with identifiable state: density = i + j
    gamma = 1.4
    root = grid.blocks[1]
    for j in 1:4, i in 1:4
        rho = Float64(i + j)
        E = 1.0 / (gamma - 1)
        root.U[i, j] = SVector(rho, 0.0, 0.0, E)
    end

    solver = LaxFriedrichsSolver()
    recon = NoReconstruction()
    bcs = (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC())
    amr_prob = AMRProblem(grid, solver, recon, bcs; final_time = 0.1, cfl = 0.4)

    # Build cache and flatten state
    cache = build_amr_cache(amr_prob)
    u0 = flatten_amr_state(cache)

    # Verify flattened length: 4*4 cells * 4 variables = 64
    @test length(u0) == 4 * 4 * SD_NVAR

    # Unfold back into padded arrays
    unfold_amr!(cache, u0)

    # Verify values survive the round-trip by checking padded array interior
    # The root block (bid=1) has interior starting at offset (3,3) in padded array
    pad = cache.per_block_padded[1]
    for j in 1:4, i in 1:4
        expected_rho = Float64(i + j)
        @test pad[i + 2, j + 2][1] == expected_rho
    end

    # Fold zeros into du, then verify shape
    du = zeros(length(u0))
    # Put known dU values in the padded dU
    du_pad = cache.per_block_dU[1]
    for j in 1:4, i in 1:4
        du_pad[i + 2, j + 2] = SVector(1.0, 2.0, 3.0, 4.0)
    end
    fold_amr!(du, cache)

    # Verify the folded du values
    du_sv = reinterpret(SVector{SD_NVAR, Float64}, du)
    for k in 1:(4 * 4)
        @test du_sv[k] == SVector(1.0, 2.0, 3.0, 4.0)
    end
end

# ============================================================
# 3. AMR state consistency with Sod-like conditions
# ============================================================

@testset "AMR state consistency" begin
    criterion = GradientRefinement(; refine_threshold = 0.5, coarsen_threshold = 0.05)
    block_size = (8, 8)
    max_levels = 2
    domain_lo = (0.0, 0.0)
    domain_hi = (1.0, 1.0)

    grid = AMRGrid(SD_LAW, criterion, block_size, max_levels, domain_lo, domain_hi, Val(SD_NVAR))

    # Fill with Sod-like conditions: left half high density, right half low density
    gamma = 1.4
    root = grid.blocks[1]
    rho_L = 1.0
    P_L = 1.0
    rho_R = 0.125
    P_R = 0.1
    for j in 1:8, i in 1:8
        if i <= 4
            rho = rho_L
            E = P_L / (gamma - 1)
        else
            rho = rho_R
            E = P_R / (gamma - 1)
        end
        root.U[i, j] = SVector(rho, 0.0, 0.0, E)
    end

    solver = LaxFriedrichsSolver()
    recon = NoReconstruction()
    bcs = (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC())
    amr_prob = AMRProblem(grid, solver, recon, bcs; final_time = 0.2, cfl = 0.4)

    # Build ODEProblem
    ode_prob = ODEProblem(amr_prob)
    cache = ode_prob.p

    # Flatten and verify density values from the state vector
    u0 = ode_prob.u0
    u0_sv = reinterpret(SVector{SD_NVAR, Float64}, u0)

    # The flat ordering is column-major: for iy in 1:ny, for ix in 1:nx
    # flat_idx = (iy - 1) * nx + ix
    nx, ny = 8, 8
    for iy in 1:ny
        for ix in 1:nx
            flat_idx = (iy - 1) * nx + ix
            density = u0_sv[flat_idx][1]
            if ix <= 4
                @test density == rho_L
            else
                @test density == rho_R
            end
        end
    end

    # Verify that the total number of cells tracked by the cache is correct
    @test cache.total_cells == nx * ny

    # Verify block_ids contains only the root block
    @test length(cache.block_ids) == 1
    @test cache.block_ids[1] == 1

    # Verify block_offsets starts at zero for the first block
    @test cache.block_offsets[1] == 0

    # Verify primitive_to_conserved consistency:
    # For the left state, rho=1, vx=vy=0, P=1 => E = P/(gamma-1) = 2.5
    w_L = SVector(rho_L, 0.0, 0.0, P_L)  # (rho, vx, vy, P)
    u_L = primitive_to_conserved(SD_LAW, w_L)
    @test u_L[1] == rho_L
    @test u_L[2] == 0.0  # rho*vx
    @test u_L[3] == 0.0  # rho*vy
    @test u_L[4] == P_L / (gamma - 1)  # E = P/(gamma-1) for zero velocity
end

# ============================================================
# AMR ODEProblem honesty guards
# ============================================================
@testset "ODEProblem(::AMRProblem) guards" begin
    criterion = GradientRefinement(; refine_threshold = 0.5, coarsen_threshold = 0.05)
    grid = AMRGrid(SD_LAW, criterion, (8, 8), 2, (0.0, 0.0), (1.0, 1.0), Val(SD_NVAR))
    root = grid.blocks[1]
    for j in 1:8, i in 1:8
        root.U[i, j] = primitive_to_conserved(SD_LAW, SVector(1.0, 0.0, 0.0, 1.0))
    end
    bcs = (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC())

    @testset "non-default reconstruction warns (RHS is first-order only)" begin
        prob_muscl = AMRProblem(
            grid, HLLCSolver(), CellCenteredMUSCL(), bcs;
            final_time = 0.01, cfl = 0.4
        )
        @test_logs (:warn,) match_mode = :any ODEProblem(prob_muscl)
    end

    @testset "multi-block grid is now supported (ghost exchange)" begin
        refine_block!(grid, 1)
        prob = AMRProblem(
            grid, HLLCSolver(), NoReconstruction(), bcs;
            final_time = 0.01, cfl = 0.4
        )
        ode_prob = ODEProblem(prob)
        @test ode_prob isa ODEProblem
        # 4 blocks of 8x8 cells, 4 variables each
        @test length(ode_prob.u0) == 4 * 8 * 8 * SD_NVAR
    end

    @testset "3D grid throws (AMR cache/RHS are 2D-only)" begin
        law3 = EulerEquations{3}(SD_EOS)
        grid3 = AMRGrid(law3, criterion, (4, 4, 4), 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        u3 = primitive_to_conserved(law3, SVector(1.0, 0.0, 0.0, 0.0, 1.0))
        b3 = grid3.blocks[1]
        for k in 1:4, j in 1:4, i in 1:4
            b3.U[i, j, k] = u3
        end
        prob3 = AMRProblem(
            grid3, HLLCSolver(), NoReconstruction(), ntuple(_ -> TransmissiveBC(), 6);
            final_time = 0.01, cfl = 0.4
        )
        @test_throws ArgumentError ODEProblem(prob3)
    end
end

# ============================================================
# Inter-block ghost exchange through the SciML RHS
# ============================================================

const SD_GX_CRIT = GradientRefinement(; refine_threshold = 1.0e9, coarsen_threshold = 0.0)
const SD_GX_BCS = (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC())

function _sd_fill_ic!(grid, ic)
    for b in values(grid.blocks)
        b.active || continue
        for j in 1:b.dims[2], i in 1:b.dims[1]
            x, y = block_cell_center(b, i, j)
            b.U[i, j] = primitive_to_conserved(SD_LAW, ic(x, y))
        end
    end
    return grid
end

@testset "same-level multi-block matches single-block reference (SciML path)" begin
    # Pulse advecting in +x across the vertical block seam at x = 0.5.
    ic = (x, y) -> SVector(1.0 + 0.5 * exp(-200.0 * ((x - 0.35)^2 + (y - 0.5)^2)), 1.0, 0.0, 1.0)

    grid4 = AMRGrid(SD_LAW, SD_GX_CRIT, (8, 8), 3, (0.0, 0.0), (1.0, 1.0), Val(SD_NVAR))
    refine_block!(grid4, 1)
    _sd_fill_ic!(grid4, ic)
    grid1 = AMRGrid(SD_LAW, SD_GX_CRIT, (16, 16), 3, (0.0, 0.0), (1.0, 1.0), Val(SD_NVAR))
    _sd_fill_ic!(grid1, ic)

    prob4 = AMRProblem(
        grid4, HLLCSolver(), NoReconstruction(), SD_GX_BCS;
        final_time = 0.05, cfl = 0.4, regrid_interval = 0
    )
    prob1 = AMRProblem(
        grid1, HLLCSolver(), NoReconstruction(), SD_GX_BCS;
        final_time = 0.05, cfl = 0.4, regrid_interval = 0
    )
    op4 = ODEProblem(prob4)
    op1 = ODEProblem(prob1)

    dt = 0.4 * (1 / 16) / 4.0
    sol4 = solve(op4, Euler(); dt = dt, adaptive = false)
    sol1 = solve(op1, Euler(); dt = dt, adaptive = false)
    @test sol4.retcode == SciMLBase.ReturnCode.Success
    @test sol1.retcode == SciMLBase.ReturnCode.Success

    cache4 = op4.p
    u4 = reinterpret(SVector{SD_NVAR, Float64}, sol4.u[end])
    u1 = reinterpret(SVector{SD_NVAR, Float64}, sol1.u[end])
    ref = grid1.blocks[1]
    maxdiff = 0.0
    for (idx, bid) in enumerate(cache4.block_ids)
        off = cache4.block_offsets[idx]
        b = cache4.grid.blocks[bid]
        nx, ny = b.dims
        for j in 1:ny, i in 1:nx
            x, y = block_cell_center(b, i, j)
            ri = Int(floor((x - ref.origin[1]) / ref.dx[1])) + 1
            rj = Int(floor((y - ref.origin[2]) / ref.dx[2])) + 1
            v4 = u4[off + (j - 1) * nx + i]
            v1 = u1[(rj - 1) * 16 + ri]
            maxdiff = max(maxdiff, maximum(abs.(v4 - v1)))
        end
    end
    @test maxdiff < 1.0e-13
end

@testset "multi-level RHS: uniform state and seam conservation (SciML path)" begin
    # Refined patch on [0, 0.5]^2: 3 level-1 leaves + 4 level-2 leaves.
    function build_patch_grid(ic)
        grid = AMRGrid(SD_LAW, SD_GX_CRIT, (8, 8), 3, (0.0, 0.0), (1.0, 1.0), Val(SD_NVAR))
        cids = refine_block!(grid, 1)
        llid = first(cid for cid in cids if grid.blocks[cid].origin == (0.0, 0.0))
        refine_block!(grid, llid)
        return _sd_fill_ic!(grid, ic)
    end

    @testset "uniform state gives exactly zero RHS" begin
        grid = build_patch_grid((x, y) -> SVector(1.3, 0.4, -0.2, 2.0))
        prob = AMRProblem(
            grid, HLLCSolver(), NoReconstruction(), SD_GX_BCS;
            final_time = 0.01, cfl = 0.4, regrid_interval = 0
        )
        op = ODEProblem(prob)
        du = zero(op.u0)
        op.f(du, op.u0, op.p, 0.0)
        @test maximum(abs.(du)) == 0.0
    end

    @testset "seam-straddling pulse: RHS conserves all invariants" begin
        # Compact pulse straddling the level-2 -> level-1 seam at x = 0.5;
        # boundary-adjacent cells are exactly background, so the total
        # volume-weighted RHS must vanish to roundoff for every conserved
        # variable. This is the conservation gate for the coarse-fine
        # seam flux correction.
        ic = (x, y) -> SVector(
            (0.4 < x < 0.6 && 0.3 < y < 0.45) ? 1.5 : 1.0,
            1.0, 0.2, 1.0
        )
        grid = build_patch_grid(ic)
        prob = AMRProblem(
            grid, HLLCSolver(), NoReconstruction(), SD_GX_BCS;
            final_time = 0.01, cfl = 0.4, regrid_interval = 0
        )
        op = ODEProblem(prob)
        cache = op.p
        du = zero(op.u0)
        op.f(du, op.u0, cache, 0.0)

        du_sv = reinterpret(SVector{SD_NVAR, Float64}, du)
        rate = zero(SVector{SD_NVAR, Float64})
        for (idx, bid) in enumerate(cache.block_ids)
            off = cache.block_offsets[idx]
            b = cache.grid.blocks[bid]
            nx, ny = b.dims
            vol = b.dx[1] * b.dx[2]
            for j in 1:ny, i in 1:nx
                rate = rate + du_sv[off + (j - 1) * nx + i] * vol
            end
        end
        for k in 1:SD_NVAR
            @test abs(rate[k]) < 1.0e-13
        end
    end
end

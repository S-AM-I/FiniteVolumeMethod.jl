using FiniteVolumeMethod
using OrdinaryDiffEq
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

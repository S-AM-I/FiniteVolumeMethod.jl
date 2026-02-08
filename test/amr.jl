using FiniteVolumeMethod
using Test
using StaticArrays
using LinearAlgebra

# ============================================================
# Helper: create a standard 2D Euler law for AMR tests
# ============================================================

const AMR_EOS = IdealGasEOS(1.4)
const AMR_LAW = EulerEquations{2}(AMR_EOS)
const AMR_NVAR = 4  # 2D Euler: [rho, rho*vx, rho*vy, E]

# ============================================================
# 1. AMRBlock tests
# ============================================================

@testset "AMRBlock constructor and basic properties" begin
    block_size = (4, 4)
    origin = (0.0, 0.0)
    dx = (0.25, 0.25)
    level = 0
    id = 1

    block = AMRBlock(id, level, origin, block_size, dx, Val(AMR_NVAR))

    # Basic field checks
    @test block.id == 1
    @test block.level == 0
    @test block.origin == (0.0, 0.0)
    @test block.dims == (4, 4)
    @test block.dx == (0.25, 0.25)
    @test block.parent_id == -1
    @test isempty(block.child_ids)
    @test block.active == true
    @test isempty(block.neighbors)

    # U array has correct size
    @test size(block.U) == (4, 4)

    # U array is zero-initialized
    zero_state = zero(SVector{AMR_NVAR, Float64})
    for j in 1:4, i in 1:4
        @test block.U[i, j] == zero_state
    end

    # is_leaf is true for a new block (no children)
    @test is_leaf(block) == true
end

@testset "AMRBlock with non-square block size" begin
    block = AMRBlock(1, 0, (0.0, 0.0), (8, 4), (0.125, 0.25), Val(AMR_NVAR))
    @test size(block.U) == (8, 4)
    @test block.dims == (8, 4)
    @test block.dx == (0.125, 0.25)
end

@testset "block_cell_center" begin
    origin = (1.0, 2.0)
    dx = (0.5, 0.5)
    block = AMRBlock(1, 0, origin, (4, 4), dx, Val(AMR_NVAR))

    # Cell (1,1): center at origin + (1-0.5)*dx = origin + 0.5*dx
    xc, yc = block_cell_center(block, 1, 1)
    @test xc ≈ 1.0 + 0.5 * 0.5
    @test yc ≈ 2.0 + 0.5 * 0.5

    # Cell (4,4): center at origin + (4-0.5)*dx
    xc, yc = block_cell_center(block, 4, 4)
    @test xc ≈ 1.0 + 3.5 * 0.5
    @test yc ≈ 2.0 + 3.5 * 0.5

    # Cell (2,3): center at origin + (2-0.5)*dx[1], origin + (3-0.5)*dx[2]
    xc, yc = block_cell_center(block, 2, 3)
    @test xc ≈ 1.0 + 1.5 * 0.5
    @test yc ≈ 2.0 + 2.5 * 0.5
end

# ============================================================
# 2. AMRGrid tests
# ============================================================

@testset "AMRGrid constructor" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    block_size = (4, 4)
    max_level = 3
    domain_lo = (0.0, 0.0)
    domain_hi = (1.0, 1.0)

    grid = AMRGrid(AMR_LAW, criterion, block_size, max_level, domain_lo, domain_hi, Val(AMR_NVAR))

    # Should have exactly one root block
    @test length(grid.blocks) == 1
    @test haskey(grid.blocks, 1)

    root = grid.blocks[1]
    @test root.id == 1
    @test root.level == 0
    @test root.origin == (0.0, 0.0)
    @test root.dims == (4, 4)
    @test root.active == true
    @test root.parent_id == -1
    @test isempty(root.child_ids)

    # dx should be (domain_hi - domain_lo) / block_size
    @test root.dx[1] ≈ 0.25
    @test root.dx[2] ≈ 0.25

    # Grid properties
    @test grid.max_level == 3
    @test grid.refinement_ratio == 2
    @test grid.block_size == (4, 4)
end

@testset "active_blocks on fresh grid" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    ab = active_blocks(grid)
    @test length(ab) == 1
    @test ab[1].id == 1
end

@testset "blocks_at_level on fresh grid" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    @test length(blocks_at_level(grid, 0)) == 1
    @test length(blocks_at_level(grid, 1)) == 0
end

@testset "max_active_level on fresh grid" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    @test max_active_level(grid) == 0
end

# ============================================================
# 3. Refinement criteria tests
# ============================================================

@testset "GradientRefinement: keyword constructor" begin
    crit = GradientRefinement(; variable_index = 2, refine_threshold = 0.5, coarsen_threshold = 0.05)
    @test crit.variable_index == 2
    @test crit.refine_threshold == 0.5
    @test crit.coarsen_threshold == 0.05
end

@testset "GradientRefinement: default keyword constructor" begin
    crit = GradientRefinement()
    @test crit.variable_index == 1
    @test crit.refine_threshold == 0.1
    @test crit.coarsen_threshold == 0.01
end

@testset "CurrentSheetRefinement: constructor" begin
    crit = CurrentSheetRefinement(; refine_threshold = 0.2, coarsen_threshold = 0.02)
    @test crit.refine_threshold == 0.2
    @test crit.coarsen_threshold == 0.02

    crit2 = CurrentSheetRefinement()
    @test crit2.refine_threshold == 0.1
    @test crit2.coarsen_threshold == 0.01
end

@testset "GradientRefinement: needs_refinement with large gradient" begin
    # Create a block with a large density jump (shock-like)
    block = AMRBlock(1, 0, (0.0, 0.0), (8, 8), (0.125, 0.125), Val(AMR_NVAR))

    # Fill left half with high density, right half with low density
    gamma = 1.4
    for j in 1:8, i in 1:8
        if i <= 4
            rho, P = 1.0, 1.0
        else
            rho, P = 0.125, 0.1
        end
        E = P / (gamma - 1)
        block.U[i, j] = SVector(rho, 0.0, 0.0, E)
    end

    # Use a low threshold so the gradient triggers refinement
    crit = GradientRefinement(; variable_index = 1, refine_threshold = 0.01, coarsen_threshold = 0.001)
    @test needs_refinement(crit, block, AMR_LAW) == true

    # Use a very high threshold so it does NOT trigger
    crit_high = GradientRefinement(; variable_index = 1, refine_threshold = 100.0, coarsen_threshold = 50.0)
    @test needs_refinement(crit_high, block, AMR_LAW) == false
end

@testset "GradientRefinement: needs_coarsening with uniform data" begin
    # Uniform data should trigger coarsening
    block = AMRBlock(1, 0, (0.0, 0.0), (8, 8), (0.125, 0.125), Val(AMR_NVAR))
    gamma = 1.4
    rho, P = 1.0, 1.0
    E = P / (gamma - 1)
    state = SVector(rho, 0.0, 0.0, E)
    for j in 1:8, i in 1:8
        block.U[i, j] = state
    end

    crit = GradientRefinement(; variable_index = 1, refine_threshold = 0.1, coarsen_threshold = 0.01)
    @test needs_coarsening(crit, block, AMR_LAW) == true
    @test needs_refinement(crit, block, AMR_LAW) == false
end

# ============================================================
# 4. Block refinement tests
# ============================================================

@testset "refine_block! creates 4 children (2D)" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    child_ids = refine_block!(grid, 1)

    # Should create 4 children in 2D
    @test length(child_ids) == 4

    # Parent becomes inactive
    parent = grid.blocks[1]
    @test parent.active == false
    @test is_leaf(parent) == false
    @test length(parent.child_ids) == 4

    # All children are active leaves
    for cid in child_ids
        child = grid.blocks[cid]
        @test child.active == true
        @test is_leaf(child) == true
        @test child.parent_id == 1
        @test child.level == 1
    end

    # Children dx should be half of parent dx
    for cid in child_ids
        child = grid.blocks[cid]
        @test child.dx[1] ≈ 0.125  # 0.25 / 2
        @test child.dx[2] ≈ 0.125
    end

    # Children should have the same block dimensions
    for cid in child_ids
        child = grid.blocks[cid]
        @test child.dims == (4, 4)
    end

    # Total blocks = 1 parent + 4 children = 5
    @test length(grid.blocks) == 5

    # active_blocks returns only the 4 children
    ab = active_blocks(grid)
    @test length(ab) == 4

    # max_active_level should be 1
    @test max_active_level(grid) == 1

    # blocks_at_level should show 4 at level 1, 0 at level 0
    @test length(blocks_at_level(grid, 0)) == 0
    @test length(blocks_at_level(grid, 1)) == 4
end

@testset "refine_block! child origins are correct" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    child_ids = refine_block!(grid, 1)

    # Collect all child origins
    origins = Set{NTuple{2, Float64}}()
    for cid in child_ids
        push!(origins, grid.blocks[cid].origin)
    end

    # With block_size=(4,4) and child_dx=(0.125,0.125):
    # child extent = 4 * 0.125 = 0.5 in each direction
    # Expected origins: (0,0), (0.5,0), (0,0.5), (0.5,0.5)
    @test (0.0, 0.0) in origins
    @test (0.5, 0.0) in origins
    @test (0.0, 0.5) in origins
    @test (0.5, 0.5) in origins
end

@testset "refine_block! respects max_level" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 1, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Refine root to level 1
    child_ids = refine_block!(grid, 1)
    @test length(child_ids) == 4

    # Try to refine a child (already at max_level=1)
    child_ids_2 = refine_block!(grid, child_ids[1])
    @test isempty(child_ids_2)
end

@testset "refine_block! inactive block returns empty" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Refine root (makes it inactive)
    refine_block!(grid, 1)

    # Try to refine the now-inactive root again
    result = refine_block!(grid, 1)
    @test isempty(result)
end

# ============================================================
# 5. Block coarsening tests
# ============================================================

@testset "coarsen_block! removes children and reactivates parent" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Refine root
    child_ids = refine_block!(grid, 1)
    @test length(child_ids) == 4
    @test grid.blocks[1].active == false

    # Coarsen back
    result = coarsen_block!(grid, 1)
    @test result == true

    # Parent is reactivated
    @test grid.blocks[1].active == true
    @test isempty(grid.blocks[1].child_ids)
    @test is_leaf(grid.blocks[1]) == true

    # Children are removed
    for cid in child_ids
        @test !haskey(grid.blocks, cid)
    end

    # Only root block remains
    @test length(grid.blocks) == 1
    @test length(active_blocks(grid)) == 1
    @test max_active_level(grid) == 0
end

@testset "coarsen_block! fails if children have grandchildren" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Refine root
    child_ids = refine_block!(grid, 1)

    # Refine one of the children
    grandchild_ids = refine_block!(grid, child_ids[1])
    @test length(grandchild_ids) == 4

    # Now try to coarsen root: should fail because child has grandchildren
    result = coarsen_block!(grid, 1)
    @test result == false
end

@testset "coarsen_block! on active block returns false" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Root is active, coarsening should fail
    result = coarsen_block!(grid, 1)
    @test result == false
end

# ============================================================
# 6. Prolongation tests
# ============================================================

@testset "prolongate! preserves uniform data" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Fill root with uniform data
    gamma = 1.4
    rho, P = 1.0, 1.0
    E = P / (gamma - 1)
    state = SVector(rho, 0.0, 0.0, E)
    parent = grid.blocks[1]
    for j in 1:4, i in 1:4
        parent.U[i, j] = state
    end

    # Refine
    child_ids = refine_block!(grid, 1)
    children = [grid.blocks[cid] for cid in child_ids]

    # Prolongate
    prolongate!(parent, children, AMR_LAW)

    # All children should have the same uniform state
    for child in children
        for j in 1:4, i in 1:4
            @test child.U[i, j] ≈ state
        end
    end
end

@testset "prolongate! with linear data" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (8, 8), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Fill parent with linear data: rho = 1 + x + y
    parent = grid.blocks[1]
    pdx, pdy = parent.dx
    for j in 1:8, i in 1:8
        xc = parent.origin[1] + (i - 0.5) * pdx
        yc = parent.origin[2] + (j - 0.5) * pdy
        rho = 1.0 + xc + yc
        P = 1.0
        E = P / (1.4 - 1)
        parent.U[i, j] = SVector(rho, 0.0, 0.0, E)
    end

    # Refine
    child_ids = refine_block!(grid, 1)
    children = [grid.blocks[cid] for cid in child_ids]

    # Prolongate
    prolongate!(parent, children, AMR_LAW)

    # Check that child density values are close to the linear function
    # The prolongation uses limited linear interpolation, so for a perfectly
    # linear field the interpolation should be exact (minmod won't limit symmetric slopes)
    for child in children
        cdx, cdy = child.dx
        for j in 1:8, i in 1:8
            xc = child.origin[1] + (i - 0.5) * cdx
            yc = child.origin[2] + (j - 0.5) * cdy
            expected_rho = 1.0 + xc + yc
            # Interior cells (away from boundary) should be very accurate
            if i > 1 && i < 8 && j > 1 && j < 8
                @test child.U[i, j][1] ≈ expected_rho atol = 0.1
            end
        end
    end
end

# ============================================================
# 7. Restriction tests
# ============================================================

@testset "restrict! with uniform data" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Refine root
    child_ids = refine_block!(grid, 1)
    children = [grid.blocks[cid] for cid in child_ids]
    parent = grid.blocks[1]

    # Fill all children with the same uniform state
    gamma = 1.4
    state = SVector(2.0, 1.0, 0.5, 2.0 / (gamma - 1) + 0.5 * (1.0^2 + 0.5^2) / 2.0)
    for child in children
        for j in 1:4, i in 1:4
            child.U[i, j] = state
        end
    end

    # Restrict
    restrict!(parent, children)

    # Parent should have the uniform state
    for j in 1:4, i in 1:4
        @test parent.U[i, j] ≈ state
    end
end

@testset "restrict! averages fine data correctly" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Refine root
    child_ids = refine_block!(grid, 1)
    children = [grid.blocks[cid] for cid in child_ids]
    parent = grid.blocks[1]

    # Fill each child with a different constant density to verify averaging
    # Each coarse cell covers exactly 4 fine cells (one from each child block
    # contributes to a different coarse cell, so for uniform child data
    # the average is just the value)
    gamma = 1.4
    for (idx, child) in enumerate(children)
        rho_val = Float64(idx)
        state = SVector(rho_val, 0.0, 0.0, 1.0 / (gamma - 1))
        for j in 1:4, i in 1:4
            child.U[i, j] = state
        end
    end

    # Restrict
    restrict!(parent, children)

    # The coarse cell values depend on which fine cells map to which coarse cell.
    # Each coarse cell gets contributions from exactly 4 fine cells.
    # With the 4 children having densities 1,2,3,4, the volume average
    # for each coarse cell should be the average of the 4 fine cell values
    # that map to it. Since children cover non-overlapping quadrants:
    # - Each child has 4x4 = 16 fine cells
    # - Parent has 4x4 = 16 coarse cells
    # - Each coarse cell gets 4 fine cells (2x2 from one child)
    # So each coarse cell should equal the child value that covers it.
    # Let us verify by checking specific cells
    # We can verify conservation: total mass should be conserved
    total_fine_mass = 0.0
    for child in children
        for j in 1:4, i in 1:4
            total_fine_mass += child.U[i, j][1]
        end
    end
    total_coarse_mass = 0.0
    for j in 1:4, i in 1:4
        total_coarse_mass += parent.U[i, j][1] * 4  # 4 fine cells per coarse cell
    end
    @test total_coarse_mass ≈ total_fine_mass
end

@testset "restrict! then prolongate! round-trip for uniform data" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Refine root
    child_ids = refine_block!(grid, 1)
    children = [grid.blocks[cid] for cid in child_ids]
    parent = grid.blocks[1]

    # Fill parent with uniform state
    gamma = 1.4
    state = SVector(1.0, 0.0, 0.0, 1.0 / (gamma - 1))
    for j in 1:4, i in 1:4
        parent.U[i, j] = state
    end

    # Prolongate to children
    prolongate!(parent, children, AMR_LAW)

    # Restrict back to parent (zeros parent first, then averages)
    restrict!(parent, children)

    # Should recover the original uniform state
    for j in 1:4, i in 1:4
        @test parent.U[i, j] ≈ state
    end
end

# ============================================================
# 8. FluxRegister tests
# ============================================================

@testset "FluxRegister constructor" begin
    n_faces = 4
    n_fine = 2  # 2D: 2 fine faces per coarse face

    reg = FluxRegister(Val(AMR_NVAR), Float64, n_faces, n_fine)

    @test reg.n_fine == 2

    # All fluxes should be zero-initialized
    zero_state = zero(SVector{AMR_NVAR, Float64})
    for face in (:left, :right, :bottom, :top, :front, :back)
        @test length(reg.coarse_flux[face]) == n_faces
        @test length(reg.fine_flux_sum[face]) == n_faces
        for k in 1:n_faces
            @test reg.coarse_flux[face][k] == zero_state
            @test reg.fine_flux_sum[face][k] == zero_state
        end
    end
end

@testset "reset_flux_register!" begin
    n_faces = 4
    reg = FluxRegister(Val(AMR_NVAR), Float64, n_faces, 2)

    # Store some non-zero data
    flux = SVector(1.0, 2.0, 3.0, 4.0)
    store_coarse_flux!(reg, :left, 1, flux)
    accumulate_fine_flux!(reg, :left, 1, flux)

    @test reg.coarse_flux[:left][1] != zero(SVector{AMR_NVAR, Float64})

    # Reset
    reset_flux_register!(reg)

    # All should be zero again
    zero_state = zero(SVector{AMR_NVAR, Float64})
    for face in (:left, :right, :bottom, :top, :front, :back)
        for k in 1:n_faces
            @test reg.coarse_flux[face][k] == zero_state
            @test reg.fine_flux_sum[face][k] == zero_state
        end
    end
end

@testset "store_coarse_flux! and accumulate_fine_flux!" begin
    n_faces = 4
    n_fine = 2
    reg = FluxRegister(Val(AMR_NVAR), Float64, n_faces, n_fine)

    # Store a coarse flux
    F_coarse = SVector(1.0, 0.5, 0.0, 2.5)
    store_coarse_flux!(reg, :left, 2, F_coarse)
    @test reg.coarse_flux[:left][2] == F_coarse

    # Accumulate two fine fluxes at the same coarse face
    F_fine1 = SVector(0.9, 0.4, 0.0, 2.3)
    F_fine2 = SVector(1.1, 0.6, 0.0, 2.7)
    accumulate_fine_flux!(reg, :left, 2, F_fine1)
    accumulate_fine_flux!(reg, :left, 2, F_fine2)

    @test reg.fine_flux_sum[:left][2] ≈ F_fine1 + F_fine2
end

@testset "apply_flux_correction_2d!" begin
    nx, ny = 4, 4
    n_fine = 2

    # Create a padded U array (nx+4 x ny+4)
    zero_state = zero(SVector{AMR_NVAR, Float64})
    U_pad = fill(zero_state, nx + 4, ny + 4)

    # Fill interior with a known state
    state = SVector(1.0, 0.0, 0.0, 2.5)
    for j in 3:(ny + 2), i in 3:(nx + 2)
        U_pad[i, j] = state
    end

    # Create a flux register
    reg = FluxRegister(Val(AMR_NVAR), Float64, ny, n_fine)

    # Store coarse flux and fine flux with a known discrepancy
    F_coarse = SVector(1.0, 0.5, 0.0, 2.0)
    F_fine_total = SVector(2.2, 1.1, 0.0, 4.4)  # sum of 2 fine fluxes

    for j in 1:ny
        store_coarse_flux!(reg, :left, j, F_coarse)
        # Accumulate fine flux (already summed)
        reg.fine_flux_sum[:left][j] = F_fine_total
    end

    dt = 0.01
    dx = 0.25
    dy = 0.25

    # Apply correction at left face
    apply_flux_correction_2d!(U_pad, reg, dt, dx, dy, nx, ny, :left)

    # The correction for each cell at the left boundary (i=1, padded index=3):
    # dU = dt/dx * (F_fine_avg - F_coarse)
    # F_fine_avg = F_fine_total / n_fine = (2.2, 1.1, 0, 4.4) / 2 = (1.1, 0.55, 0, 2.2)
    # correction = (1.1 - 1.0, 0.55 - 0.5, 0, 2.2 - 2.0) = (0.1, 0.05, 0, 0.2)
    # dU = 0.01/0.25 * (0.1, 0.05, 0, 0.2) = 0.04 * (0.1, 0.05, 0, 0.2)
    F_fine_avg = F_fine_total / n_fine
    correction = F_fine_avg - F_coarse
    expected = state + (dt / dx) * correction

    for j in 1:ny
        @test U_pad[3, j + 2] ≈ expected
    end
end

# ============================================================
# 9. regrid! tests
# ============================================================

@testset "regrid! refines blocks with large gradients" begin
    criterion = GradientRefinement(; variable_index = 1, refine_threshold = 0.01, coarsen_threshold = 0.001)
    grid = AMRGrid(AMR_LAW, criterion, (8, 8), 2, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Insert a density jump in the root block
    root = grid.blocks[1]
    gamma = 1.4
    for j in 1:8, i in 1:8
        if i <= 4
            rho = 1.0
        else
            rho = 0.1
        end
        E = 1.0 / (gamma - 1)
        root.U[i, j] = SVector(rho, 0.0, 0.0, E)
    end

    # Regrid should refine the root
    regrid!(grid)

    @test grid.blocks[1].active == false
    @test length(active_blocks(grid)) == 4
    @test max_active_level(grid) == 1
end

@testset "regrid! coarsens blocks with smooth data" begin
    criterion = GradientRefinement(; variable_index = 1, refine_threshold = 0.5, coarsen_threshold = 0.4)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 2, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Manually refine the root
    child_ids = refine_block!(grid, 1)

    # Fill all children with uniform data (very smooth)
    gamma = 1.4
    state = SVector(1.0, 0.0, 0.0, 1.0 / (gamma - 1))
    for cid in child_ids
        child = grid.blocks[cid]
        for j in 1:4, i in 1:4
            child.U[i, j] = state
        end
    end

    @test max_active_level(grid) == 1

    # Regrid should coarsen back since all children have zero gradient
    regrid!(grid)

    @test grid.blocks[1].active == true
    @test length(active_blocks(grid)) == 1
    @test max_active_level(grid) == 0
end

# ============================================================
# 10. Multi-level refinement tests
# ============================================================

@testset "multi-level refinement and active_blocks sorting" begin
    criterion = GradientRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    grid = AMRGrid(AMR_LAW, criterion, (4, 4), 3, (0.0, 0.0), (1.0, 1.0), Val(AMR_NVAR))

    # Refine root to level 1
    child_ids_l1 = refine_block!(grid, 1)
    @test length(child_ids_l1) == 4

    # Refine one child to level 2
    grandchild_ids = refine_block!(grid, child_ids_l1[1])
    @test length(grandchild_ids) == 4

    # Now we should have:
    # - 3 active blocks at level 1
    # - 4 active blocks at level 2
    # - Total 7 active blocks
    ab = active_blocks(grid)
    @test length(ab) == 7

    # active_blocks should be sorted by level
    levels = [b.level for b in ab]
    @test issorted(levels)

    @test length(blocks_at_level(grid, 0)) == 0
    @test length(blocks_at_level(grid, 1)) == 3
    @test length(blocks_at_level(grid, 2)) == 4
    @test max_active_level(grid) == 2
end

# ============================================================
# 11. AMRGrid with non-unit domain
# ============================================================

@testset "AMRGrid with non-unit domain" begin
    criterion = GradientRefinement()
    domain_lo = (-1.0, -2.0)
    domain_hi = (3.0, 2.0)
    block_size = (8, 8)

    grid = AMRGrid(AMR_LAW, criterion, block_size, 2, domain_lo, domain_hi, Val(AMR_NVAR))

    root = grid.blocks[1]
    @test root.origin == (-1.0, -2.0)
    @test root.dx[1] ≈ (3.0 - (-1.0)) / 8  # 0.5
    @test root.dx[2] ≈ (2.0 - (-2.0)) / 8  # 0.5

    # Verify cell centers
    xc, yc = block_cell_center(root, 1, 1)
    @test xc ≈ -1.0 + 0.5 * 0.5
    @test yc ≈ -2.0 + 0.5 * 0.5
end

# ============================================================
# 12. Type parameterization tests
# ============================================================

@testset "AMRBlock type parameters" begin
    block = AMRBlock(1, 0, (0.0, 0.0), (4, 4), (0.25, 0.25), Val(AMR_NVAR))
    @test block isa AMRBlock{AMR_NVAR, Float64, 2}

    # Verify SVector element type
    @test eltype(block.U) == SVector{AMR_NVAR, Float64}
end

@testset "GradientRefinement and CurrentSheetRefinement are AbstractRefinementCriterion" begin
    @test GradientRefinement <: AbstractRefinementCriterion
    @test CurrentSheetRefinement <: AbstractRefinementCriterion

    crit_g = GradientRefinement()
    crit_c = CurrentSheetRefinement()
    @test crit_g isa AbstractRefinementCriterion
    @test crit_c isa AbstractRefinementCriterion
end

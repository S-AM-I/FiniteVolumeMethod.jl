using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # AMR Sedov Blast
# This tutorial demonstrates block-structured adaptive mesh refinement
# (AMR) for the Sedov blast wave problem. The AMR automatically refines
# around the shock front while keeping the smooth interior at coarse
# resolution, saving computational cost.
#
# ## Problem Setup
# We solve the 2D Euler equations with a point-like energy release,
# using the AMR infrastructure: `AMRGrid`, `AMRProblem`, and `solve_amr`.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{2}(eos)

# ## Creating the AMR Grid
# An `AMRGrid` starts with a single root block covering the entire domain.
# Each block has a fixed number of cells (here $8 \times 8$), and blocks
# can be refined up to `max_level` levels, with each refinement halving
# the grid spacing.
block_size = (8, 8)
max_level = 2
domain_lo = (-1.0, -1.0)
domain_hi = (1.0, 1.0)

# The `GradientRefinement` criterion refines blocks where the density
# gradient exceeds a threshold:
criterion = GradientRefinement(
    variable_index = 1,              ## monitor density (index 1)
    refine_threshold = 0.1,
    coarsen_threshold = 0.01
)

grid = AMRGrid(law, criterion, block_size, max_level, domain_lo, domain_hi, Val(4))

# ## Setting the Initial Condition
# We fill the root block with the Sedov blast wave initial condition:
# uniform density with a high-pressure region near the origin.
root_block = grid.blocks[1]
dx_root = root_block.dx[1]
P_bg = 1.0e-5
P_blast = 1.0
r_blast = 3.0 * dx_root

for j in 1:block_size[2], i in 1:block_size[1]
    xc, yc = block_cell_center(root_block, i, j)
    r = sqrt(xc^2 + yc^2)
    P = r < r_blast ? P_blast : P_bg
    rho = 1.0
    E = P / (gamma - 1)
    root_block.U[i, j] = SVector(rho, 0.0, 0.0, E)
end

# ## Initial Refinement
# Before solving, we refine the grid around the blast to capture
# the initial discontinuity:
regrid!(grid)

# We can inspect the AMR hierarchy:
n_active = length(active_blocks(grid))
max_lev = max_active_level(grid)

# ## Solving with AMR
# The `AMRProblem` packages the grid, solver, and time integration
# parameters. The `solve_amr` function uses Berger-Oliger subcycling:
# finer levels take smaller time steps (half the coarse step).
bcs = (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC())
prob = AMRProblem(
    grid, HLLCSolver(), NoReconstruction(), bcs;
    final_time = 0.05, cfl = 0.3, regrid_interval = 4
)
final_grid, t_final = solve_amr(prob)
final_grid |> tc #hide

# ## Inspecting the Result
n_blocks_final = length(active_blocks(final_grid))
max_lev_final = max_active_level(final_grid)

# ## Visualisation
# We collect cell centres and densities from all active blocks
# to create a scatter plot showing the AMR structure.
using CairoMakie

xs = Float64[]
ys = Float64[]
rhos = Float64[]
levels = Int[]

for block in active_blocks(final_grid)
    nx, ny = block.dims
    for j in 1:ny, i in 1:nx
        xc, yc = block_cell_center(block, i, j)
        push!(xs, xc)
        push!(ys, yc)
        w = conserved_to_primitive(law, block.U[i, j])
        push!(rhos, w[1])
        push!(levels, block.level)
    end
end

fig = Figure(fontsize = 24, size = (1100, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "y",
    title = "Density (AMR, t=$(round(t_final, digits = 3)))",
    aspect = DataAspect()
)
sc1 = scatter!(ax1, xs, ys, color = rhos, markersize = 4, colormap = :inferno)
Colorbar(fig[1, 2], sc1, label = L"\rho")

ax2 = Axis(
    fig[1, 3], xlabel = "x", ylabel = "y",
    title = "AMR levels", aspect = DataAspect()
)
sc2 = scatter!(ax2, xs, ys, color = levels, markersize = 4, colormap = :Set1_3)
Colorbar(fig[1, 4], sc2, label = "Level")
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "amr_sedov_blast.png") fig #src

# The left panel shows the density field with the blast wave, while
# the right panel shows the AMR refinement levels. The grid is
# refined around the shock front where density gradients are large,
# while the smooth interior remains at the coarsest level.
all(rhos .> 0) || @warn("Negative densities detected in AMR Sedov blast") #hide

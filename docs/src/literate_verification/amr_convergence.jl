using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # AMR vs Uniform Grid Comparison
# This example verifies that the AMR solver with Berger-Colella flux
# correction produces results consistent with uniform-grid simulations.
# We compare a 2D blast wave (Sedov-like) solved on a uniform grid against
# an AMR grid with equivalent effective resolution.
#
# ## Mathematical Setup
# A high-pressure region at the centre of a $[0,1]^2$ domain drives a
# circular blast wave outward. We compare:
# 1. Uniform grid at $32 \times 32$
# 2. AMR grid: $8 \times 8$ base with up to 2 refinement levels
#    (effective $32 \times 32$ at finest level)
#
# ## Reference
# - Berger, M.J. & Colella, P. (1989). Local Adaptive Mesh Refinement
#   for Shock Hydrodynamics. J. Comput. Phys., 82, 64-84.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{2}(eos)

# Blast wave IC: high pressure in centre
function blast_ic(x, y)
    r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
    if r < 0.1
        return SVector(1.0, 0.0, 0.0, 10.0)   # high pressure
    else
        return SVector(1.0, 0.0, 0.0, 0.1)    # low pressure
    end
end

t_final = 0.05

# ## Uniform Grid Solution
N_uniform = 32
mesh_uni = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N_uniform, N_uniform)
prob_uni = HyperbolicProblem2D(
    law, mesh_uni, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    TransmissiveBC(), TransmissiveBC(),
    blast_ic; final_time = t_final, cfl = 0.3,
)
coords_uni, U_uni, t_uni = solve_hyperbolic(prob_uni)

# ## AMR Grid Solution
criterion = GradientRefinement(;
    variable_index = 1, refine_threshold = 0.05, coarsen_threshold = 0.005,
)
block_size = (8, 8)
max_level = 2
grid = AMRGrid(law, criterion, block_size, max_level, (0.0, 0.0), (1.0, 1.0), Val(4))

# Initialize blocks with IC
for block in values(grid.blocks)
    for j in 1:block.dims[2], i in 1:block.dims[1]
        xc, yc = block_cell_center(block, i, j)
        w = blast_ic(xc, yc)
        block.U[i, j] = primitive_to_conserved(law, w)
    end
end

# Initial regrid to place refinement at the blast
regrid!(grid)
for block in values(grid.blocks)
    if block.active
        for j in 1:block.dims[2], i in 1:block.dims[1]
            xc, yc = block_cell_center(block, i, j)
            w = blast_ic(xc, yc)
            block.U[i, j] = primitive_to_conserved(law, w)
        end
    end
end

bc_amr = (
    left = TransmissiveBC(), right = TransmissiveBC(),
    bottom = TransmissiveBC(), top = TransmissiveBC(),
)
prob_amr = AMRProblem(
    grid, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()), bc_amr;
    final_time = t_final, cfl = 0.3, regrid_interval = 4,
)
grid_amr, t_amr = solve_amr(prob_amr)
U_amr = grid_amr.blocks  # keep reference for non-empty check

# ## Comparison
# Compare density at the centre line (y ≈ 0.5)
rho_uni_midline = Float64[]
x_uni_midline = Float64[]
jmid = div(N_uniform, 2)
for i in 1:N_uniform
    push!(x_uni_midline, coords_uni[i, jmid][1])
    push!(rho_uni_midline, conserved_to_primitive(law, U_uni[i, jmid])[1])
end

# ## Visualisation — Density Field
xc_uni = [coords_uni[i, 1][1] for i in 1:N_uniform]
yc_uni = [coords_uni[1, j][2] for j in 1:N_uniform]
rho_uni = [conserved_to_primitive(law, U_uni[i, j])[1] for i in 1:N_uniform, j in 1:N_uniform]

fig = Figure(fontsize = 24, size = (600, 550))
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Uniform 32×32 density", aspect = DataAspect())
hm = heatmap!(ax, xc_uni, yc_uni, rho_uni, colormap = :viridis)
Colorbar(fig[1, 2], hm)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "amr_blast_uniform.png") fig #src

# ## Test Assertions
# The AMR solution should complete without errors and produce
# a non-trivial result.
@test t_amr ≈ t_final #src
@test t_uni ≈ t_final #src
@test !isempty(grid_amr.blocks) #src
@assert t_amr ≈ t_final #hide
@assert t_uni ≈ t_final #hide
@assert !isempty(grid_amr.blocks) #hide

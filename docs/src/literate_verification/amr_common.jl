using FiniteVolumeMethod
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase: ReturnCode
using StaticArrays

const AMR_GAMMA = 1.4
const AMR_EOS = IdealGasEOS(AMR_GAMMA)
const AMR_LAW = EulerEquations{2}(AMR_EOS)
const AMR_SOLVER = HLLCSolver()
const AMR_RECONSTRUCTION = CellCenteredMUSCL(MinmodLimiter())
const AMR_BCS = (
    left = TransmissiveBC(),
    right = TransmissiveBC(),
    bottom = TransmissiveBC(),
    top = TransmissiveBC(),
)

const AMR_RHO_BACKGROUND = 1.0
const AMR_PRESSURE_BACKGROUND = 1.0
const AMR_VX = 0.25
const AMR_VY = 0.15
const AMR_PULSE_AMPLITUDE = 0.35
const AMR_PULSE_SIGMA = 0.08
const AMR_PULSE_X0 = 0.35
const AMR_PULSE_Y0 = 0.4
const AMR_VERIFICATION_FINAL_TIME = 0.1
const AMR_DYNAMIC_FINAL_TIME = 0.08
const AMR_CFL = 0.35

const AMR_CONSERVED_NAMES = ["mass", "momentum_x", "momentum_y", "energy"]

function transported_entropy_pulse(x, y, t = 0.0)
    x_shifted = x - AMR_VX * t
    y_shifted = y - AMR_VY * t
    rho = AMR_RHO_BACKGROUND + AMR_PULSE_AMPLITUDE *
        exp(-((x_shifted - AMR_PULSE_X0)^2 + (y_shifted - AMR_PULSE_Y0)^2) / (2 * AMR_PULSE_SIGMA^2))
    return SVector(rho, AMR_VX, AMR_VY, AMR_PRESSURE_BACKGROUND)
end

transported_entropy_density(x, y, t = 0.0) = transported_entropy_pulse(x, y, t)[1]

function refill_grid!(grid; time = 0.0)
    for block in values(grid.blocks)
        for j in 1:block.dims[2], i in 1:block.dims[1]
            x, y = block_cell_center(block, i, j)
            block.U[i, j] = primitive_to_conserved(AMR_LAW, transported_entropy_pulse(x, y, time))
        end
    end
    return grid
end

function build_amr_grid(
        base_cells,
        max_level;
        refine_threshold = 0.03,
        coarsen_threshold = 0.005,
    )
    criterion = GradientRefinement(
        variable_index = 1,
        refine_threshold = refine_threshold,
        coarsen_threshold = coarsen_threshold,
    )
    grid = AMRGrid(
        AMR_LAW,
        criterion,
        (base_cells, base_cells),
        max_level,
        (0.0, 0.0),
        (1.0, 1.0),
        Val(4),
    )
    refill_grid!(grid)
    regrid!(grid)
    refill_grid!(grid)
    return grid
end

function active_block_count(grid)
    return count(block -> block.active, values(grid.blocks))
end

function active_cell_count(grid)
    return sum(prod(block.dims) for block in values(grid.blocks) if block.active)
end

function fixed_hierarchy_amr_case(
        base_cells;
        max_level = 1,
        final_time = AMR_VERIFICATION_FINAL_TIME,
        cfl = AMR_CFL,
    )
    grid = build_amr_grid(base_cells, max_level)
    prob = AMRProblem(
        grid,
        AMR_SOLVER,
        AMR_RECONSTRUCTION,
        AMR_BCS;
        final_time,
        cfl,
        regrid_interval = 0,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    return (; prob, sol, dt0, accessor = solution_accessor(prob))
end

function fixed_hierarchy_exact_error(
        base_cells;
        max_level = 1,
        final_time = AMR_VERIFICATION_FINAL_TIME,
        cfl = AMR_CFL,
    )
    case = fixed_hierarchy_amr_case(base_cells; max_level, final_time, cfl)
    coords = solution_coordinates(case.accessor)
    primitive = get_primitive(case.accessor, case.sol, length(case.sol.t))

    density_error = 0.0
    for (block_id, block_coords) in coords
        dV = prod(case.prob.grid.blocks[block_id].dx)
        block_primitive = primitive[block_id]
        for j in axes(block_coords, 2), i in axes(block_coords, 1)
            x, y = block_coords[i, j]
            density_error += abs(block_primitive[i, j][1] - transported_entropy_density(x, y, final_time)) * dV
        end
    end

    return (
        retcode = case.sol.retcode,
        final_time = case.sol.t[end],
        density_error = density_error,
        active_cells = active_cell_count(case.prob.grid),
        active_blocks = active_block_count(case.prob.grid),
        levels = max_active_level(case.prob.grid),
    )
end

function uniform_reference_density(reference_cells; final_time = AMR_DYNAMIC_FINAL_TIME, cfl = AMR_CFL)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, reference_cells, reference_cells)
    prob = HyperbolicProblem2D(
        AMR_LAW,
        mesh,
        AMR_SOLVER,
        AMR_RECONSTRUCTION,
        TransmissiveBC(),
        TransmissiveBC(),
        TransmissiveBC(),
        TransmissiveBC(),
        (x, y) -> transported_entropy_pulse(x, y, 0.0);
        final_time,
        cfl,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    accessor = solution_accessor(prob)
    primitive = get_primitive(accessor, sol, length(sol.t))
    density = [state[1] for state in primitive]
    return (; density, retcode = sol.retcode, final_time = sol.t[end])
end

function sample_uniform_density(reference_density, reference_cells, x, y)
    ix = clamp(round(Int, x * reference_cells + 0.5), 1, reference_cells)
    iy = clamp(round(Int, y * reference_cells + 0.5), 1, reference_cells)
    return reference_density[(iy - 1) * reference_cells + ix]
end

function dynamic_reference_tracking_case(
        base_cells;
        max_level = 2,
        reference_density,
        reference_cells,
        final_time = AMR_DYNAMIC_FINAL_TIME,
        cfl = AMR_CFL,
    )
    grid = build_amr_grid(base_cells, max_level)
    initial_active_blocks = active_block_count(grid)
    prob = AMRProblem(
        grid,
        AMR_SOLVER,
        AMR_RECONSTRUCTION,
        AMR_BCS;
        final_time,
        cfl,
        regrid_interval = 1,
    )
    grid_out, t_final = solve_amr(prob)

    density_error = 0.0
    for block in values(grid_out.blocks)
        block.active || continue
        dV = prod(block.dx)
        for j in 1:block.dims[2], i in 1:block.dims[1]
            x, y = block_cell_center(block, i, j)
            rho = conserved_to_primitive(AMR_LAW, block.U[i, j])[1]
            density_error += abs(rho - sample_uniform_density(reference_density, reference_cells, x, y)) * dV
        end
    end

    active_cells = active_cell_count(grid_out)
    return (
        density_error = density_error,
        active_cells = active_cells,
        active_blocks = active_block_count(grid_out),
        initial_active_blocks = initial_active_blocks,
        compression = active_cells / (reference_cells^2),
        levels = max_active_level(grid_out),
        final_time = t_final,
    )
end

function conserved_totals(grid)
    totals = zeros(4)
    for block in values(grid.blocks)
        block.active || continue
        dV = prod(block.dx)
        for j in 1:block.dims[2], i in 1:block.dims[1]
            totals .+= block.U[i, j] .* dV
        end
    end
    return totals
end

function dynamic_conservation_case(
        base_cells;
        max_level = 2,
        final_time = AMR_DYNAMIC_FINAL_TIME,
        cfl = AMR_CFL,
    )
    grid = build_amr_grid(base_cells, max_level)
    q0 = conserved_totals(grid)
    prob = AMRProblem(
        grid,
        AMR_SOLVER,
        AMR_RECONSTRUCTION,
        AMR_BCS;
        final_time,
        cfl,
        regrid_interval = 1,
    )
    grid_out, t_final = solve_amr(prob)
    q1 = conserved_totals(grid_out)
    relative_drift = abs.((q1 .- q0) ./ q0)
    return (
        initial_totals = q0,
        final_totals = q1,
        relative_drift = relative_drift,
        max_relative_drift = maximum(relative_drift),
        active_cells = active_cell_count(grid_out),
        active_blocks = active_block_count(grid_out),
        levels = max_active_level(grid_out),
        final_time = t_final,
    )
end

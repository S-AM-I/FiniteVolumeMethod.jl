# System assembly for coupled physics
# Migrated from Simu.jl SimuFVM/assembly/assembly_system.jl

"""Abstract supertype for inter-equation coupling terms in multi-physics systems."""
abstract type AbstractCoupling end

"""
    LinearCoupling

Represents a linear coupling term to be added to the system matrix (LHS).
Adds `coeff * phi_source` to the equation for `target_idx`.

If your equation is `L(u) = c * v` (source term on RHS), rearranging to `L(u) - c * v = 0` means
you should provide a coefficient of `-c`.

Fields:
- `target_idx::Int`: Index of the equation (row block) to add the term to.
- `source_idx::Int`: Index of the variable (column block) that is the source of the coupling.
- `coeff`: Coefficient. Can be Float64 (constant), Vector{Float64} (spatial), or Function.
"""
struct LinearCoupling <: AbstractCoupling
    target_idx::Int
    source_idx::Int
    coeff::Union{Float64, Vector{Float64}, Function}
end

"""
    assemble_coupled_system(models, mesh, bcs_list; couplings=[], transient=false, source_list=nothing)

Assembles a block-coupled system of equations using efficient triplet assembly.

# Arguments
- `models`: Vector of equation models.
- `mesh`: The computational mesh.
- `bcs_list`: Vector of boundary conditions.
- `couplings`: Vector of `AbstractCoupling` definitions.
- `transient`: Boolean.
- `source_list`: Vector of source terms.

# Returns
- `A_global`: The global sparse matrix.
- `b_global`: The global RHS vector.
"""
function assemble_coupled_system(
        models::Vector, mesh, bcs_list::Vector;
        couplings::Vector{AbstractCoupling} = AbstractCoupling[],
        transient = false,
        source_list = nothing
    )

    num_models = length(models)

    # 1. Assemble individual systems to get local matrices and RHS
    blocks_diag = Vector{SparseArrays.SparseMatrixCSC}(undef, num_models)
    rhss = Vector{Vector{Float64}}(undef, num_models)

    for i in 1:num_models
        source = (source_list !== nothing && length(source_list) >= i) ? source_list[i] : nothing
        A, b = assemble_system(models[i], mesh, bcs_list[i]; transient = transient, source = source)
        blocks_diag[i] = A
        rhss[i] = b
    end

    num_cells = length(rhss[1])
    total_dofs = num_models * num_cells

    # 2. Collect Triplets for Global Matrix
    I_global = Int[]
    J_global = Int[]
    V_global = Float64[]

    # Estimate non-zeros to reserve memory
    sizehint!(I_global, total_dofs * 7)
    sizehint!(J_global, total_dofs * 7)
    sizehint!(V_global, total_dofs * 7)

    # Helper to add block
    function add_block!(block_row, block_col, A_local::SparseArrays.SparseMatrixCSC)
        rows = SparseArrays.rowvals(A_local)
        vals = SparseArrays.nonzeros(A_local)
        m, n = size(A_local)
        row_offset = (block_row - 1) * num_cells
        col_offset = (block_col - 1) * num_cells

        for j in 1:n
            for k in SparseArrays.nzrange(A_local, j)
                row = rows[k]
                val = vals[k]
                push!(I_global, row + row_offset)
                push!(J_global, j + col_offset)
                push!(V_global, val)
            end
        end
        return
    end

    # Add Diagonal Blocks
    for i in 1:num_models
        add_block!(i, i, blocks_diag[i])
    end

    # Add Off-Diagonal Couplings
    for coupling in couplings
        if coupling isa LinearCoupling
            i, j = coupling.target_idx, coupling.source_idx
            C = build_linear_coupling_block(mesh, coupling.coeff)
            add_block!(i, j, C)
        end
    end

    # 3. Create Global Matrix
    A_global = sparse(I_global, J_global, V_global, total_dofs, total_dofs)

    # 4. Create Global RHS
    b_global = reduce(vcat, rhss)

    return A_global, b_global
end

"""Build a diagonal coupling matrix block with uniform coefficient `coeff` for the given mesh."""
function build_linear_coupling_block(mesh, coeff::Float64)
    n = length(mesh.cells)
    V = [mesh.cells[k].volume for k in 1:n]
    return SparseArrays.spdiagm(0 => coeff .* V)
end

function build_linear_coupling_block(mesh, coeff::Vector{Float64})
    n = length(mesh.cells)
    V = [mesh.cells[k].volume for k in 1:n]
    return SparseArrays.spdiagm(0 => coeff .* V)
end

function build_linear_coupling_block(mesh, coeff::Function)
    n = length(mesh.cells)
    vals = Float64[]
    for k in 1:n
        cell = mesh.cells[k]
        val = 0.0
        if length(cell.center) == 3
            val = coeff(cell.center[1], cell.center[2], cell.center[3])
        elseif length(cell.center) == 2
            val = coeff(cell.center[1], cell.center[2])
        else
            val = coeff(cell.center[1])
        end
        push!(vals, val * cell.volume)
    end
    return SparseArrays.spdiagm(0 => vals)
end

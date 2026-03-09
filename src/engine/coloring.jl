# coloring.jl - Graph coloring for efficient Jacobian computation
# Migrated from Simu.jl SimuEngine/coloring.jl
# SparseArrays and LinearAlgebra are already imported by the parent module.

"""
    color_graph_greedy(adj_list)

Greedy graph coloring algorithm.
adj_list: Vector{Vector{Int}}, where adj_list[i] contains neighbors of node i.
Returns a vector of colors (integers) for each node.
"""
function color_graph_greedy(adj_list::Vector{Vector{Int}})
    n = length(adj_list)
    colors = zeros(Int, n)
    forbidden = fill(false, n + 1) # +1 to handle 1-based indexing safety

    for u in 1:n
        # Reset forbidden colors
        # Only check neighbors that are already colored
        for v in adj_list[u]
            if colors[v] != 0
                if colors[v] <= n
                    forbidden[colors[v]] = true
                end
            end
        end

        # Find first non-forbidden color
        c = 1
        while forbidden[c]
            c += 1
        end
        colors[u] = c

        # Reset forbidden array for next iteration (optimization: only reset used ones)
        for v in adj_list[u]
            if colors[v] != 0 && colors[v] <= n
                forbidden[colors[v]] = false
            end
        end
    end

    return colors
end

"""
    compute_colored_jacobian!(J, f!, x, coloring; epsilon=1e-8)

Compute Jacobian using graph coloring to reduce function evaluations.
Number of evaluations = Number of colors + 1.

J must be a pre-allocated SparseMatrixCSC with the correct sparsity pattern.
The coloring must be structurally orthogonal with respect to J's sparsity.
"""
function compute_colored_jacobian!(J::SparseArrays.SparseMatrixCSC, f!, x::Vector{Float64}, coloring::Vector{Int}; epsilon = 1.0e-8)
    n = length(x)
    num_colors = maximum(coloring)

    fx = zeros(n)
    f!(fx, x)

    x_perturb = copy(x)
    f_perturb = zeros(n)

    # Iterate over colors
    for c in 1:num_colors
        # Reset perturbation
        x_perturb .= x

        cols_c = findall(coloring .== c)

        for j in cols_c
            x_perturb[j] += epsilon
        end

        f!(f_perturb, x_perturb)

        # Difference vector
        diff = f_perturb - fx

        # Fill Jacobian: for each column j with color c, recover J[:, j]
        # Structural orthogonality guarantees that for a given row i,
        # only one column with color c is non-zero.
        rows = SparseArrays.rowvals(J)
        vals = SparseArrays.nonzeros(J)

        for j in cols_c
            for k in SparseArrays.nzrange(J, j)
                i = rows[k]
                vals[k] = diff[i] / epsilon
            end
        end
    end
    return
end

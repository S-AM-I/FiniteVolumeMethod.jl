# # P1 Radiation Slab Attenuation
# This case verifies `solve_p1_radiation` against the closed-form 1D
# solution of the P1 attenuation problem in a cold medium ($T_m = 0$):
# ```math
# -\frac{1}{3a} \frac{\mathrm{d}^2 G}{\mathrm{d}x^2} + a G = 0
# \quad\Longrightarrow\quad
# G(x) = G_0 \frac{\sinh(\sqrt{3}\,a\,(L - x))}{\sinh(\sqrt{3}\,a\,L)},
# ```
# satisfying $G(0) = G_0$ and $G(L) = 0$. Neumann top/bottom boundaries
# make the 2D solve strictly 1D, which is itself checked (column spread
# below $10^{-10}$).
#
# ## Acceptance Gates
# - Monotone decay along $x$, strictly positive $G$, endpoints within 10%
#   of $G_0$ and 0
# - y-independence: column spread $< 10^{-10}$
# - $O(h^2)$ interior grid convergence: observed order in $(1.8, 2.2)$
#   across $N \in \{20, 40, 80\}$, monotone, finest error $< 10^{-4}$

using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using LinearSolve
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

L = 1.0
a_coeff = 1.0
G0 = 1.0

G_exact(x) = G0 * sinh(sqrt(3.0) * a_coeff * (L - x)) / sinh(sqrt(3.0) * a_coeff * L)

# ## Solve
# The linear solver is pinned for cross-platform determinism.
function solve_slab(Nx, Ny)
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, L, 0.2)
    rad = P1Model(; a = a_coeff)
    T_field = CollocatedScalarField(:T, mesh; value = 0.0)
    bcs_G = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(G0),
        :right => DirichletBC(0.0),
        :bottom => NeumannBC(0.0),
        :top => NeumannBC(0.0),
    )
    G = solve_p1_radiation(rad, T_field, mesh, bcs_G; linear_solver = LUFactorization())
    return mesh, G
end

function slab_l2_error(Nx, Ny)
    mesh, G = solve_slab(Nx, Ny)
    err_sq = 0.0
    vol = 0.0
    for c in 1:length(mesh.cell_volumes)
        x = mesh.cell_centers[1, c]
        if 0.1 < x < 0.9
            err_sq += mesh.cell_volumes[c] * (G.internal[c] - G_exact(x))^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

# ## Shape and 1D Invariance
mesh_shape, G_shape = solve_slab(40, 4)
columns = Dict{Float64, Vector{Float64}}()
for c in 1:length(mesh_shape.cell_volumes)
    x = round(mesh_shape.cell_centers[1, c]; digits = 10)
    push!(get!(columns, x, Float64[]), G_shape.internal[c])
end
xs = sort(collect(keys(columns)))
column_G = [first(columns[x]) for x in xs]
monotone = all(diff(column_G) .<= 1.0e-10)
positive = all(>(0.0), G_shape.internal)
endpoints_ok = column_G[1] > 0.9 * G0 && column_G[end] < 0.1 * G0

mesh_y, G_y = solve_slab(20, 8)
buckets_y = Dict{Float64, Vector{Float64}}()
for c in 1:length(mesh_y.cell_volumes)
    x = round(mesh_y.cell_centers[1, c]; digits = 10)
    push!(get!(buckets_y, x, Float64[]), G_y.internal[c])
end
column_spread = maximum(
    length(vals) >= 2 ? maximum(vals) - minimum(vals) : 0.0
        for (_, vals) in buckets_y
)

# ## Grid Convergence
mesh_sizes = [20, 40, 80]
errors_L2 = [slab_l2_error(N, 4) for N in mesh_sizes]
orders = [log2(errors_L2[i] / errors_L2[i + 1]) for i in 1:(length(errors_L2) - 1)]

# ## Visualisation — Profile and Convergence
x_dense = range(0.0, L; length = 200)
fig1 = Figure(fontsize = 24, size = (1000, 450))
ax1 = Axis(
    fig1[1, 1], xlabel = "x", ylabel = "G(x)",
    title = "P1 slab attenuation"
)
lines!(ax1, x_dense, G_exact.(x_dense), color = :black, linewidth = 2, label = "Exact")
scatter!(ax1, xs, column_G, color = :red, markersize = 10, label = "P1 (40 × 4)")
axislegend(ax1, position = :rt)
h_vals = [L / N for N in mesh_sizes]
ax2 = Axis(
    fig1[1, 2], xlabel = "h", ylabel = "L² error",
    xscale = log10, yscale = log10, title = "Convergence"
)
scatterlines!(
    ax2, h_vals, errors_L2, marker = :circle, color = :blue,
    linewidth = 2, markersize = 12
)
lines!(
    ax2, h_vals, errors_L2[1] .* (h_vals ./ h_vals[1]) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1.5
)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("p1_slab_attenuation.png"), fig1)
end

# ## Acceptance
@test monotone #src
@test positive #src
@test endpoints_ok #src
@test column_spread < 1.0e-10 #src
for p in orders #src
    @test 1.8 < p < 2.2 #src
end #src
@test all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)) #src
@test errors_L2[end] < 1.0e-4 #src
@assert monotone #hide
@assert positive #hide
@assert endpoints_ok #hide
@assert column_spread < 1.0e-10 #hide
@assert all(p -> 1.8 < p < 2.2, orders) #hide
@assert all(errors_L2[i] > errors_L2[i + 1] for i in 1:(length(errors_L2) - 1)) #hide
@assert errors_L2[end] < 1.0e-4 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "l2_errors" => errors_L2,
            "observed_orders" => orders,
            "column_spread" => column_spread,
        ),
        artifacts = ["p1_slab_attenuation.png"],
        notes = [
            "Verification-stage exact-solution evidence for radiation: solve_p1_radiation against the closed-form cold-medium slab attenuation, with O(h^2) grid convergence and strict 1D invariance.",
        ],
        summary = Dict(
            "mesh_sizes" => mesh_sizes,
            "absorption" => a_coeff,
        ),
    )
end

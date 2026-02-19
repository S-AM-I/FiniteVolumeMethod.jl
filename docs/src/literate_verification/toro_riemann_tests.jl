using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Toro Riemann Problem Suite
# This example runs the five standard Euler Riemann problems from Toro (2009),
# Table 4.1. Each test exercises different aspects of the Riemann solver:
# rarefactions, contacts, shocks, near-vacuum, and strong discontinuities.
#
# ## Reference
# - Toro, E.F. (2009). Riemann Solvers and Numerical Methods for Fluid
#   Dynamics, 3rd ed., Springer. Chapter 4, Table 4.1.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)
N = 400

# ## Exact Riemann Solver (star-state values from Toro Table 4.1)
toro_tests = [
    (
        name = "Sod", rhoL = 1.0, uL = 0.0, pL = 1.0,
        rhoR = 0.125, uR = 0.0, pR = 0.1,
        x0 = 0.5, t = 0.2, p_star = 0.30313, u_star = 0.92745,
    ),
    (
        name = "123 (two rarefactions)", rhoL = 1.0, uL = -2.0, pL = 0.4,
        rhoR = 1.0, uR = 2.0, pR = 0.4,
        x0 = 0.5, t = 0.15, p_star = 0.00189, u_star = 0.0,
    ),
    (
        name = "Woodward-Colella blast", rhoL = 1.0, uL = 0.0, pL = 1000.0,
        rhoR = 1.0, uR = 0.0, pR = 0.01,
        x0 = 0.5, t = 0.012, p_star = 460.894, u_star = 19.5975,
    ),
    (
        name = "Two-shock collision", rhoL = 5.99924, uL = 19.5975, pL = 460.894,
        rhoR = 5.99242, uR = -6.19633, pR = 46.095,
        x0 = 0.4, t = 0.035, p_star = 1691.64, u_star = 8.68975,
    ),
    (
        name = "Stationary contact", rhoL = 1.0, uL = -19.5975, pL = 1000.0,
        rhoR = 1.0, uR = -19.5975, pR = 0.01,
        x0 = 0.8, t = 0.012, p_star = 460.894, u_star = 0.0,
    ),
]

# ## Run All Five Tests
results = map(toro_tests) do tt
    wL = SVector(tt.rhoL, tt.uL, tt.pL)
    wR = SVector(tt.rhoR, tt.uR, tt.pR)
    ic(x) = x < tt.x0 ? wL : wR
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR), ic;
        final_time = tt.t, cfl = 0.4,
    )
    x, U, t_end = solve_hyperbolic(prob)
    W = [conserved_to_primitive(law, U[i]) for i in eachindex(U)]
    (x = x, W = W, t = t_end, ref = tt)
end

# ## Visualisation — All Five Tests
fig = Figure(fontsize = 18, size = (1500, 900))
for (idx, r) in enumerate(results)
    row = (idx - 1) ÷ 3 + 1
    col = (idx - 1) % 3 + 1
    ax = Axis(
        fig[row, col], xlabel = "x", ylabel = L"\rho",
        title = "Test $(idx): $(r.ref.name)"
    )
    rho = [r.W[i][1] for i in eachindex(r.W)]
    lines!(ax, r.x, rho, color = :blue, linewidth = 1.5)
end
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "toro_riemann_tests.png") fig #src

# ## Quantitative Verification
# For each test, find the numerical pressure in the star region
# (the plateau between the waves) and compare to the exact value.
# The contact travels at speed `u_star` from the initial discontinuity `x0`,
# so we sample near `x0 + u_star * t`.
function measure_star_pressure(W, x, x_contact)
    ## Sample cells near the contact location
    i_centre = argmin(abs.(x .- x_contact))
    ## Average pressure over a few cells near the contact
    i_lo = max(1, i_centre - 5)
    i_hi = min(length(x), i_centre + 5)
    return sum(W[i][3] for i in i_lo:i_hi) / (i_hi - i_lo + 1)
end

star_pressures = [
    measure_star_pressure(r.W, r.x, r.ref.x0 + r.ref.u_star * r.ref.t) for r in results
]
exact_pressures = [r.ref.p_star for r in results]
rel_errors = [abs(star_pressures[i] - exact_pressures[i]) / max(abs(exact_pressures[i]), 1.0e-10) for i in eachindex(results)]

# ## Test Assertions
# Star-region pressure should match published values within tolerance.
# Test 2 (near-vacuum, p_star ≈ 0.002) uses absolute tolerance since
# the relative error is dominated by numerics at near-zero pressure.
for i in eachindex(results)
    if toro_tests[i].p_star < 0.01 #src
        @test abs(star_pressures[i] - exact_pressures[i]) < 0.01 #src
    else #src
        @test rel_errors[i] < 0.1 #src
    end #src
end
for i in eachindex(results) #hide
    if toro_tests[i].p_star < 0.01 #hide
        @assert abs(star_pressures[i] - exact_pressures[i]) < 0.01 "Test $(i): abs error $(abs(star_pressures[i] - exact_pressures[i]))" #hide
    else #hide
        @assert rel_errors[i] < 0.1 "Test $(i) ($(toro_tests[i].name)): rel error $(rel_errors[i])" #hide
    end #hide
end #hide

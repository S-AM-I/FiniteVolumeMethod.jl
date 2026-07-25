# test/v_and_v_couette.jl — Plane Couette flow V&V (v3.22)
#
# Verifies the steady collocated SIMPLE solver against the
# classical Couette flow analytical solution: linear velocity
# profile driven by shear from a moving plate with no pressure
# gradient.
#
# Geometry: [0, L] × [0, H] with moving top plate.
#   u(y) = U_top · y / H   (linear in y)
#   v(y) = 0
#   p    = const
#
# This is the third analytical benchmark for `incompressible_ns`,
# complementing Ghia Re = 100 (v3.1) and Poiseuille parabolic
# (v3.10, grid-convergence v3.11). Together they cover:
#   • Lid-driven recirculation (Ghia)
#   • Pressure-driven laminar channel (Poiseuille)
#   • Shear-driven parallel flow (Couette).
#
# Evidence toward eventual `stable` promotion per the project's
# "3+ published benchmarks per feature" gate.

using FiniteVolumeMethod
using FiniteVolumeMethod: SpatialVelocityBC
using LinearSolve
using StaticArrays: SVector
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: Couette — linear velocity profile" begin
    H = 1.0
    L = 4.0
    U_top = 1.0

    Nx = 40
    Ny = 20
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, L, H)

    # Inlet: prescribed linear u(y). Outlet: fixed pressure.
    # Top: moving wall at velocity (U_top, 0). Bottom: no-slip.
    u_inlet = x -> SVector(U_top * x[2] / H, 0.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => SpatialVelocityBC(u_inlet, Val(2), Float64),
        :right => FixedPressureBC(0.0),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(U_top, 0.0)),
    )

    algo = SIMPLE(0.5, 0.2, 500, 1.0e-6)
    prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = 0.1, density = 1.0)
    sol = solve(prob, algo)

    @test sol.result.iterations > 0

    # Sample centerline column at x = L/2.
    i_mid = 20
    u_num = [sol.result.state.U.internal[(j - 1) * Nx + i_mid][1] for j in 1:Ny]
    v_num = [sol.result.state.U.internal[(j - 1) * Nx + i_mid][2] for j in 1:Ny]
    y_mesh = [mesh.cell_centers[2, (j - 1) * Nx + i_mid] for j in 1:Ny]

    # Analytical: u = U_top · y / H. Measure max relative error in
    # the fully-developed interior band (0.1 H < y < 0.9 H).
    max_rel_u = 0.0
    for (y, u) in zip(y_mesh, u_num)
        u_ex = U_top * y / H
        if y > 0.1 * H && y < 0.9 * H
            max_rel_u = max(max_rel_u, abs(u - u_ex) / max(u_ex, 1.0e-6))
        end
    end
    @test max_rel_u < 0.05

    # Transverse velocity should be essentially zero.
    max_v = maximum(abs, v_num)
    @test max_v < 0.05

    # Monotone u(y): Couette is linear and strictly increasing in y.
    sorted_pairs = sort!(collect(zip(y_mesh, u_num)); by = first)
    sorted_u = [p[2] for p in sorted_pairs]
    # Interior columns must be monotone within numerical tolerance.
    diffs = diff(sorted_u)
    @test all(d -> d > -0.01, diffs)
end

@testset "V&V: Couette — no pressure gradient" begin
    H = 1.0
    L = 4.0
    U_top = 1.0

    Nx = 40
    Ny = 20
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, L, H)

    u_inlet = x -> SVector(U_top * x[2] / H, 0.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => SpatialVelocityBC(u_inlet, Val(2), Float64),
        :right => FixedPressureBC(0.0),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(U_top, 0.0)),
    )

    algo = SIMPLE(0.5, 0.2, 500, 1.0e-6)
    prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = 0.1, density = 1.0)
    sol = solve(prob, algo)

    # Analytical Couette has ∂p/∂x = 0, so the pressure field should
    # be (nearly) uniform across the channel. Measure mid-column
    # pressures at two streamwise stations; they should match.
    p_field = sol.result.state.p.internal

    x_left = 10
    x_right = 30
    j_mid = div(Ny, 2)

    p_left = p_field[(j_mid - 1) * Nx + x_left]
    p_right = p_field[(j_mid - 1) * Nx + x_right]

    # Streamwise pressure drop should be small compared to the
    # characteristic dynamic pressure (1/2)·ρ·U² = 0.5.
    @test abs(p_left - p_right) < 0.05
end

@testset "V&V: Couette — velocity is linear in y at fixed mesh" begin
    # Linear regression: fit u = a + b·y on the mid-column; the
    # slope should match U_top / H and the residual from the linear
    # fit should be small (u is linear, so quadratic-or-higher
    # residual measures departure from analytical).
    H = 1.0
    L = 4.0
    U_top = 1.0

    Nx = 40
    Ny = 20
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, L, H)
    u_inlet = x -> SVector(U_top * x[2] / H, 0.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => SpatialVelocityBC(u_inlet, Val(2), Float64),
        :right => FixedPressureBC(0.0),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(U_top, 0.0)),
    )

    algo = SIMPLE(0.5, 0.2, 500, 1.0e-6)
    prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = 0.1, density = 1.0)
    sol = solve(prob, algo)

    i_mid = 20
    ys = Float64[]
    us = Float64[]
    for j in 1:Ny
        y = mesh.cell_centers[2, (j - 1) * Nx + i_mid]
        if y > 0.1 * H && y < 0.9 * H
            push!(ys, y)
            push!(us, sol.result.state.U.internal[(j - 1) * Nx + i_mid][1])
        end
    end

    # Least-squares linear fit u ≈ a + b·y.
    n = length(ys)
    y_bar = sum(ys) / n
    u_bar = sum(us) / n
    num = sum((ys[i] - y_bar) * (us[i] - u_bar) for i in 1:n)
    den = sum((ys[i] - y_bar)^2 for i in 1:n)
    b_fit = num / den
    a_fit = u_bar - b_fit * y_bar

    # Slope: U_top / H = 1.0. Intercept: 0.
    @test abs(b_fit - U_top / H) < 0.05
    @test abs(a_fit) < 0.05

    # Residual from linear fit (max across the band).
    max_resid = 0.0
    for i in 1:n
        max_resid = max(max_resid, abs(us[i] - (a_fit + b_fit * ys[i])))
    end
    @test max_resid < 0.02
end

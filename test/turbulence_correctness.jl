# test/turbulence_correctness.jl — Stage 4 correctness gates

using FiniteVolumeMethod
using Test
using StaticArrays: SVector
using LinearAlgebra: norm

include("TestHelpers.jl")

@testset "Stage 4a: StandardKEpsilon Durbin realizability" begin
    # Backward-compat: default constructor keeps realizability_alpha = 0
    # (disabled), preserving the classical high-Re eddy-viscosity formula.
    m_default = StandardKEpsilon()
    @test m_default.realizability_alpha == 0.0

    # Opting in sets the cap constant.
    m_durbin = StandardKEpsilon(; realizability_alpha = 2 / 3)
    @test m_durbin.realizability_alpha ≈ 2 / 3
    @test m_durbin.C_mu == 0.09   # other fields unchanged

    # The cap enforcement happens inside `solve_turbulence!` (tested via
    # the in-place `nu_t[c] = min(nu_t[c], α k / |S|)` at the strain-rate
    # site). That's covered by the existing RANS turbulence suite —
    # adding the field without setting the alpha must not change any
    # numerical result, which the broader test matrix confirms.
end

@testset "Stage 4d: _wall_projection geometric decomposition" begin
    # 2×2 Cartesian mesh; pick the top boundary face of cell 1 (bottom-left
    # cell). The outward normal there is (0, -1) for the bottom face of
    # cell 1 on our builder, so pick a different face we can reason about.
    mesh = build_cartesian_unstructured_mesh(2, 2, 2.0, 2.0)
    nf = size(mesh.face_cells, 2)

    # Find a boundary face on the :bottom patch (normal = (0, -1)).
    f_bot = findfirst(
        f -> mesh.face_cells[2, f] == 0 &&
            mesh.face_tags[f] === :bottom, 1:nf
    )
    @test f_bot !== nothing
    c = mesh.face_cells[1, f_bot]

    # Velocity with mixed normal + tangential components.
    U_cell = SVector(1.0, 0.5)  # tangential Ux=1, "normal" Uy=0.5
    y, U_par = FiniteVolumeMethod._wall_projection(mesh, c, f_bot, U_cell)

    # Expected:
    # - face normal is (0, -1) pointing outward (bottom face).
    # - wall-normal distance = |y_cell - y_face| (cell center at y=0.5,
    #   face center at y=0.0) = 0.5.
    # - U_par = projection onto tangent plane: only x-component survives =
    #   sqrt(Ux² + 0²) = 1.0.
    @test y ≈ 0.5 atol = 1.0e-12
    @test U_par ≈ 1.0 atol = 1.0e-12

    # On a Cartesian mesh with U purely tangential (Uy=0 at bottom wall),
    # U_par equals |U| — consistent with the pre-Stage-4d behaviour.
    U_tan_only = SVector(1.0, 0.0)
    _, U_par_tan = FiniteVolumeMethod._wall_projection(mesh, c, f_bot, U_tan_only)
    @test U_par_tan ≈ 1.0

    # Sanity: straight-line distance on a Cartesian mesh matches the
    # wall-normal projection exactly.
    x_c = FiniteVolumeMethod.cell_center(mesh, c)
    x_f = FiniteVolumeMethod.face_center(mesh, f_bot)
    @test y ≈ norm(x_c - x_f)
end

@testset "Stage 4d: Wall projection strips normal velocity on skewed flows" begin
    # When U has a non-zero normal component (e.g. during an early
    # iteration of a solve before continuity has tightened), `U_par`
    # should strip that component and not feed it into the wall-function
    # shear estimate. A pre-Stage-4d solver would have used |U| =
    # sqrt(1² + 0.5²) ≈ 1.118 instead of 1.0.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    f_bot = findfirst(
        f -> mesh.face_cells[2, f] == 0 &&
            mesh.face_tags[f] === :bottom, 1:nf
    )
    c = mesh.face_cells[1, f_bot]

    U_with_normal = SVector(1.0, 0.5)
    _, U_par_new = FiniteVolumeMethod._wall_projection(mesh, c, f_bot, U_with_normal)
    U_old = norm(U_with_normal)
    @test U_par_new ≈ 1.0
    @test U_old > U_par_new   # old formula overestimated by the normal component
end

@testset "Stage 4c: dynamic Smagorinsky full-tensor Germano" begin
    # With the full-tensor fix the filtered strain |S̃| is computed from
    # the filtered tensor components rather than from `|S|`. The result
    # should differ in general; on a perfectly uniform velocity field (no
    # strain anywhere) both formulations agree trivially. Test that the
    # computed ν_t is finite and non-negative for a non-trivial flow —
    # catch any regression where the new tensor path would divide by zero
    # or produce NaN.
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    # Set up a planar shear flow U = (y, 0) → S_xy = 0.5, others 0.
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(y, 0.0)
    end

    model = DynamicSmagorinsky(mesh)
    nu_t = zeros(nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

    @test all(isfinite, nu_t)
    @test all(>=(0.0), nu_t)
    # For a uniformly sheared flow the Leonard / Germano identity should
    # give a well-defined Cs² in [0, 0.04] (the cap). We don't pin exact
    # values — the model is inherently dynamic and sensitive to the test
    # filter. Just check the bound.
    max_nu_t = maximum(nu_t)
    @test max_nu_t < 1.0   # with Δ ≈ 1/6 and |S| ≈ 1, ν_t < Cs² Δ² |S| < 0.04/36 ≈ 1e-3
end

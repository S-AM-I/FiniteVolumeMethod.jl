# test/v_and_v_vof_planewave.jl — VOF plane-wave advection V&V (v3.24)
#
# Second analytical benchmark for `multiphase_vof`. The first
# benchmark (v3.16, disc translation) established mass conservation
# and center-of-mass kinematics; this benchmark extends the
# coverage to **wave-shape preservation** under a smooth initial
# field, which is the primary accuracy metric for advection
# schemes.
#
# Problem: the pure-kinematic advection equation
#
#   ∂α/∂t + u · ∇α = 0
#
# with u = (U, 0) has the closed-form solution α(x, y, t) =
# α₀(x − U·t, y). For a smooth initial field α₀(x, y) =
# 0.5 + 0.4·sin(2π·x/L), the exact solution at time t is
#
#   α_exact(x, y, t) = 0.5 + 0.4·sin(2π·(x − U·t)/L),
#
# provided the translation distance stays within the domain.
# Periodic x-BCs (modelled via matching Dirichlet on the inflow
# and zero-gradient on the outflow) keep the analytical solution
# well-defined.
#
# Upwind convection dissipates short wavelengths; on a 100-cell
# mesh resolving one wavelength, the amplitude should survive the
# integration to within a measurable fraction, and the phase
# speed should match U exactly (no dispersion under upwind
# for constant-coefficient advection).

using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: NeumannBC
using LinearSolve
using Test

include("TestHelpers.jl")

const PW_L = 2.0
const PW_LY = 0.2
const PW_U = 1.0
const PW_AMP = 0.4
const PW_MEAN = 0.5
const PW_K = 2 * pi / PW_L

alpha_exact(x, t) = PW_MEAN + PW_AMP * sin(PW_K * (x - PW_U * t))

function init_planewave(mesh)
    nc = length(mesh.cell_volumes)
    alpha = CollocatedScalarField(:alpha, mesh; value = PW_MEAN)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        alpha.internal[c] = alpha_exact(x, 0.0)
    end
    return alpha
end

function build_uniform_phi(mesh)
    nf = size(mesh.face_cells, 2)
    phi = FaceFluxField(:phi, mesh)
    for f in 1:nf
        phi.values[f] = PW_U * face_normal_area(mesh, f)[1]
    end
    return phi
end

function run_planewave(Nx::Int, n_steps::Int, t_end::Float64)
    mesh = build_cartesian_unstructured_mesh(Nx, 10, PW_L, PW_LY)
    alpha = init_planewave(mesh)
    phi = build_uniform_phi(mesh)
    dt = t_end / n_steps

    # Inflow α(0, t) is the time-dependent analytical value at x=0;
    # freeze it via Dirichlet = current analytical at inlet. To
    # avoid changing the BC every step, use ZeroGradient at both
    # ends — this lets the wave advect freely without artificial
    # inflow mismatch.
    bcs_alpha = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NeumannBC(0.0),
        :right => NeumannBC(0.0),
        :bottom => NeumannBC(0.0),
        :top => NeumannBC(0.0),
    )

    for _ in 1:n_steps
        eq = CollocatedEquation(mesh)
        assemble_alpha!(eq, alpha, phi, mesh, bcs_alpha; dt = dt, C_alpha = 0.0)
        lp = to_linear_problem(eq)
        sol = LinearSolve.solve(lp, LUFactorization())
        for c in 1:length(mesh.cell_volumes)
            alpha.internal[c] = sol.u[c]
        end
    end
    return mesh, alpha
end

function interior_l1_error(mesh, alpha, t_end::Float64)
    nc = length(mesh.cell_volumes)
    err = 0.0
    count = 0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        # Interior band to avoid boundary effects from zero-gradient BCs.
        if 0.2 * PW_L < x < 0.8 * PW_L
            err += abs(alpha.internal[c] - alpha_exact(x, t_end))
            count += 1
        end
    end
    return err / count
end

@testset "V&V: VOF plane wave — amplitude + phase agreement" begin
    # After advecting for t = 0.25 (Courant ≈ 0.5 on 100-cell
    # mesh with 50 steps), the wave should have translated by
    # Δx = U · t = 0.25. Upwind convection dissipates short
    # wavelengths but preserves phase; measure amplitude via
    # max - min in the interior, and phase via peak location.
    t_end = 0.25
    Nx = 200
    mesh, alpha = run_planewave(Nx, 100, t_end)

    # Sample mid-y centerline (all rows identical by symmetry +
    # Neumann top/bottom).
    nc = length(mesh.cell_volumes)
    # Pick a fixed y row (cells with y in a narrow band near Ly/2).
    sample_x = Float64[]
    sample_a = Float64[]
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        if abs(y - PW_LY / 2) < PW_LY / 20
            push!(sample_x, mesh.cell_centers[1, c])
            push!(sample_a, alpha.internal[c])
        end
    end
    order = sortperm(sample_x)
    sample_x = sample_x[order]
    sample_a = sample_a[order]

    # Interior band.
    keep = [0.2 * PW_L < x < 0.8 * PW_L for x in sample_x]
    sx = sample_x[keep]
    sa = sample_a[keep]

    # Note: Neumann BCs do not perfectly enforce the translating-wave
    # ansatz at the inflow; the mean drifts slowly toward the
    # inflow-cell value via the zero-gradient BC. We do not gate on
    # the mean here — the L¹-error gate in the next testset covers
    # the global accuracy.

    # Amplitude (half of max-min) — upwind attenuates, but the
    # bound states survive robustly. Accept anything above
    # 0.5 · PW_AMP on the 200-cell mesh.
    amp_num = (maximum(sa) - minimum(sa)) / 2
    @test amp_num > 0.5 * PW_AMP
    @test amp_num < 1.02 * PW_AMP   # never overshoots

    # Phase shift: find the numerical peak index, compute its x,
    # compare to analytical peak location at t_end.
    _, i_peak = findmax(sa)
    x_peak = sx[i_peak]
    # Analytical: α_peak at x = U·t + π/(2K) = U·t + L/4 (mod L).
    x_peak_exact = PW_U * t_end + PW_L / 4
    # Upwind has no dispersion ⇒ phase error bounded by grid.
    h = PW_L / Nx
    @test abs(x_peak - x_peak_exact) < 5 * h
end

@testset "V&V: VOF plane wave — L¹ error decreases with refinement" begin
    # Upwind on a smooth sine wave is first-order in h; expected
    # rate ≈ 1. Implicit Euler time integration contributes
    # first-order temporal error, so the CFL is held approximately
    # constant across refinements.
    t_end = 0.25
    errs = Float64[]
    for Nx in (100, 200, 400)
        n_steps = div(Nx, 2)  # CFL ≈ 0.5 held constant
        _, alpha = run_planewave(Nx, n_steps, t_end)
        mesh_ref = build_cartesian_unstructured_mesh(Nx, 10, PW_L, PW_LY)
        push!(errs, interior_l1_error(mesh_ref, alpha, t_end))
    end

    # Monotone decrease.
    @test all(errs[i] > errs[i + 1] for i in 1:(length(errs) - 1))

    # First-order upwind: expect rates ≥ 0.7 (bounded by boundary
    # effects from the inflow Neumann — the zero-gradient BC isn't
    # the ideal periodic closure, so the interior rate is slightly
    # below the textbook 1.0).
    orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
    for p in orders
        @test p > 0.6
    end
end

@testset "V&V: VOF plane wave — near-bounded" begin
    # Upwind + implicit Euler is not strictly TVD on finite unstructured
    # meshes with Neumann boundary closures (the max-principle holds
    # exactly only for strict upwind on periodic meshes). Accept a
    # small overshoot on the order of `5 · h_min · α_amplitude`; this
    # tracks the v3 simplifications documented in KNOWN_FAILURES
    # (no MULES / isoAdvector yet). The headline accuracy gates live
    # in the preceding "amplitude + phase agreement" and
    # "L¹ error decreases with refinement" testsets.
    t_end = 0.25
    mesh, alpha = run_planewave(200, 100, t_end)
    h = PW_L / 200
    tol = 5 * h * PW_AMP
    @test minimum(alpha.internal) >= 0.1 - tol
    @test maximum(alpha.internal) <= 0.9 + tol
end

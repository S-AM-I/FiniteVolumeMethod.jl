# test/v_and_v_kepsilon_dhit.jl — k-ε decaying homogeneous turbulence V&V (v3.18)
#
# Verifies `solve_turbulence!` for `StandardKEpsilon` against the
# closed-form ODE solution of decaying homogeneous isotropic
# turbulence (DHIT). With uniform fields, no mean-shear production,
# no flux (phi = 0), and Neumann BCs, the k-ε transport system
# degenerates to
#
#   dk/dt = -ε,            dε/dt = -C_ε2 · ε²/k,
#
# whose closed-form solution (with τ = ε_0 · t / k_0) is
#
#   k(t) = k_0 (1 + (C_ε2 − 1) τ)^(-1/(C_ε2 − 1))
#   ε(t) = ε_0 (1 + (C_ε2 − 1) τ)^(-C_ε2/(C_ε2 − 1))
#
# Standard values: C_ε2 = 1.92, giving k ~ τ^(-1.087) asymptotically.
# Evidence for promoting `turbulence_rans` from `experimental`/
# `smoke_tested` to `provisional`/`convergence_verified` on the
# source-term side.

using FiniteVolumeMethod
using LinearSolve
using StaticArrays
using Test

include("TestHelpers.jl")

const DHIT_K0 = 1.0
const DHIT_EPS0 = 1.0
const DHIT_C2 = 1.92

function k_exact(t::Float64)
    tau = DHIT_EPS0 * t / DHIT_K0
    return DHIT_K0 * (1 + (DHIT_C2 - 1) * tau)^(-1 / (DHIT_C2 - 1))
end

function eps_exact(t::Float64)
    tau = DHIT_EPS0 * t / DHIT_K0
    return DHIT_EPS0 * (1 + (DHIT_C2 - 1) * tau)^(-DHIT_C2 / (DHIT_C2 - 1))
end

function run_dhit(n_steps::Int, dt::Float64)
    # 4x4 mesh — values are uniform across cells by construction, so
    # mesh topology is immaterial.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    model = StandardKEpsilon()

    # Uniform zero velocity ⇒ no mean-shear, S = 0, P_k = 0.
    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))

    # Zero face flux ⇒ no convection.
    phi = FaceFluxField(:phi, mesh; value = 0.0)

    # Uniform initial k and ε.
    turb_state = RANSTurbulenceState(
        model, mesh; k = DHIT_K0, epsilon = DHIT_EPS0,
    )

    # Initial nu_t = C_mu · k²/ε.
    FiniteVolumeMethod.turbulent_viscosity!(turb_state.nu_t, model, turb_state, mesh)

    # Neumann(0) on all walls for both fields.
    bc_neumann = ParabolicNeumann(0.0)
    bcs_turb = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
        :k => Dict(
            :left => bc_neumann, :right => bc_neumann,
            :bottom => bc_neumann, :top => bc_neumann,
        ),
        :epsilon => Dict(
            :left => bc_neumann, :right => bc_neumann,
            :bottom => bc_neumann, :top => bc_neumann,
        ),
    )

    nu = 1.0e-6
    k_hist = Float64[DHIT_K0]
    e_hist = Float64[DHIT_EPS0]

    t = 0.0
    for _ in 1:n_steps
        FiniteVolumeMethod.solve_turbulence!(
            turb_state, model, U, phi, nu, mesh, bcs_turb;
            dt = dt, linear_solver = LUFactorization(),
        )
        # All cells are uniform by symmetry; sample cell 1.
        push!(k_hist, turb_state.fields[:k].internal[1])
        push!(e_hist, turb_state.fields[:epsilon].internal[1])
        t += dt
    end

    return (
        turb_state = turb_state,
        k_hist = k_hist, e_hist = e_hist, t_end = t,
    )
end

@testset "V&V: DHIT — realizability (k, ε, ν_t ≥ 0 and monotone)" begin
    res = run_dhit(200, 0.005)

    # Fields remain non-negative throughout (realizability).
    @test all(>=(0.0), res.k_hist)
    @test all(>=(0.0), res.e_hist)
    @test all(>=(0.0), res.turb_state.nu_t)

    # Strict monotone decay — no oscillation, no overshoot.
    @test all(diff(res.k_hist) .<= 1.0e-14)
    @test all(diff(res.e_hist) .<= 1.0e-14)
end

@testset "V&V: DHIT — endpoint agreement with analytical" begin
    # 1000 small steps over t ∈ [0, 1.0] → strong convergence for
    # implicit Euler on this stiff-free ODE.
    res = run_dhit(1000, 0.001)
    k_num = res.k_hist[end]
    e_num = res.e_hist[end]

    k_an = k_exact(res.t_end)
    e_an = eps_exact(res.t_end)

    # k decays at rate 1.087; implicit Euler over-damps slightly.
    # At dt/t = 1e-3, expect ≤ 1 % error on k.
    @test abs(k_num - k_an) / k_an < 1.0e-2
    # ε decays at rate 2.087 (faster); expect ≤ 1 % error as well.
    @test abs(e_num - e_an) / e_an < 1.0e-2
end

@testset "V&V: DHIT — first-order convergence in Δt" begin
    # At coarser dt, error grows linearly. Expected rate ≈ 1.0.
    errors = Float64[]
    for (n_steps, dt) in ((100, 0.01), (200, 0.005), (400, 0.0025))
        res = run_dhit(n_steps, dt)
        push!(errors, abs(res.k_hist[end] - k_exact(res.t_end)))
    end

    # Monotone decrease.
    @test all(errors[i] > errors[i + 1] for i in 1:(length(errors) - 1))

    orders = [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]
    for p in orders
        @test 0.8 < p < 1.3
    end
end

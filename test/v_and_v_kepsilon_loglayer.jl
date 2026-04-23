# test/v_and_v_kepsilon_loglayer.jl — k-ε log-layer equilibrium V&V (v3.23)
#
# In the inertial sublayer of a wall-bounded turbulent flow the
# standard k-ε closure admits an exact local equilibrium: the
# production of turbulent kinetic energy balances its dissipation
# (P_k = ε) under the log-law scaling
#
#   U(y) = (u_τ / κ) · log(y / y_0),
#   k(y) = u_τ² / √C_μ          (uniform across the sublayer),
#   ε(y) = u_τ³ / (κ · y)       (decays as 1/y).
#
# From these:
#
#   ∂U/∂y = u_τ / (κ · y),   |S|² = 2 S_ij S_ij = (∂U/∂y)² = (u_τ/κy)²,
#   ν_t   = C_μ · k² / ε = κ · y · u_τ,
#   P_k   = ν_t · |S|² = u_τ³ / (κ·y) = ε.
#
# Hence P_k / ε ≡ 1 exactly at every y in the sublayer. This is a
# closed-form invariant of the standard k-ε model that can be
# verified cell-by-cell on a prescribed velocity/k/ε field without
# running a transient simulation. Second benchmark for
# `turbulence_rans` (first: DHIT ODE, v3.18), progressing toward
# future stable promotion.

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

const U_TAU = 0.05
const KAPPA = 0.41
const C_MU = 0.09

@testset "V&V: k-ε log-layer — P_k / ε ≡ 1 (equilibrium)" begin
    # Build the mesh on [0, 1] × [0, Ly]. Interpret the physical
    # wall coordinate as y_phys = y_mesh + y_offset so the
    # log-profile is evaluated at y_phys > 0 (avoids the
    # ε = u_τ³/(κy) singularity at y = 0).
    y_offset = 0.05
    Nx = 8
    Ny = 40
    Lx = 1.0
    Ly = 0.5

    mesh = build_cartesian_unstructured_mesh(Nx, Ny, Lx, Ly)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y_phys = mesh.cell_centers[2, c] + y_offset
        # Log-law velocity. ∂U/∂y = u_τ / (κ · y_phys).
        u = (U_TAU / KAPPA) * log(y_phys / y_offset)
        U.internal[c] = SVector(u, 0.0)
    end

    # Prescribed k and ε matching the inertial-sublayer scaling.
    k_val = fill(U_TAU^2 / sqrt(C_MU), nc)
    eps_val = [U_TAU^3 / (KAPPA * (mesh.cell_centers[2, c] + y_offset)) for c in 1:nc]

    # ν_t from the standard k-ε formula.
    nu_t = [C_MU * k_val[c]^2 / eps_val[c] for c in 1:nc]

    # Strain-rate magnitude |S| from the FVM gradient of U.
    S_mag = compute_strain_rate(U, mesh)

    # Production P_k = ν_t · |S|².
    P_k = [nu_t[c] * S_mag[c]^2 for c in 1:nc]

    # Interior band mask: away from boundaries (gradient stencil
    # truncation) and away from the log-profile curvature region
    # near y_phys = y_offset.
    mask = falses(nc)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        x = mesh.cell_centers[1, c]
        if 0.3 * Ly < y < 0.7 * Ly && 0.2 * Lx < x < 0.8 * Lx
            mask[c] = true
        end
    end
    @test count(mask) > 10

    # P_k / ε ratio should be close to 1 in the log layer. The
    # discrete gradient carries O(h² · d²U/dy²) truncation error,
    # and d²U/dy² = −u_τ/(κ y²) is finite but non-zero, so a 15 %
    # tolerance is consistent with cell-centered FVM truncation on
    # a logarithmic field.
    n_checked = 0
    for c in 1:nc
        if mask[c]
            ratio = P_k[c] / eps_val[c]
            @test 0.85 < ratio < 1.15
            n_checked += 1
        end
    end
    @test n_checked > 10
end

@testset "V&V: k-ε log-layer — ν_t = κ · y · u_τ invariant" begin
    # Independent algebraic invariant (no strain-rate dependency):
    #   ν_t = C_μ · k² / ε
    #       = C_μ · (u_τ² / √C_μ)² / (u_τ³ / (κ y))
    #       = C_μ / C_μ · u_τ⁴ · κ y / u_τ³
    #       = κ y u_τ.
    #
    # Verify numerically at a dense y sweep.
    for y in range(0.01, 0.5; length = 20)
        k_local = U_TAU^2 / sqrt(C_MU)
        eps_local = U_TAU^3 / (KAPPA * y)
        nu_t_local = C_MU * k_local^2 / eps_local

        expected = KAPPA * y * U_TAU
        @test isapprox(nu_t_local, expected; rtol = 1.0e-12)
    end
end

@testset "V&V: k-ε log-layer — realizability consistency" begin
    # For the prescribed log-layer state, Durbin realizability cap
    #   ν_t ≤ α · k / |S|
    # yields a *soft* bound:
    #   α · k / |S| = α · (u_τ² / √C_μ) / (u_τ / κ y) = α · κ y u_τ / √C_μ
    # For C_μ = 0.09 ⇒ √C_μ ≈ 0.3, so the cap is (α/0.3)·κ·y·u_τ ≈
    # 3.33 α times the unconstrained ν_t. With α = 0.6 (Durbin
    # 1996), the cap is 2·ν_t — inactive by design in the log
    # region, where the closure is already in equilibrium.
    #
    # This test verifies that the numerical cap never triggers
    # on the prescribed equilibrium state.
    alpha = 0.6
    for y in range(0.01, 0.5; length = 20)
        k_local = U_TAU^2 / sqrt(C_MU)
        eps_local = U_TAU^3 / (KAPPA * y)
        nu_t_local = C_MU * k_local^2 / eps_local
        S_local = U_TAU / (KAPPA * y)

        cap = alpha * k_local / S_local
        @test nu_t_local < cap   # realizability is inactive in equilibrium
    end
end

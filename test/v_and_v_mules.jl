# test/v_and_v_mules.jl — MULES flux limiter invariants V&V (v3.91)

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

const _mules_limit_flux! = FiniteVolumeMethod.mules_limit_flux!

@testset "V&V: MULES — pure upwind ⇒ limited = upwind (F_ad = 0)" begin
    # When phi_high equals phi_upwind there is no anti-diffusive
    # correction to apply — the limiter must reduce to the upwind flux
    # identically on every face, regardless of dt or α.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.5)
    phi_up = FaceFluxField(:phi_up, mesh; value = 0.0)
    phi_hi = FaceFluxField(:phi_hi, mesh; value = 0.0)
    for f in 1:nf
        phi_up.values[f] = 0.25 * (f % 7 - 3)
        phi_hi.values[f] = phi_up.values[f]   # no anti-diffusion
    end
    limited = FaceFluxField(:lim, mesh; value = 0.0)
    _mules_limit_flux!(limited, alpha, phi_up, phi_hi, mesh, 0.01)
    for f in 1:nf
        @test limited.values[f] == phi_up.values[f]
    end
end

@testset "V&V: MULES — already-bounded ⇒ full anti-diffusion (λ = 1)" begin
    # With alpha = 0.5 everywhere, zero upwind flux, and small uniform
    # anti-diffusive flux F_ad, the budget q_plus and q_minus are both
    # large (half of the full range available) so r_plus and r_minus
    # saturate to 1 ⇒ the limited flux equals the full high-order flux.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.5)
    phi_up = FaceFluxField(:phi_up, mesh; value = 0.0)
    phi_hi = FaceFluxField(:phi_hi, mesh; value = 0.0)
    for f in 1:nf
        phi_hi.values[f] = 1.0e-6   # tiny anti-diffusion
    end
    limited = FaceFluxField(:lim, mesh; value = 0.0)
    _mules_limit_flux!(limited, alpha, phi_up, phi_hi, mesh, 1.0e-6)
    # Limited should equal phi_hi everywhere (λ = 1).
    for f in 1:nf
        @test isapprox(limited.values[f], phi_hi.values[f]; rtol = 1.0e-12)
    end
end

@testset "V&V: MULES — zero dt ⇒ λ constraint relaxed" begin
    # dt → 0 drives q_plus, q_minus → ∞ — but the function guards against
    # division by zero internally. Just check the call succeeds and the
    # output is finite.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.5)
    phi_up = FaceFluxField(:phi_up, mesh; value = 0.1)
    phi_hi = FaceFluxField(:phi_hi, mesh; value = 0.2)
    limited = FaceFluxField(:lim, mesh; value = 0.0)
    _mules_limit_flux!(limited, alpha, phi_up, phi_hi, mesh, 1.0e-10)
    for f in 1:nf
        @test isfinite(limited.values[f])
    end
end

@testset "V&V: MULES — limited flux bounded between phi_upwind and phi_high" begin
    # λ ∈ [0, 1] ⇒ limited = phi_up + λ · (phi_hi - phi_up) lies on the
    # closed segment between phi_up and phi_hi on every face.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.3)
    phi_up = FaceFluxField(:phi_up, mesh; value = 0.0)
    phi_hi = FaceFluxField(:phi_hi, mesh; value = 0.0)
    for f in 1:nf
        phi_up.values[f] = 0.1 * sin(f)
        phi_hi.values[f] = phi_up.values[f] + 0.5 * cos(f)
    end
    limited = FaceFluxField(:lim, mesh; value = 0.0)
    _mules_limit_flux!(limited, alpha, phi_up, phi_hi, mesh, 1.0e-3)
    for f in 1:nf
        lo = min(phi_up.values[f], phi_hi.values[f])
        hi = max(phi_up.values[f], phi_hi.values[f])
        @test lo - 1.0e-12 <= limited.values[f] <= hi + 1.0e-12
    end
end

@testset "V&V: MULES — boundedness of α under limited flux" begin
    # Advance α by one explicit Euler step using the MULES-limited flux
    # starting from α = 0.5. By construction the result must stay in
    # [0, 1] on every cell. This is the MULES guarantee.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.5)
    phi_up = FaceFluxField(:phi_up, mesh; value = 0.0)
    phi_hi = FaceFluxField(:phi_hi, mesh; value = 0.0)
    # Force an aggressive anti-diffusive flux field.
    for f in 1:nf
        phi_hi.values[f] = 0.9 * sin(f)
    end
    limited = FaceFluxField(:lim, mesh; value = 0.0)
    dt = 1.0e-2
    _mules_limit_flux!(limited, alpha, phi_up, phi_hi, mesh, dt)
    # Apply one explicit Euler step with the limited flux: α_P -= F·dt/V.
    alpha_next = copy(alpha.internal)
    for f in 1:nf
        F = limited.values[f] * dt
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        alpha_next[P] -= F / mesh.cell_volumes[P]
        if N != 0
            alpha_next[N] += F / mesh.cell_volumes[N]
        end
    end
    for c in 1:nc
        @test -1.0e-10 <= alpha_next[c] <= 1.0 + 1.0e-10
    end
end

@testset "V&V: MULES — output shape" begin
    # limited_flux shares size with phi_upwind and phi_high.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.5)
    phi_up = FaceFluxField(:phi_up, mesh; value = 0.01)
    phi_hi = FaceFluxField(:phi_hi, mesh; value = 0.02)
    limited = FaceFluxField(:lim, mesh; value = 0.0)
    _mules_limit_flux!(limited, alpha, phi_up, phi_hi, mesh, 0.01)
    @test length(limited.values) == nf
    @test length(limited.values) == length(phi_up.values)
end

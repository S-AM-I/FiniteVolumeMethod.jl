# test/v_and_v_six_dof.jl — Newton-Euler 6-DOF rigid-body V&V
#
# Invariants checked:
# 1. Zero force + zero torque ⇒ state unchanged to rtol 1e-14
# 2. Constant force ⇒ r(t) = r0 + v0·t + 0.5·a·t² rtol 1e-10
# 3. Angular momentum preservation under zero torque
# 4. Quaternion norm stays ≈ 1 after every step
# 5. Symmetric-inertia body rotates at constant ω under no torque

using LinearAlgebra
using StaticArrays
using Test

# Load the six-DOF source module into the test scope. This keeps the test
# runnable stand-alone even before the main-thread wiring of the new files
# into src/FiniteVolumeMethod.jl lands. The standalone form avoids depending
# on FiniteVolumeMethod's precompile graph while wave-2 parallel work is in
# flight.
include(joinpath(@__DIR__, "..", "src", "collocated", "dynamic_mesh", "six_dof.jl"))

@testset "V&V 6-DOF: zero force and zero torque ⇒ state frozen" begin
    I3 = SMatrix{3, 3, Float64, 9}(Matrix(1.0I, 3, 3))
    body = RigidBody6DOF(
        2.0, I3;
        position = SVector(1.0, 2.0, 3.0),
        velocity = SVector(0.0, 0.0, 0.0),
        orientation = SVector(1.0, 0.0, 0.0, 0.0),
        angular_velocity = SVector(0.0, 0.0, 0.0),
    )

    r0 = body.position
    v0 = body.velocity
    q0 = body.orientation
    w0 = body.angular_velocity

    for _ in 1:100
        advance_six_dof!(body, SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0), 0.01)
    end

    @test body.position ≈ r0 rtol = 1.0e-14
    @test body.velocity ≈ v0 rtol = 1.0e-14
    @test body.orientation ≈ q0 rtol = 1.0e-14
    @test body.angular_velocity ≈ w0 rtol = 1.0e-14
end

@testset "V&V 6-DOF: constant force matches closed-form kinematics" begin
    I3 = SMatrix{3, 3, Float64, 9}(Matrix(1.0I, 3, 3))
    m = 2.0
    v0 = SVector(0.5, -0.2, 0.1)
    r0 = SVector(0.0, 0.0, 0.0)
    body = RigidBody6DOF(m, I3; position = r0, velocity = v0)

    F = SVector(1.0, -0.5, 0.25)
    a = F / m
    dt = 1.0e-6
    N = 10_000

    for _ in 1:N
        advance_six_dof!(body, F, SVector(0.0, 0.0, 0.0), dt)
    end

    # Explicit Euler r_{n+1} = r_n + v_n dt and v_{n+1} = v_n + a dt
    # give r_N = r0 + v0·T + 0.5·a·T²·(1 - 1/N) after N steps, T = N·dt.
    # For small dt this matches the continuous closed form within O(dt).
    T = N * dt
    r_analytic = r0 + v0 * T + 0.5 * a * T^2
    v_analytic = v0 + a * T

    # Explicit Euler has O(dt) global error ~ 0.5 * |a| * dt * T,
    # which for these parameters is ~1e-6 ≪ 1 so rtol 1e-4 is comfortable.
    @test body.position ≈ r_analytic rtol = 1.0e-4
    @test body.velocity ≈ v_analytic rtol = 1.0e-10
end

@testset "V&V 6-DOF: torque-free spherical-inertia body spins at constant ω" begin
    # With isotropic inertia, Euler's equation ω̇ = I⁻¹(τ − ω×Iω) simplifies
    # to ω̇ = I⁻¹·τ − ω × ω = I⁻¹·τ. For τ = 0, ω is exactly preserved.
    I3 = SMatrix{3, 3, Float64, 9}(Matrix(2.5I, 3, 3))
    body = RigidBody6DOF(
        1.0, I3;
        angular_velocity = SVector(0.3, 0.1, -0.2),
    )
    ω0 = body.angular_velocity

    for _ in 1:5_000
        advance_six_dof!(body, SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0), 1.0e-3)
    end

    @test body.angular_velocity ≈ ω0 rtol = 1.0e-10
    # Angular momentum preservation (exact for isotropic inertia)
    L0 = I3 * ω0
    @test angular_momentum(body) ≈ L0 rtol = 1.0e-10
end

@testset "V&V 6-DOF: quaternion stays normalized every step" begin
    I3 = SMatrix{3, 3, Float64, 9}(diagm([1.0, 2.0, 3.0]))
    body = RigidBody6DOF(
        1.0, I3;
        angular_velocity = SVector(0.4, 0.7, 0.2),
    )

    for k in 1:2_000
        advance_six_dof!(body, SVector(0.0, 0.0, 0.0), SVector(0.1, -0.05, 0.02), 1.0e-3)
        @test quaternion_norm(body) ≈ 1.0 atol = 1.0e-12
    end
end

@testset "V&V 6-DOF: symmetric (spherical) inertia has equal principal moments" begin
    # Confirm the constructor preserves isotropy and the inverse is also isotropic.
    I_iso = 1.7
    I3 = SMatrix{3, 3, Float64, 9}(Matrix(I_iso * I, 3, 3))
    body = RigidBody6DOF(1.0, I3)
    @test body.inertia[1, 1] == I_iso
    @test body.inertia[2, 2] == I_iso
    @test body.inertia[3, 3] == I_iso
    @test body.inertia_inv[1, 1] ≈ 1 / I_iso rtol = 1.0e-14
    @test body.inertia_inv[2, 2] ≈ 1 / I_iso rtol = 1.0e-14
    @test body.inertia_inv[3, 3] ≈ 1 / I_iso rtol = 1.0e-14

    # And evolving under zero torque preserves ω magnitude.
    body.angular_velocity = SVector(1.0, 0.0, 0.0)
    for _ in 1:1_000
        advance_six_dof!(body, SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0), 1.0e-3)
    end
    @test norm(body.angular_velocity) ≈ 1.0 rtol = 1.0e-10
end

@testset "V&V 6-DOF: kinetic_energy accounts for translation and rotation" begin
    I3 = SMatrix{3, 3, Float64, 9}(diagm([2.0, 3.0, 4.0]))
    body = RigidBody6DOF(
        5.0, I3;
        velocity = SVector(0.0, 0.0, 0.0),
        angular_velocity = SVector(0.0, 0.0, 0.0),
    )
    @test kinetic_energy(body) ≈ 0.0 atol = 1.0e-14

    body.velocity = SVector(1.0, 2.0, -1.0)  # ‖v‖² = 6
    body.angular_velocity = SVector(1.0, 0.0, 0.0)  # ωᵀIω = 2
    expected = 0.5 * 5.0 * 6 + 0.5 * 2.0
    @test kinetic_energy(body) ≈ expected rtol = 1.0e-14
end

@testset "V&V 6-DOF: constructor rejects invalid inputs" begin
    I3 = SMatrix{3, 3, Float64, 9}(Matrix(1.0I, 3, 3))
    @test_throws ErrorException RigidBody6DOF(-1.0, I3)
    bad_inertia = [1.0 0.0; 0.0 1.0]  # 2×2 instead of 3×3
    @test_throws ErrorException RigidBody6DOF(1.0, bad_inertia)
end

# ============================================================
# Special Relativistic MHD Equations
# ============================================================
#
# 8-variable system for special relativistic magnetohydrodynamics.
#
# Primitive:  W = [ρ, vx, vy, vz, P, Bx, By, Bz]
# Conserved:  U = [D, Sx, Sy, Sz, τ, Bx, By, Bz]
#
#   D = ρW                          (Lorentz-contracted density)
#   S_j = (ρhW² + b²)v_j − b⁰b_j  (relativistic momentum)
#   τ = ρhW² − P_tot − (b⁰)² − D  (energy minus rest mass)
#
# Lorentz factor: W = 1/√(1 − v²)
# Specific enthalpy: h = 1 + ε + P/ρ
# Magnetic 4-vector: b⁰ = W(v·B), b² = B²/W² + (v·B)²
# Total pressure: P_tot = P + ½b²

"""
    SRMHDEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}

The special relativistic magnetohydrodynamics equations in `Dim` spatial dimensions.

## Variables (8 components in all dimensions)
- Primitive: `W = [ρ, vx, vy, vz, P, Bx, By, Bz]`
- Conserved: `U = [D, Sx, Sy, Sz, τ, Bx, By, Bz]`

The conserved↔primitive conversion requires iterative root-finding (Con2Prim).

# Fields
- `eos::EOS`: Equation of state.
- `con2prim_tol::Float64`: Tolerance for con2prim convergence (default 1e-12).
- `con2prim_maxiter::Int`: Maximum con2prim iterations (default 50).
"""
struct SRMHDEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}
    eos::EOS
    con2prim_tol::Float64
    con2prim_maxiter::Int
end

function SRMHDEquations{Dim}(eos::EOS; con2prim_tol = 1.0e-12, con2prim_maxiter = 50) where {Dim, EOS <: AbstractEOS}
    return SRMHDEquations{Dim, EOS}(eos, con2prim_tol, con2prim_maxiter)
end

nvariables(::SRMHDEquations) = 8

# ============================================================
# Lorentz factor and magnetic 4-vector helpers
# ============================================================

"""
    lorentz_factor(vx, vy, vz) -> W

Compute the Lorentz factor W = 1/√(1 − v²).
"""
@inline function lorentz_factor(vx, vy, vz)
    v_sq = vx^2 + vy^2 + vz^2
    v_sq = min(v_sq, 1 - 1.0e-10)  # cap to prevent superluminal
    return 1 / sqrt(1 - v_sq)
end

"""
    srmhd_b_quantities(vx, vy, vz, Bx, By, Bz, W) -> (b0, b_sq)

Compute the magnetic 4-vector temporal component b⁰ = W(v·B)
and the magnetic pressure term b² = B²/W² + (v·B)².
"""
@inline function srmhd_b_quantities(vx, vy, vz, Bx, By, Bz, W)
    vdotB = vx * Bx + vy * By + vz * Bz
    B_sq = Bx^2 + By^2 + Bz^2
    b0 = W * vdotB
    b_sq = B_sq / W^2 + vdotB^2
    return b0, b_sq
end

# ============================================================
# Conserved ↔ Primitive Conversion
# ============================================================

"""
    conserved_to_primitive(law::SRMHDEquations, u::SVector{8}) -> SVector{8}

Convert SRMHD conserved `[D, Sx, Sy, Sz, τ, Bx, By, Bz]` to
primitive `[ρ, vx, vy, vz, P, Bx, By, Bz]` via iterative con2prim.
"""
@inline function conserved_to_primitive(law::SRMHDEquations, u::SVector{8})
    w, _ = srmhd_con2prim(law.eos, u, law.con2prim_tol, law.con2prim_maxiter)
    return w
end

"""
    primitive_to_conserved(law::SRMHDEquations, w::SVector{8}) -> SVector{8}

Convert SRMHD primitive `[ρ, vx, vy, vz, P, Bx, By, Bz]` to
conserved `[D, Sx, Sy, Sz, τ, Bx, By, Bz]`.
"""
@inline function primitive_to_conserved(law::SRMHDEquations, w::SVector{8})
    ρ, vx, vy, vz, P, Bx, By, Bz = w
    γ_eos = law.eos.gamma

    W = lorentz_factor(vx, vy, vz)
    W_sq = W^2

    ε = P / ((γ_eos - 1) * ρ)
    h = 1 + ε + P / ρ

    B_sq = Bx^2 + By^2 + Bz^2
    vdotB = vx * Bx + vy * By + vz * Bz
    _, b_sq = srmhd_b_quantities(vx, vy, vz, Bx, By, Bz, W)

    rho_h_W2 = ρ * h * W_sq
    Ptot = P + 0.5 * b_sq

    D = ρ * W
    # S_j = (ρhW² + B²)v_j − (v·B)B_j
    Sx_c = (rho_h_W2 + B_sq) * vx - vdotB * Bx
    Sy_c = (rho_h_W2 + B_sq) * vy - vdotB * By
    Sz_c = (rho_h_W2 + B_sq) * vz - vdotB * Bz
    # τ = ρhW² + B² − P_tot − D  (since (ρh+b²)W² = ρhW² + B² + b0²)
    tau = rho_h_W2 + B_sq - Ptot - D

    return SVector(D, Sx_c, Sy_c, Sz_c, tau, Bx, By, Bz)
end

# ============================================================
# Physical Flux
# ============================================================

"""
    physical_flux(law::SRMHDEquations, w::SVector{8}, dir::Int) -> SVector{8}

Compute the SRMHD flux in direction `dir` (1=x, 2=y) from primitive variables.

Standard formulation: F^i = [D*v^i, (ρhW²+b²)v_j*v^i - b_j*b^i + P_tot*δ^i_j,
                              (τ+P_tot)*v^i - b⁰*b^i, B^j*v^i - B^i*v^j]

where b^i = B^i/W + b⁰*v^i is the spatial part of the magnetic 4-vector.
"""
@inline function physical_flux(law::SRMHDEquations, w::SVector{8}, dir::Int)
    ρ, vx, vy, vz, P, Bx, By, Bz = w
    γ_eos = law.eos.gamma

    W = lorentz_factor(vx, vy, vz)
    ε = P / ((γ_eos - 1) * ρ)
    h = 1 + ε + P / ρ

    B_sq = Bx^2 + By^2 + Bz^2
    b0, b_sq = srmhd_b_quantities(vx, vy, vz, Bx, By, Bz, W)
    rho_h_W2 = ρ * h * W^2
    Ptot = P + 0.5 * b_sq

    D = ρ * W
    tau = rho_h_W2 + B_sq - Ptot - D

    # Covariant magnetic 4-vector spatial components: b_j = B_j/W + b⁰ v_j
    bx = Bx / W + b0 * vx
    by = By / W + b0 * vy
    bz = Bz / W + b0 * vz

    # Total momentum-energy factor: (ρh + b²)W² = ρhW² + B² + b0²
    wtot = rho_h_W2 + B_sq + b0^2

    if dir == 1
        vn = vx
        bn = bx
        return SVector(
            D * vn,
            wtot * vx * vn - bx * bn + Ptot,
            wtot * vy * vn - by * bn,
            wtot * vz * vn - bz * bn,
            (wtot - D) * vn - b0 * bn,
            zero(vn),
            By * vn - Bx * vy,
            Bz * vn - Bx * vz
        )
    else  # dir == 2
        vn = vy
        bn = by
        return SVector(
            D * vn,
            wtot * vx * vn - bx * bn,
            wtot * vy * vn - by * bn + Ptot,
            wtot * vz * vn - bz * bn,
            (wtot - D) * vn - b0 * bn,
            Bx * vn - By * vx,
            zero(vn),
            Bz * vn - By * vz
        )
    end
end

# ============================================================
# Wave Speeds
# ============================================================

"""
    max_wave_speed(law::SRMHDEquations, w::SVector{8}, dir::Int) -> Real

Maximum wave speed from primitive variables.
"""
@inline function max_wave_speed(law::SRMHDEquations, w::SVector{8}, dir::Int)
    λm, λp = _srmhd_fast_speeds(law, w, dir)
    return max(abs(λm), abs(λp))
end

"""
    wave_speeds(law::SRMHDEquations, w::SVector{8}, dir::Int) -> (λ_min, λ_max)

Return the fastest left-going and right-going wave speeds.
"""
@inline function wave_speeds(law::SRMHDEquations, w::SVector{8}, dir::Int)
    return _srmhd_fast_speeds(law, w, dir)
end

"""
    _srmhd_fast_speeds(law::SRMHDEquations, w::SVector{8}, dir::Int) -> (λ_min, λ_max)

Compute the relativistic fast magnetosonic wave speeds in direction `dir`.

  λ± = [vn(1−cs²) ± cs√((1−v²)[1−v²cs² − vn²(1−cs²)])] / (1−v²cs²)

where cs² = a² + va² − a²·va² is the total signal speed squared,
a² = γP/(ρh), va² = b²/(ρh + b²).
"""
@inline function _srmhd_fast_speeds(law::SRMHDEquations, w::SVector{8}, dir::Int)
    ρ, vx, vy, vz, P, Bx, By, Bz = w
    γ_eos = law.eos.gamma

    v_sq = vx^2 + vy^2 + vz^2
    v_sq = min(v_sq, 1 - 1.0e-10)

    W = 1 / sqrt(1 - v_sq)
    ε = P / ((γ_eos - 1) * ρ)
    h = 1 + ε + P / ρ

    cs_sq = γ_eos * P / (ρ * h)
    cs_sq = min(cs_sq, 1 - 1.0e-10)

    B_sq = Bx^2 + By^2 + Bz^2
    vdotB = vx * Bx + vy * By + vz * Bz
    b_sq = B_sq / W^2 + vdotB^2

    rho_h = ρ * h
    ca_sq = b_sq / (rho_h + b_sq)
    ca_sq = min(ca_sq, 1 - 1.0e-10)

    c_ms_sq = cs_sq + ca_sq - cs_sq * ca_sq
    c_ms_sq = clamp(c_ms_sq, zero(c_ms_sq), 1 - 1.0e-10)
    c_ms = sqrt(c_ms_sq)

    vn = dir == 1 ? vx : vy

    denom = 1 - v_sq * c_ms_sq
    discriminant = (1 - v_sq) * (1 - v_sq * c_ms_sq - vn^2 * (1 - c_ms_sq))
    discriminant = max(discriminant, zero(discriminant))
    sqrt_disc = sqrt(discriminant)

    λ_minus = (vn * (1 - c_ms_sq) - c_ms * sqrt_disc) / denom
    λ_plus = (vn * (1 - c_ms_sq) + c_ms * sqrt_disc) / denom

    return λ_minus, λ_plus
end

# ============================================================
# ReflectiveBC for 1D SRMHD
# ============================================================

function apply_bc_left!(U::AbstractVector, ::ReflectiveBC, law::SRMHDEquations{1}, ncells::Int, t)
    w1 = conserved_to_primitive(law, U[3])
    w2 = conserved_to_primitive(law, U[4])
    w1_ghost = SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8])
    w2_ghost = SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8])
    U[2] = primitive_to_conserved(law, w1_ghost)
    U[1] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::ReflectiveBC, law::SRMHDEquations{1}, ncells::Int, t)
    w1 = conserved_to_primitive(law, U[ncells + 2])
    w2 = conserved_to_primitive(law, U[ncells + 1])
    w1_ghost = SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8])
    w2_ghost = SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8])
    U[ncells + 3] = primitive_to_conserved(law, w1_ghost)
    U[ncells + 4] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

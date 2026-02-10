# ============================================================
# Special Relativistic Hydrodynamics Equations
# ============================================================
#
# Conservation law for special relativistic hydrodynamics (no magnetic fields).
# Simpler than SRMHD: fewer variables and analytic-friendly con2prim.
#
# 1D: 3 variables
#   Primitive:  W = [ρ, vx, P]
#   Conserved:  U = [D, Sx, τ]
#
# 2D: 4 variables
#   Primitive:  W = [ρ, vx, vy, P]
#   Conserved:  U = [D, Sx, Sy, τ]
#
#   D = ρW                    (Lorentz-contracted density)
#   S_j = ρhW²v_j            (relativistic momentum)
#   τ = ρhW² - P - D         (energy minus rest mass)
#
# Lorentz factor: W = 1/√(1 − v²)
# Specific enthalpy: h = 1 + ε + P/ρ

"""
    SRHydroEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}

The special relativistic hydrodynamics equations in `Dim` spatial dimensions.

This is SRMHD without magnetic fields — simpler con2prim, useful for
relativistic jets and gamma-ray bursts.

## 1D (Dim=1): 3 variables
- Primitive: `W = [ρ, vx, P]`
- Conserved: `U = [D, Sx, τ]`

## 2D (Dim=2): 4 variables
- Primitive: `W = [ρ, vx, vy, P]`
- Conserved: `U = [D, Sx, Sy, τ]`

The conserved↔primitive conversion requires iterative root-finding (Con2Prim),
but is simpler than SRMHD since B = 0 eliminates magnetic terms.

# Fields
- `eos::EOS`: Equation of state.
- `con2prim_tol::Float64`: Tolerance for con2prim convergence (default 1e-12).
- `con2prim_maxiter::Int`: Maximum con2prim iterations (default 50).
"""
struct SRHydroEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}
    eos::EOS
    con2prim_tol::Float64
    con2prim_maxiter::Int
end

function SRHydroEquations{Dim}(eos::EOS; con2prim_tol = 1.0e-12, con2prim_maxiter = 50) where {Dim, EOS <: AbstractEOS}
    return SRHydroEquations{Dim, EOS}(eos, con2prim_tol, con2prim_maxiter)
end

nvariables(::SRHydroEquations{1}) = 3
nvariables(::SRHydroEquations{2}) = 4

# ============================================================
# Conservative-to-Primitive Recovery (B = 0 specialization)
# ============================================================
#
# Newton-Raphson on ξ = ρhW². With B = 0 the formulas simplify:
#   v² = S²/ξ²
#   W = 1/√(1 - v²)
#   ρ = D/W
#   P = (γ-1)/γ * (ξ/W² - ρ)
#   Residual: f(ξ) = ξ - P - (τ + D)

"""
    srhydro_con2prim(eos, u::SVector{3}, tol, maxiter) -> (SVector{3}, Con2PrimResult)

Recover 1D primitive variables `[ρ, vx, P]` from conserved variables
`[D, Sx, τ]` using Newton-Raphson on ξ = ρhW².
"""
function srhydro_con2prim(eos, u::SVector{3}, tol::Real, maxiter::Int)
    D = u[1]
    Sx = u[2]
    tau = u[3]

    γ = eos.gamma
    gm1 = γ - 1

    S_sq = Sx^2

    # Floor conserved variables
    D = max(D, 1.0e-12)

    # Initial guess: non-relativistic estimate
    E_approx = tau + D
    P_guess = max(gm1 * (E_approx - 0.5 * S_sq / max(E_approx, 1.0e-30)), 1.0e-16)
    xi = E_approx + P_guess

    FT = typeof(xi)
    converged = false
    iterations = 0
    residual = FT(Inf)

    for iter in 1:maxiter
        iterations = iter

        # v² = S²/ξ² (B = 0 simplification)
        v_sq = S_sq / xi^2
        v_sq = min(v_sq, 1 - 1.0e-10)
        v_sq = max(v_sq, zero(v_sq))

        W_sq = 1 / (1 - v_sq)
        W = sqrt(W_sq)

        rho = D / W
        rho = max(rho, 1.0e-12)

        # P = (γ-1)/γ * (ξ/W² - ρ)
        P_val = gm1 / γ * (xi / W_sq - rho)
        P_val = max(P_val, 1.0e-16)

        # Residual: f(ξ) = ξ - P - (τ + D)
        f_val = xi - P_val - tau - D
        residual = abs(f_val)

        if residual < tol * max(abs(xi), one(xi))
            converged = true
            break
        end

        # Numerical derivative
        dxi = max(abs(xi) * 1.0e-7, 1.0e-20)
        xi_p = xi + dxi

        v_sq_p = S_sq / xi_p^2
        v_sq_p = min(v_sq_p, 1 - 1.0e-10)
        v_sq_p = max(v_sq_p, zero(v_sq_p))
        W_sq_p = 1 / (1 - v_sq_p)
        W_p = sqrt(W_sq_p)
        rho_p = max(D / W_p, 1.0e-12)
        P_val_p = max(gm1 / γ * (xi_p / W_sq_p - rho_p), 1.0e-16)
        f_val_p = xi_p - P_val_p - tau - D

        df_dxi = (f_val_p - f_val) / dxi

        if abs(df_dxi) < 1.0e-30
            break
        end

        xi_new = xi - f_val / df_dxi
        xi = max(xi_new, D)  # ξ ≥ D always
    end

    # Final recovery
    vx_val = Sx / xi

    v_sq = vx_val^2
    v_sq = min(v_sq, 1 - 1.0e-10)
    W = 1 / sqrt(1 - v_sq)
    rho = max(D / W, 1.0e-12)
    P_val = max(gm1 / γ * (xi / W^2 - rho), 1.0e-16)

    w = SVector(rho, vx_val, P_val)
    result = Con2PrimResult(converged, iterations, residual)

    return w, result
end

"""
    srhydro_con2prim(eos, u::SVector{4}, tol, maxiter) -> (SVector{4}, Con2PrimResult)

Recover 2D primitive variables `[ρ, vx, vy, P]` from conserved variables
`[D, Sx, Sy, τ]` using Newton-Raphson on ξ = ρhW².
"""
function srhydro_con2prim(eos, u::SVector{4}, tol::Real, maxiter::Int)
    D = u[1]
    Sx = u[2]
    Sy = u[3]
    tau = u[4]

    γ = eos.gamma
    gm1 = γ - 1

    S_sq = Sx^2 + Sy^2

    # Floor conserved variables
    D = max(D, 1.0e-12)

    # Initial guess: non-relativistic estimate
    E_approx = tau + D
    P_guess = max(gm1 * (E_approx - 0.5 * S_sq / max(E_approx, 1.0e-30)), 1.0e-16)
    xi = E_approx + P_guess

    FT = typeof(xi)
    converged = false
    iterations = 0
    residual = FT(Inf)

    for iter in 1:maxiter
        iterations = iter

        # v² = S²/ξ² (B = 0 simplification)
        v_sq = S_sq / xi^2
        v_sq = min(v_sq, 1 - 1.0e-10)
        v_sq = max(v_sq, zero(v_sq))

        W_sq = 1 / (1 - v_sq)
        W = sqrt(W_sq)

        rho = D / W
        rho = max(rho, 1.0e-12)

        # P = (γ-1)/γ * (ξ/W² - ρ)
        P_val = gm1 / γ * (xi / W_sq - rho)
        P_val = max(P_val, 1.0e-16)

        # Residual: f(ξ) = ξ - P - (τ + D)
        f_val = xi - P_val - tau - D
        residual = abs(f_val)

        if residual < tol * max(abs(xi), one(xi))
            converged = true
            break
        end

        # Numerical derivative
        dxi = max(abs(xi) * 1.0e-7, 1.0e-20)
        xi_p = xi + dxi

        v_sq_p = S_sq / xi_p^2
        v_sq_p = min(v_sq_p, 1 - 1.0e-10)
        v_sq_p = max(v_sq_p, zero(v_sq_p))
        W_sq_p = 1 / (1 - v_sq_p)
        W_p = sqrt(W_sq_p)
        rho_p = max(D / W_p, 1.0e-12)
        P_val_p = max(gm1 / γ * (xi_p / W_sq_p - rho_p), 1.0e-16)
        f_val_p = xi_p - P_val_p - tau - D

        df_dxi = (f_val_p - f_val) / dxi

        if abs(df_dxi) < 1.0e-30
            break
        end

        xi_new = xi - f_val / df_dxi
        xi = max(xi_new, D)  # ξ ≥ D always
    end

    # Final recovery
    vx_val = Sx / xi
    vy_val = Sy / xi

    v_sq = vx_val^2 + vy_val^2
    v_sq = min(v_sq, 1 - 1.0e-10)
    W = 1 / sqrt(1 - v_sq)
    rho = max(D / W, 1.0e-12)
    P_val = max(gm1 / γ * (xi / W^2 - rho), 1.0e-16)

    w = SVector(rho, vx_val, vy_val, P_val)
    result = Con2PrimResult(converged, iterations, residual)

    return w, result
end

# ============================================================
# Conserved <-> Primitive Conversion
# ============================================================

"""
    conserved_to_primitive(law::SRHydroEquations{1}, u::SVector{3}) -> SVector{3}

Convert 1D conserved `[D, Sx, τ]` to primitive `[ρ, vx, P]` via iterative con2prim.
"""
@inline function conserved_to_primitive(law::SRHydroEquations{1}, u::SVector{3})
    w, _ = srhydro_con2prim(law.eos, u, law.con2prim_tol, law.con2prim_maxiter)
    return w
end

"""
    conserved_to_primitive(law::SRHydroEquations{2}, u::SVector{4}) -> SVector{4}

Convert 2D conserved `[D, Sx, Sy, τ]` to primitive `[ρ, vx, vy, P]` via iterative con2prim.
"""
@inline function conserved_to_primitive(law::SRHydroEquations{2}, u::SVector{4})
    w, _ = srhydro_con2prim(law.eos, u, law.con2prim_tol, law.con2prim_maxiter)
    return w
end

"""
    primitive_to_conserved(law::SRHydroEquations{1}, w::SVector{3}) -> SVector{3}

Convert 1D primitive `[ρ, vx, P]` to conserved `[D, Sx, τ]`.
"""
@inline function primitive_to_conserved(law::SRHydroEquations{1}, w::SVector{3})
    ρ, vx, P = w
    γ_eos = law.eos.gamma

    W = lorentz_factor(vx, 0.0, 0.0)
    W_sq = W^2

    ε = P / ((γ_eos - 1) * ρ)
    h = 1 + ε + P / ρ

    rho_h_W2 = ρ * h * W_sq

    D = ρ * W
    Sx_c = rho_h_W2 * vx
    tau = rho_h_W2 - P - D

    return SVector(D, Sx_c, tau)
end

"""
    primitive_to_conserved(law::SRHydroEquations{2}, w::SVector{4}) -> SVector{4}

Convert 2D primitive `[ρ, vx, vy, P]` to conserved `[D, Sx, Sy, τ]`.
"""
@inline function primitive_to_conserved(law::SRHydroEquations{2}, w::SVector{4})
    ρ, vx, vy, P = w
    γ_eos = law.eos.gamma

    W = lorentz_factor(vx, vy, 0.0)
    W_sq = W^2

    ε = P / ((γ_eos - 1) * ρ)
    h = 1 + ε + P / ρ

    rho_h_W2 = ρ * h * W_sq

    D = ρ * W
    Sx_c = rho_h_W2 * vx
    Sy_c = rho_h_W2 * vy
    tau = rho_h_W2 - P - D

    return SVector(D, Sx_c, Sy_c, tau)
end

# ============================================================
# Physical Flux
# ============================================================

"""
    physical_flux(law::SRHydroEquations{1}, w::SVector{3}, ::Int) -> SVector{3}

Compute the 1D SR hydro flux from primitive variables `[ρ, vx, P]`.

    F = [D*vx, Sx*vx + P, (τ + P)*vx]
"""
@inline function physical_flux(law::SRHydroEquations{1}, w::SVector{3}, ::Int)
    ρ, vx, P = w
    γ_eos = law.eos.gamma

    W = lorentz_factor(vx, 0.0, 0.0)
    ε = P / ((γ_eos - 1) * ρ)
    h = 1 + ε + P / ρ

    rho_h_W2 = ρ * h * W^2

    D = ρ * W
    Sx_c = rho_h_W2 * vx
    tau = rho_h_W2 - P - D

    return SVector(D * vx, Sx_c * vx + P, (tau + P) * vx)
end

"""
    physical_flux(law::SRHydroEquations{2}, w::SVector{4}, dir::Int) -> SVector{4}

Compute the 2D SR hydro flux in direction `dir` (1=x, 2=y) from primitive
variables `[ρ, vx, vy, P]`.

    Fx = [D*vx, ρhW²vx² + P, ρhW²vx*vy, (τ+P)*vx]
    Fy = [D*vy, ρhW²vx*vy, ρhW²vy² + P, (τ+P)*vy]
"""
@inline function physical_flux(law::SRHydroEquations{2}, w::SVector{4}, dir::Int)
    ρ, vx, vy, P = w
    γ_eos = law.eos.gamma

    W = lorentz_factor(vx, vy, 0.0)
    ε = P / ((γ_eos - 1) * ρ)
    h = 1 + ε + P / ρ

    rho_h_W2 = ρ * h * W^2

    D = ρ * W
    tau = rho_h_W2 - P - D

    if dir == 1
        return SVector(
            D * vx,
            rho_h_W2 * vx * vx + P,
            rho_h_W2 * vx * vy,
            (tau + P) * vx
        )
    else  # dir == 2
        return SVector(
            D * vy,
            rho_h_W2 * vx * vy,
            rho_h_W2 * vy * vy + P,
            (tau + P) * vy
        )
    end
end

# ============================================================
# Wave Speeds (relativistic sound cone, no B)
# ============================================================

"""
    max_wave_speed(law::SRHydroEquations{1}, w::SVector{3}, dir::Int) -> Real

Maximum wave speed from 1D primitive variables.
"""
@inline function max_wave_speed(law::SRHydroEquations{1}, w::SVector{3}, dir::Int)
    λm, λp = _srhydro_sound_speeds(law, w, dir)
    return max(abs(λm), abs(λp))
end

"""
    max_wave_speed(law::SRHydroEquations{2}, w::SVector{4}, dir::Int) -> Real

Maximum wave speed from 2D primitive variables in direction `dir`.
"""
@inline function max_wave_speed(law::SRHydroEquations{2}, w::SVector{4}, dir::Int)
    λm, λp = _srhydro_sound_speeds(law, w, dir)
    return max(abs(λm), abs(λp))
end

"""
    wave_speeds(law::SRHydroEquations{1}, w::SVector{3}, dir::Int) -> (λ_min, λ_max)

Return the fastest left-going and right-going wave speeds for 1D SR hydro.
"""
@inline function wave_speeds(law::SRHydroEquations{1}, w::SVector{3}, dir::Int)
    return _srhydro_sound_speeds(law, w, dir)
end

"""
    wave_speeds(law::SRHydroEquations{2}, w::SVector{4}, dir::Int) -> (λ_min, λ_max)

Return the fastest left-going and right-going wave speeds for 2D SR hydro.
"""
@inline function wave_speeds(law::SRHydroEquations{2}, w::SVector{4}, dir::Int)
    return _srhydro_sound_speeds(law, w, dir)
end

"""
    _srhydro_sound_speeds(law::SRHydroEquations{1}, w::SVector{3}, dir::Int) -> (λ_min, λ_max)

Compute the relativistic sound speeds for 1D SR hydro.

    λ± = [vn(1-cs²) ± cs√((1-v²)(1-v²cs²-vn²(1-cs²)))] / (1-v²cs²)

where cs² = γP/(ρh), and v² = vx² for 1D.
"""
@inline function _srhydro_sound_speeds(law::SRHydroEquations{1}, w::SVector{3}, dir::Int)
    ρ, vx, P = w
    γ_eos = law.eos.gamma

    v_sq = vx^2
    v_sq = min(v_sq, 1 - 1.0e-10)

    ε = P / ((γ_eos - 1) * ρ)
    h = 1 + ε + P / ρ

    cs_sq = γ_eos * P / (ρ * h)
    cs_sq = min(cs_sq, 1 - 1.0e-10)
    cs = sqrt(cs_sq)

    vn = vx

    denom = 1 - v_sq * cs_sq
    discriminant = (1 - v_sq) * (1 - v_sq * cs_sq - vn^2 * (1 - cs_sq))
    discriminant = max(discriminant, zero(discriminant))
    sqrt_disc = sqrt(discriminant)

    λ_minus = (vn * (1 - cs_sq) - cs * sqrt_disc) / denom
    λ_plus = (vn * (1 - cs_sq) + cs * sqrt_disc) / denom

    return λ_minus, λ_plus
end

"""
    _srhydro_sound_speeds(law::SRHydroEquations{2}, w::SVector{4}, dir::Int) -> (λ_min, λ_max)

Compute the relativistic sound speeds for 2D SR hydro.

    λ± = [vn(1-cs²) ± cs√((1-v²)(1-v²cs²-vn²(1-cs²)))] / (1-v²cs²)

where cs² = γP/(ρh), vn is the normal velocity, and v² = vx² + vy².
"""
@inline function _srhydro_sound_speeds(law::SRHydroEquations{2}, w::SVector{4}, dir::Int)
    ρ, vx, vy, P = w
    γ_eos = law.eos.gamma

    v_sq = vx^2 + vy^2
    v_sq = min(v_sq, 1 - 1.0e-10)

    ε = P / ((γ_eos - 1) * ρ)
    h = 1 + ε + P / ρ

    cs_sq = γ_eos * P / (ρ * h)
    cs_sq = min(cs_sq, 1 - 1.0e-10)
    cs = sqrt(cs_sq)

    vn = dir == 1 ? vx : vy

    denom = 1 - v_sq * cs_sq
    discriminant = (1 - v_sq) * (1 - v_sq * cs_sq - vn^2 * (1 - cs_sq))
    discriminant = max(discriminant, zero(discriminant))
    sqrt_disc = sqrt(discriminant)

    λ_minus = (vn * (1 - cs_sq) - cs * sqrt_disc) / denom
    λ_plus = (vn * (1 - cs_sq) + cs * sqrt_disc) / denom

    return λ_minus, λ_plus
end

# ============================================================
# ReflectiveBC for 1D SRHydro
# ============================================================

function apply_bc_left!(U::AbstractVector, ::ReflectiveBC, law::SRHydroEquations{1}, ncells::Int, t)
    w1 = conserved_to_primitive(law, U[3])
    w2 = conserved_to_primitive(law, U[4])
    w1_ghost = SVector(w1[1], -w1[2], w1[3])
    w2_ghost = SVector(w2[1], -w2[2], w2[3])
    U[2] = primitive_to_conserved(law, w1_ghost)
    U[1] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::ReflectiveBC, law::SRHydroEquations{1}, ncells::Int, t)
    w1 = conserved_to_primitive(law, U[ncells + 2])
    w2 = conserved_to_primitive(law, U[ncells + 1])
    w1_ghost = SVector(w1[1], -w1[2], w1[3])
    w2_ghost = SVector(w2[1], -w2[2], w2[3])
    U[ncells + 3] = primitive_to_conserved(law, w1_ghost)
    U[ncells + 4] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

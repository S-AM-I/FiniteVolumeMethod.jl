# ============================================================
# Ideal MHD Equations
# ============================================================
#
# 8-variable system for ideal magnetohydrodynamics.
#
# Primitive:  W = [ρ, vx, vy, vz, P, Bx, By, Bz]
# Conserved:  U = [ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz]
#
# Total energy:   E = P/(γ-1) + ½ρ|v|² + ½|B|²
# Total pressure: P_tot = P + ½|B|²
#
# In 1D, Bx is constant (its flux is zero).
# In 2D, ∇·B = 0 is maintained via constrained transport.

"""
    IdealMHDEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}

The ideal magnetohydrodynamics equations in `Dim` spatial dimensions.

## Variables (8 components in all dimensions)
- Primitive: `W = [ρ, vx, vy, vz, P, Bx, By, Bz]`
- Conserved: `U = [ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz]`

Total energy: `E = P/(γ-1) + ½ρ|v|² + ½|B|²`

## 1D
Bx is constant (its x-flux is zero). The 1D system still has 8 variables
but the Bx equation is trivially satisfied.

## 2D
In 2D with constrained transport, the ∇·B = 0 constraint is maintained
via a staggered update of face-centered magnetic fields.

# Fields
- `eos::EOS`: Equation of state.
"""
struct IdealMHDEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}
    eos::EOS
end

IdealMHDEquations{Dim}(eos::EOS) where {Dim, EOS <: AbstractEOS} = IdealMHDEquations{Dim, EOS}(eos)

nvariables(::IdealMHDEquations) = 8

# ============================================================
# Conserved ↔ Primitive Conversion
# ============================================================

"""
    conserved_to_primitive(law::IdealMHDEquations, u::SVector{8}) -> SVector{8}

Convert MHD conserved `[ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz]` to
primitive `[ρ, vx, vy, vz, P, Bx, By, Bz]`.
"""
@inline function conserved_to_primitive(law::IdealMHDEquations, u::SVector{8})
    ρ = u[1]
    vx = u[2] / ρ
    vy = u[3] / ρ
    vz = u[4] / ρ
    E = u[5]
    Bx = u[6]
    By = u[7]
    Bz = u[8]
    KE = 0.5 * ρ * (vx^2 + vy^2 + vz^2)
    ME = 0.5 * (Bx^2 + By^2 + Bz^2)
    ε = (E - KE - ME) / ρ
    P = pressure(law.eos, ρ, ε)
    return SVector(ρ, vx, vy, vz, P, Bx, By, Bz)
end

"""
    primitive_to_conserved(law::IdealMHDEquations, w::SVector{8}) -> SVector{8}

Convert MHD primitive `[ρ, vx, vy, vz, P, Bx, By, Bz]` to
conserved `[ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz]`.
"""
@inline function primitive_to_conserved(law::IdealMHDEquations, w::SVector{8})
    ρ, vx, vy, vz, P, Bx, By, Bz = w
    KE = 0.5 * ρ * (vx^2 + vy^2 + vz^2)
    ME = 0.5 * (Bx^2 + By^2 + Bz^2)
    E = P / (law.eos.gamma - 1) + KE + ME
    return SVector(ρ, ρ * vx, ρ * vy, ρ * vz, E, Bx, By, Bz)
end

# ============================================================
# Physical Flux
# ============================================================

"""
    physical_flux(law::IdealMHDEquations, w::SVector{8}, dir::Int) -> SVector{8}

Compute the MHD flux in direction `dir` (1=x, 2=y) from primitive variables.

x-flux: `F = [ρvx, ρvx²+Ptot-Bx², ρvx·vy-Bx·By, ρvx·vz-Bx·Bz,
              (E+Ptot)vx-Bx(v·B), 0, By·vx-Bx·vy, Bz·vx-Bx·vz]`

y-flux: `G = [ρvy, ρvx·vy-Bx·By, ρvy²+Ptot-By², ρvy·vz-By·Bz,
              (E+Ptot)vy-By(v·B), Bx·vy-By·vx, 0, Bz·vy-By·vz]`
"""
@inline function physical_flux(law::IdealMHDEquations, w::SVector{8}, dir::Int)
    ρ, vx, vy, vz, P, Bx, By, Bz = w
    v_dot_B = vx * Bx + vy * By + vz * Bz
    B_sq = Bx^2 + By^2 + Bz^2
    P_tot = P + 0.5 * B_sq
    KE = 0.5 * ρ * (vx^2 + vy^2 + vz^2)
    E = P / (law.eos.gamma - 1) + KE + 0.5 * B_sq

    if dir == 1  # x-flux
        return SVector(
            ρ * vx,
            ρ * vx^2 + P_tot - Bx^2,
            ρ * vx * vy - Bx * By,
            ρ * vx * vz - Bx * Bz,
            (E + P_tot) * vx - Bx * v_dot_B,
            zero(Bx),
            By * vx - Bx * vy,
            Bz * vx - Bx * vz
        )
    else  # y-flux (dir == 2)
        return SVector(
            ρ * vy,
            ρ * vx * vy - Bx * By,
            ρ * vy^2 + P_tot - By^2,
            ρ * vy * vz - By * Bz,
            (E + P_tot) * vy - By * v_dot_B,
            Bx * vy - By * vx,
            zero(By),
            Bz * vy - By * vz
        )
    end
end

# ============================================================
# Wave Speeds
# ============================================================

"""
    fast_magnetosonic_speed(law::IdealMHDEquations, w::SVector{8}, dir::Int) -> cf

Compute the fast magnetosonic speed in direction `dir`:
  `cf² = ½(a² + b² + √((a² + b²)² - 4 a² bn²))`
where `a² = γP/ρ`, `b² = |B|²/ρ`, `bn = Bn/√ρ`.
"""
@inline function fast_magnetosonic_speed(law::IdealMHDEquations, w::SVector{8}, dir::Int)
    ρ, vx, vy, vz, P, Bx, By, Bz = w
    γ = law.eos.gamma

    a_sq = γ * P / ρ                       # sound speed²
    B_sq = Bx^2 + By^2 + Bz^2
    b_sq = B_sq / ρ                         # total Alfvén speed²
    Bn = dir == 1 ? Bx : By
    bn_sq = Bn^2 / ρ                        # normal Alfvén speed²

    discriminant = (a_sq + b_sq)^2 - 4 * a_sq * bn_sq
    discriminant = max(discriminant, zero(discriminant))
    cf_sq = 0.5 * (a_sq + b_sq + sqrt(discriminant))
    return sqrt(max(cf_sq, zero(cf_sq)))
end

"""
    slow_magnetosonic_speed(law::IdealMHDEquations, w::SVector{8}, dir::Int) -> cs

Compute the slow magnetosonic speed in direction `dir`:
  `cs² = ½(a² + b² - √((a² + b²)² - 4 a² bn²))`
"""
@inline function slow_magnetosonic_speed(law::IdealMHDEquations, w::SVector{8}, dir::Int)
    ρ, vx, vy, vz, P, Bx, By, Bz = w
    γ = law.eos.gamma

    a_sq = γ * P / ρ
    B_sq = Bx^2 + By^2 + Bz^2
    b_sq = B_sq / ρ
    Bn = dir == 1 ? Bx : By
    bn_sq = Bn^2 / ρ

    discriminant = (a_sq + b_sq)^2 - 4 * a_sq * bn_sq
    discriminant = max(discriminant, zero(discriminant))
    cs_sq = 0.5 * (a_sq + b_sq - sqrt(discriminant))
    return sqrt(max(cs_sq, zero(cs_sq)))
end

"""
    max_wave_speed(law::IdealMHDEquations, w::SVector{8}, dir::Int) -> Real

Maximum wave speed `|vn| + cf` from primitive variables.
"""
@inline function max_wave_speed(law::IdealMHDEquations, w::SVector{8}, dir::Int)
    vn = dir == 1 ? w[2] : w[3]
    cf = fast_magnetosonic_speed(law, w, dir)
    return abs(vn) + cf
end

"""
    wave_speeds(law::IdealMHDEquations, w::SVector{8}, dir::Int) -> (λ_min, λ_max)

Return the fastest left-going and right-going wave speeds (fast magnetosonic).
"""
@inline function wave_speeds(law::IdealMHDEquations, w::SVector{8}, dir::Int)
    vn = dir == 1 ? w[2] : w[3]
    cf = fast_magnetosonic_speed(law, w, dir)
    return vn - cf, vn + cf
end

# ============================================================
# ReflectiveBC for 1D MHD
# ============================================================
# Negate normal velocity, keep everything else (including B).

function apply_bc_left!(U::AbstractVector, ::ReflectiveBC, law::IdealMHDEquations{1}, ncells::Int, t)
    w1 = conserved_to_primitive(law, U[3])
    w2 = conserved_to_primitive(law, U[4])
    w1_ghost = SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8])
    w2_ghost = SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8])
    U[2] = primitive_to_conserved(law, w1_ghost)
    U[1] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::ReflectiveBC, law::IdealMHDEquations{1}, ncells::Int, t)
    w1 = conserved_to_primitive(law, U[ncells + 2])
    w2 = conserved_to_primitive(law, U[ncells + 1])
    w1_ghost = SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8])
    w2_ghost = SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8])
    U[ncells + 3] = primitive_to_conserved(law, w1_ghost)
    U[ncells + 4] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

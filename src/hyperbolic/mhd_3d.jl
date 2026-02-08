# ============================================================
# 3D MHD Extensions
# ============================================================
#
# Adds dir==3 (z-direction) support for IdealMHDEquations.
# The existing physical_flux in mhd.jl only handles dir==1 and dir==2.
#
# z-flux:
#   Hz = [ρvz, ρvx·vz - Bx·Bz, ρvy·vz - By·Bz, ρvz² + Ptot - Bz²,
#         (E+Ptot)vz - Bz(v·B), Bx·vz - Bz·vx, By·vz - Bz·vy, 0]

"""
    physical_flux(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int) -> SVector{8}

Compute the MHD flux in direction `dir` (1=x, 2=y, 3=z) from primitive variables
for the 3D IdealMHDEquations.
"""
@inline function physical_flux(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int)
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
    elseif dir == 2  # y-flux
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
    else  # z-flux (dir == 3)
        return SVector(
            ρ * vz,
            ρ * vx * vz - Bx * Bz,
            ρ * vy * vz - By * Bz,
            ρ * vz^2 + P_tot - Bz^2,
            (E + P_tot) * vz - Bz * v_dot_B,
            Bx * vz - Bz * vx,
            By * vz - Bz * vy,
            zero(Bz)
        )
    end
end

"""
    fast_magnetosonic_speed(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int) -> cf

Compute the fast magnetosonic speed in direction `dir` (1=x, 2=y, 3=z).
"""
@inline function fast_magnetosonic_speed(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int)
    ρ, vx, vy, vz, P, Bx, By, Bz = w
    γ = law.eos.gamma

    a_sq = γ * P / ρ
    B_sq = Bx^2 + By^2 + Bz^2
    b_sq = B_sq / ρ
    if dir == 1
        Bn = Bx
    elseif dir == 2
        Bn = By
    else
        Bn = Bz
    end
    bn_sq = Bn^2 / ρ

    discriminant = (a_sq + b_sq)^2 - 4 * a_sq * bn_sq
    discriminant = max(discriminant, zero(discriminant))
    cf_sq = 0.5 * (a_sq + b_sq + sqrt(discriminant))
    return sqrt(max(cf_sq, zero(cf_sq)))
end

"""
    slow_magnetosonic_speed(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int) -> cs

Compute the slow magnetosonic speed in direction `dir` (1=x, 2=y, 3=z).
"""
@inline function slow_magnetosonic_speed(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int)
    ρ, vx, vy, vz, P, Bx, By, Bz = w
    γ = law.eos.gamma

    a_sq = γ * P / ρ
    B_sq = Bx^2 + By^2 + Bz^2
    b_sq = B_sq / ρ
    if dir == 1
        Bn = Bx
    elseif dir == 2
        Bn = By
    else
        Bn = Bz
    end
    bn_sq = Bn^2 / ρ

    discriminant = (a_sq + b_sq)^2 - 4 * a_sq * bn_sq
    discriminant = max(discriminant, zero(discriminant))
    cs_sq = 0.5 * (a_sq + b_sq - sqrt(discriminant))
    return sqrt(max(cs_sq, zero(cs_sq)))
end

"""
    max_wave_speed(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int) -> Real

Maximum wave speed `|vn| + cf` from primitive variables for 3D MHD.
"""
@inline function max_wave_speed(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int)
    if dir == 1
        vn = w[2]
    elseif dir == 2
        vn = w[3]
    else
        vn = w[4]
    end
    cf = fast_magnetosonic_speed(law, w, dir)
    return abs(vn) + cf
end

"""
    wave_speeds(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int) -> (λ_min, λ_max)

Return the fastest left-going and right-going wave speeds for 3D MHD.
"""
@inline function wave_speeds(law::IdealMHDEquations{3}, w::SVector{8}, dir::Int)
    if dir == 1
        vn = w[2]
    elseif dir == 2
        vn = w[3]
    else
        vn = w[4]
    end
    cf = fast_magnetosonic_speed(law, w, dir)
    return vn - cf, vn + cf
end

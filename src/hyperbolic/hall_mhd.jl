# ============================================================
# Hall MHD Equations
# ============================================================
#
# 8-variable system extending ideal MHD with the Hall term.
#
# Primitive:  W = [ρ, vx, vy, vz, P, Bx, By, Bz]
# Conserved:  U = [ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz]
#
# The Hall term arises from the generalized Ohm's law:
#   E = -v×B + (J×B)/(n_e*e) + η*J
#
# In the FVM context, the hyperbolic part is identical to ideal MHD.
# The Hall term (J×B)/(n_e*e) = di² * (J×B)/ρ modifies the induction
# equation via an additional flux correction that depends on J = ∇×B.
#
# The Hall term introduces dispersive whistler waves with speed
# c_w = di * k * B / √ρ, making the CFL condition resolution-dependent.

"""
    HallMHDEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}

The Hall magnetohydrodynamics equations in `Dim` spatial dimensions.

Extends ideal MHD with the Hall term from the generalized Ohm's law.
The hyperbolic (inviscid) part is identical to [`IdealMHDEquations`](@ref);
the Hall term adds a flux correction to the induction equation.

## Variables (8 components in all dimensions)
- Primitive: `W = [ρ, vx, vy, vz, P, Bx, By, Bz]`
- Conserved: `U = [ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz]`

## Hall term
The Hall electric field:
  `E_Hall = di² (J × B) / ρ`
where `J = ∇×B` is the current density and `di` is the ion inertial length.

## Resistivity
Optional Ohmic resistivity `η` adds diffusive flux `η J` to the induction equation.

## Whistler waves
The Hall term introduces whistler waves with phase speed `c_w ~ di k vA`,
making the CFL condition resolution-dependent (parabolic-like scaling).

# Fields
- `eos::EOS`: Equation of state.
- `di::Float64`: Ion inertial length (Hall parameter).
- `eta::Float64`: Resistivity (default 0).
"""
struct HallMHDEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}
    eos::EOS
    di::Float64
    eta::Float64
end

function HallMHDEquations{Dim}(eos::EOS; di = 1.0, eta = 0.0) where {Dim, EOS <: AbstractEOS}
    return HallMHDEquations{Dim, EOS}(eos, di, eta)
end

# ============================================================
# Internal helper: delegate to IdealMHDEquations
# ============================================================

"""
    _ideal_mhd(law::HallMHDEquations{Dim}) -> IdealMHDEquations{Dim}

Construct the underlying [`IdealMHDEquations`](@ref) for delegation.
"""
@inline _ideal_mhd(law::HallMHDEquations{Dim}) where {Dim} = IdealMHDEquations{Dim}(law.eos)

# ============================================================
# Delegation to IdealMHDEquations
# ============================================================

nvariables(::HallMHDEquations) = 8

@inline conserved_to_primitive(law::HallMHDEquations, u::SVector{8}) = conserved_to_primitive(_ideal_mhd(law), u)
@inline primitive_to_conserved(law::HallMHDEquations, w::SVector{8}) = primitive_to_conserved(_ideal_mhd(law), w)
@inline physical_flux(law::HallMHDEquations, w::SVector{8}, dir::Int) = physical_flux(_ideal_mhd(law), w, dir)
@inline max_wave_speed(law::HallMHDEquations, w::SVector{8}, dir::Int) = max_wave_speed(_ideal_mhd(law), w, dir)
@inline wave_speeds(law::HallMHDEquations, w::SVector{8}, dir::Int) = wave_speeds(_ideal_mhd(law), w, dir)

# ============================================================
# HLLD Riemann solver forwarding
# ============================================================

"""
    solve_riemann(::HLLDSolver, law::HallMHDEquations, wL, wR, dir)

Forward the HLLD Riemann solve to [`IdealMHDEquations`](@ref).
The Hall term is a non-ideal correction applied separately.
"""
function solve_riemann(s::HLLDSolver, law::HallMHDEquations, wL::SVector{8}, wR::SVector{8}, dir::Int)
    return solve_riemann(s, _ideal_mhd(law), wL, wR, dir)
end

# ============================================================
# Whistler wave speed
# ============================================================

"""
    whistler_speed(law::HallMHDEquations, rho, B_mag, dx) -> Real

Compute the whistler wave speed `c_w = di * |B| / (√ρ * dx)`.

The whistler dispersion relation gives `ω = di k² vA`, so the
phase speed is `c_w = di k vA` where `k ~ 1/dx` and `vA = |B|/√ρ`.
"""
@inline function whistler_speed(law::HallMHDEquations, rho, B_mag, dx)
    return law.di * B_mag / (sqrt(rho) * dx)
end

# ============================================================
# Hall flux corrections
# ============================================================

"""
    hall_flux_x(law::HallMHDEquations, uL::SVector{8}, uR::SVector{8}, dx) -> SVector{8}

Compute the Hall flux correction at an x-direction face.

Uses central differences to approximate the current density
`J = ∇×B` from the left and right conserved states:
- `Jy = -∂Bz/∂x`
- `Jz = ∂By/∂x`

The Hall electric field modifies the induction equation:
- `E_Hall_y = di²/ρ * (Jz Bx - Jx Bz)`
- `E_Hall_z = di²/ρ * (Jx By - Jy Bx)`

In 1D (x-only), `Jx = 0`.

Resistive terms `η J` are included when `eta > 0`.
"""
@inline function hall_flux_x(law::HallMHDEquations, uL::SVector{8}, uR::SVector{8}, dx)
    # Extract magnetic field components
    BxL, ByL, BzL = uL[6], uL[7], uL[8]
    BxR, ByR, BzR = uR[6], uR[7], uR[8]

    # Face-averaged density and magnetic field
    ρ_face = 0.5 * (uL[1] + uR[1])
    Bx_face = 0.5 * (BxL + BxR)
    By_face = 0.5 * (ByL + ByR)
    Bz_face = 0.5 * (BzL + BzR)

    # Current density components from ∇×B (x-direction differences only)
    # Jx = ∂Bz/∂y - ∂By/∂z  (not available from x-face neighbors alone, set to 0)
    # Jy = ∂Bx/∂z - ∂Bz/∂x = -∂Bz/∂x  (in 1D/2D, ∂Bx/∂z = 0)
    # Jz = ∂By/∂x - ∂Bx/∂y = ∂By/∂x   (in 1D/2D, ∂Bx/∂y = 0)
    Jy = -(BzR - BzL) / dx
    Jz = (ByR - ByL) / dx

    # Hall electric field: E_Hall = di²/ρ * (J × B)
    di_sq = law.di^2
    coeff = di_sq / ρ_face

    # (J × B)_y = Jz*Bx - Jx*Bz  (Jx = 0 at x-face)
    # (J × B)_z = Jx*By - Jy*Bx  (Jx = 0 at x-face)
    E_hall_y = coeff * (Jz * Bx_face)
    E_hall_z = coeff * (-Jy * Bx_face)

    # Resistive contribution: E_res = η * J
    if law.eta > 0
        E_hall_y += law.eta * Jy
        E_hall_z += law.eta * Jz
    end

    # The induction equation in conservation form is ∂B/∂t + ∇×E = 0.
    # The x-flux for the induction equation contributes to By and Bz:
    #   F_By = -E_z  (flux of By in x-direction)
    #   F_Bz =  E_y  (flux of Bz in x-direction)
    # Energy flux from Hall/resistive: v_Hall · (J×B) contributes to energy,
    # but the Hall term conserves total energy. The energy correction is
    # E_Hall × B contribution:
    #   F_E = (E_Hall × B)_x = E_hall_y * Bz_face - E_hall_z * By_face
    F_E = E_hall_y * Bz_face - E_hall_z * By_face

    return SVector(
        zero(F_E),      # mass: no correction
        zero(F_E),      # momentum-x: no correction
        zero(F_E),      # momentum-y: no correction
        zero(F_E),      # momentum-z: no correction
        F_E,            # energy: Poynting flux correction
        zero(F_E),      # Bx: no correction (Bx const in 1D, CT in 2D)
        -E_hall_z,      # By: -Ez
        E_hall_y        # Bz: +Ey
    )
end

"""
    hall_flux_y(law::HallMHDEquations, uB::SVector{8}, uT::SVector{8}, dy) -> SVector{8}

Compute the Hall flux correction at a y-direction face.

Uses central differences to approximate the current density
`J = ∇×B` from the bottom and top conserved states:
- `Jx = ∂Bz/∂y`
- `Jz = -∂Bx/∂y`

The Hall electric field modifies the induction equation:
- `E_Hall_x = di²/ρ * (Jy Bz - Jz By)`
- `E_Hall_z = di²/ρ * (Jx By - Jy Bx)`

In the y-sweep, `Jy = 0` (only y-direction differences available).

Resistive terms `η J` are included when `eta > 0`.
"""
@inline function hall_flux_y(law::HallMHDEquations, uB::SVector{8}, uT::SVector{8}, dy)
    # Extract magnetic field components
    BxB, ByB, BzB = uB[6], uB[7], uB[8]
    BxT, ByT, BzT = uT[6], uT[7], uT[8]

    # Face-averaged density and magnetic field
    ρ_face = 0.5 * (uB[1] + uT[1])
    Bx_face = 0.5 * (BxB + BxT)
    By_face = 0.5 * (ByB + ByT)
    Bz_face = 0.5 * (BzB + BzT)

    # Current density components from ∇×B (y-direction differences only)
    # Jx = ∂Bz/∂y - ∂By/∂z = ∂Bz/∂y  (in 2D, ∂By/∂z = 0)
    # Jy = ∂Bx/∂z - ∂Bz/∂x  (not available from y-face neighbors alone, set to 0)
    # Jz = ∂By/∂x - ∂Bx/∂y = -∂Bx/∂y  (in 2D, ∂By/∂x not available)
    Jx = (BzT - BzB) / dy
    Jz = -(BxT - BxB) / dy

    # Hall electric field: E_Hall = di²/ρ * (J × B)
    di_sq = law.di^2
    coeff = di_sq / ρ_face

    # (J × B)_x = Jy*Bz - Jz*By  (Jy = 0 at y-face)
    # (J × B)_z = Jx*By - Jy*Bx  (Jy = 0 at y-face)
    E_hall_x = coeff * (-Jz * By_face)
    E_hall_z = coeff * (Jx * By_face)

    # Resistive contribution: E_res = η * J
    if law.eta > 0
        E_hall_x += law.eta * Jx
        E_hall_z += law.eta * Jz
    end

    # The y-flux for the induction equation:
    #   G_Bx =  E_z  (flux of Bx in y-direction)
    #   G_Bz = -E_x  (flux of Bz in y-direction)
    # Energy flux: (E_Hall × B)_y = E_hall_z * Bx_face - E_hall_x * Bz_face
    F_E = E_hall_z * Bx_face - E_hall_x * Bz_face

    return SVector(
        zero(F_E),      # mass: no correction
        zero(F_E),      # momentum-x: no correction
        zero(F_E),      # momentum-y: no correction
        zero(F_E),      # momentum-z: no correction
        F_E,            # energy: Poynting flux correction
        E_hall_z,       # Bx: +Ez
        zero(F_E),      # By: no correction (By const in y-sweep, CT in 2D)
        -E_hall_x       # Bz: -Ex
    )
end

# ============================================================
# Hall CFL (whistler-limited time step)
# ============================================================

"""
    hall_dt(law::HallMHDEquations{1}, U, dx, nc) -> Real

Compute the maximum stable time step for the Hall term in 1D.

The whistler wave speed scales as `c_w ~ di |B| / (√ρ dx)`, giving
a parabolic-like CFL constraint: `dt_hall = 0.5 dx / max(c_w)`.

# Arguments
- `U`: Vector of conserved state `SVector{8}` (with ghost cells, 2 per side).
- `dx`: Cell width.
- `nc`: Number of interior cells.
"""
function hall_dt(law::HallMHDEquations{1}, U, dx, nc)
    max_cw = zero(dx)
    @inbounds for i in 3:(nc + 2)
        ρ = U[i][1]
        Bx, By, Bz = U[i][6], U[i][7], U[i][8]
        B_mag = sqrt(Bx^2 + By^2 + Bz^2)
        cw = whistler_speed(law, ρ, B_mag, dx)
        max_cw = max(max_cw, cw)
    end
    if max_cw < eps(dx)
        return typeof(dx)(Inf)
    end
    return 0.5 * dx / max_cw
end

"""
    hall_dt(law::HallMHDEquations{2}, U, dx, dy, nx, ny) -> Real

Compute the maximum stable time step for the Hall term in 2D.

Uses the multi-dimensional CFL constraint:
  `dt_hall = 0.5 / max(c_w_x/dx + c_w_y/dy)`

# Arguments
- `U`: 2D array of conserved state `SVector{8}` (with 2 ghost cells per side).
- `dx`, `dy`: Cell widths.
- `nx`, `ny`: Number of interior cells in each direction.
"""
function hall_dt(law::HallMHDEquations{2}, U, dx, dy, nx, ny)
    max_cw = zero(dx)
    @inbounds for j in 3:(ny + 2)
        for i in 3:(nx + 2)
            ρ = U[i, j][1]
            Bx, By, Bz = U[i, j][6], U[i, j][7], U[i, j][8]
            B_mag = sqrt(Bx^2 + By^2 + Bz^2)
            cw_x = whistler_speed(law, ρ, B_mag, dx)
            cw_y = whistler_speed(law, ρ, B_mag, dy)
            cw = cw_x / dx + cw_y / dy
            max_cw = max(max_cw, cw)
        end
    end
    if max_cw < eps(dx)
        return typeof(dx)(Inf)
    end
    return 0.5 / max_cw
end

# ============================================================
# ReflectiveBC for 1D Hall MHD
# ============================================================
# Negate normal velocity, keep everything else (including B).

function apply_bc_left!(U::AbstractVector, ::ReflectiveBC, law::HallMHDEquations{1}, ncells::Int, t)
    w1 = conserved_to_primitive(law, U[3])
    w2 = conserved_to_primitive(law, U[4])
    w1_ghost = SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8])
    w2_ghost = SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8])
    U[2] = primitive_to_conserved(law, w1_ghost)
    U[1] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::ReflectiveBC, law::HallMHDEquations{1}, ncells::Int, t)
    w1 = conserved_to_primitive(law, U[ncells + 2])
    w2 = conserved_to_primitive(law, U[ncells + 1])
    w1_ghost = SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8])
    w2_ghost = SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8])
    U[ncells + 3] = primitive_to_conserved(law, w1_ghost)
    U[ncells + 4] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

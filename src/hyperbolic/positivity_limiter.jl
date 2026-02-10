# ============================================================
# Positivity-Preserving Limiter (Zhang & Shu 2010)
# ============================================================
#
# Post-processing limiter that ensures density and pressure
# remain non-negative after high-order reconstruction.
#
# Reference: X. Zhang and C.-W. Shu, "On positivity-preserving
# high order discontinuous Galerkin schemes for compressible
# Euler equations on rectangular meshes", J. Comput. Phys., 2010.

"""
    PositivityLimiter{FT}

Positivity-preserving limiter following Zhang & Shu (2010).

Applied as a post-processing step after high-order reconstruction
to ensure density and pressure remain non-negative. Uses a
theta-blending strategy that mixes reconstructed face values back
toward the cell average when positivity is violated.

# Fields
- `epsilon::FT`: Floor value for density and pressure (default `1e-13`).
"""
struct PositivityLimiter{FT}
    epsilon::FT
end

PositivityLimiter() = PositivityLimiter(1.0e-13)

# ============================================================
# 1D cell-based positivity enforcement
# ============================================================

"""
    apply_positivity_limiter!(U, law, limiter::PositivityLimiter, nc)

Apply positivity enforcement to all interior cells of a 1D solution array.

For each interior cell `i` (indices `3:nc+2` in the ghost-padded array):
1. If density `ρ < epsilon`, set density to `epsilon`.
2. If pressure `P < epsilon`, scale the velocity to reduce kinetic energy
   and bring the pressure up to `epsilon`.

This modifies `U` in-place.
"""
function apply_positivity_limiter!(
        U::AbstractVector, law::AbstractConservationLaw{1},
        limiter::PositivityLimiter, nc::Int
    )
    eps = limiter.epsilon
    nvar = nvariables(law)
    @inbounds for i in 3:(nc + 2)
        u = U[i]
        rho = u[1]

        # Step 1: enforce density floor
        if rho < eps
            rho = eps
            u = SVector{nvar}(ntuple(k -> k == 1 ? rho : u[k], Val(nvar)))
        end

        # Step 2: check pressure and enforce if needed
        w = conserved_to_primitive(law, u)
        P = w[nvar]  # pressure is the last primitive variable

        if P < eps
            # Reduce kinetic energy to bring pressure up.
            # For 1D Euler: u = [rho, rho*v, E]
            # P = (gamma-1)*(E - 0.5*rho*v^2)
            # We want P_new = eps, so E_new - 0.5*rho*v_new^2 = eps/(gamma-1)
            # Strategy: scale momentum by theta so that
            #   E - 0.5*rho*(theta*v)^2 >= eps/(gamma-1)
            # Solve: theta^2 = (E - eps/(gamma-1)) / (0.5*rho*v^2)
            rho_v = u[2]
            E = u[nvar]
            KE = 0.5 * rho_v^2 / rho
            # Internal energy required for pressure = eps
            w_target = conserved_to_primitive(law, SVector{nvar}(ntuple(k -> k == 1 ? rho : (k == nvar ? E : zero(E)), Val(nvar))))
            P_zero_vel = w_target[nvar]
            if P_zero_vel < eps
                # Even with zero velocity, pressure is too low.
                # Set velocity to zero and reconstruct total energy for P = eps.
                e_int = internal_energy(law.eos, rho, eps)
                E_new = rho * e_int
                u = SVector{nvar}(ntuple(k -> k == 1 ? rho : (k == nvar ? E_new : zero(E_new)), Val(nvar)))
            else
                # Scale velocity to achieve P = eps
                if KE > zero(KE)
                    e_int_needed = internal_energy(law.eos, rho, eps)
                    E_int_needed = rho * e_int_needed
                    theta_sq = (E - E_int_needed) / KE
                    theta_sq = clamp(theta_sq, zero(theta_sq), one(theta_sq))
                    theta = sqrt(theta_sq)
                    # Scale momentum components (indices 2 to nvar-1)
                    u = SVector{nvar}(ntuple(k -> k == 1 ? rho : (k == nvar ? E : theta * u[k]), Val(nvar)))
                end
            end
        end

        U[i] = u
    end
    return nothing
end

# ============================================================
# 2D cell-based positivity enforcement
# ============================================================

"""
    apply_positivity_limiter_2d!(U, law, limiter::PositivityLimiter, nx, ny)

Apply positivity enforcement to all interior cells of a 2D solution array.

For each interior cell `(ix, iy)` at indices `(ix+2, iy+2)`:
1. If density `ρ < epsilon`, set density to `epsilon`.
2. If pressure `P < epsilon`, scale the velocity to reduce kinetic energy
   and bring the pressure up to `epsilon`.

This modifies `U` in-place.
"""
function apply_positivity_limiter_2d!(
        U::AbstractMatrix, law::AbstractConservationLaw{2},
        limiter::PositivityLimiter, nx::Int, ny::Int
    )
    eps = limiter.epsilon
    nvar = nvariables(law)
    @inbounds for iy in 1:ny
        jj = iy + 2
        for ix in 1:nx
            ii = ix + 2
            u = U[ii, jj]
            rho = u[1]

            # Step 1: enforce density floor
            if rho < eps
                rho = eps
                u = SVector{nvar}(ntuple(k -> k == 1 ? rho : u[k], Val(nvar)))
            end

            # Step 2: check pressure and enforce if needed
            w = conserved_to_primitive(law, u)
            P = w[nvar]  # pressure is the last primitive variable

            if P < eps
                # Scale momentum to bring pressure up.
                # Momentum components are indices 2:(nvar-1), energy is index nvar.
                E = u[nvar]
                KE = zero(E)
                for m in 2:(nvar - 1)
                    KE += 0.5 * u[m]^2 / rho
                end
                # Check pressure at zero velocity
                u_zero = SVector{nvar}(ntuple(k -> k == 1 ? rho : (k == nvar ? E : zero(E)), Val(nvar)))
                w_zero = conserved_to_primitive(law, u_zero)
                P_zero_vel = w_zero[nvar]
                if P_zero_vel < eps
                    # Even with zero velocity, pressure is too low.
                    e_int = internal_energy(law.eos, rho, eps)
                    E_new = rho * e_int
                    u = SVector{nvar}(ntuple(k -> k == 1 ? rho : (k == nvar ? E_new : zero(E_new)), Val(nvar)))
                else
                    # Scale velocity to achieve P = eps
                    if KE > zero(KE)
                        e_int_needed = internal_energy(law.eos, rho, eps)
                        E_int_needed = rho * e_int_needed
                        theta_sq = (E - E_int_needed) / KE
                        theta_sq = clamp(theta_sq, zero(theta_sq), one(theta_sq))
                        theta = sqrt(theta_sq)
                        u = SVector{nvar}(ntuple(k -> k == 1 ? rho : (k == nvar ? E : theta * u[k]), Val(nvar)))
                    end
                end
            end

            U[ii, jj] = u
        end
    end
    return nothing
end

# ============================================================
# Reconstructed-state theta-limiting (Zhang & Shu)
# ============================================================

"""
    limit_reconstructed_states(limiter::PositivityLimiter, law, wL, wR, wL_face, wR_face)
        -> (wL_face_limited, wR_face_limited)

Apply Zhang & Shu theta-limiting to reconstructed face states to ensure
positivity of density and pressure.

Given cell-average primitive states `wL`, `wR` and reconstructed face
states `wL_face`, `wR_face`, blend each face state back toward its
cell average using the minimum theta that maintains:
1. `rho_face >= epsilon`
2. `P_face >= epsilon`

# Arguments
- `limiter::PositivityLimiter`: The limiter with floor value `epsilon`.
- `law`: Conservation law (provides `conserved_to_primitive`, `primitive_to_conserved`).
- `wL`: Cell-average primitive state for the left cell.
- `wR`: Cell-average primitive state for the right cell.
- `wL_face`: Reconstructed primitive state at the face from the left cell.
- `wR_face`: Reconstructed primitive state at the face from the right cell.

# Returns
- `(wL_face_limited, wR_face_limited)`: Positivity-corrected face states.
"""
function limit_reconstructed_states(
        limiter::PositivityLimiter, law::AbstractConservationLaw,
        wL, wR, wL_face, wR_face
    )
    eps = limiter.epsilon
    wL_face_limited = _limit_face_state(law, wL, wL_face, eps)
    wR_face_limited = _limit_face_state(law, wR, wR_face, eps)
    return wL_face_limited, wR_face_limited
end

"""
    _limit_face_state(law, w_cell, w_face, eps) -> w_face_limited

Blend `w_face` toward `w_cell` to enforce positivity of density and pressure.

Step 1: Ensure density positivity.
Step 2: Ensure pressure positivity (using the blended state from step 1).
"""
@inline function _limit_face_state(law::AbstractConservationLaw, w_cell, w_face, eps)
    nvar = nvariables(law)

    # --- Step 1: density positivity ---
    rho_cell = w_cell[1]
    rho_face = w_face[1]

    if rho_face < eps
        if rho_cell > eps
            # Blend toward cell average
            theta_rho = (rho_cell - eps) / (rho_cell - rho_face)
            theta_rho = clamp(theta_rho, zero(theta_rho), one(theta_rho))
        else
            # Cell average itself is at or below floor; snap to cell average
            theta_rho = zero(rho_cell)
        end
        w_face = theta_rho * w_face + (one(theta_rho) - theta_rho) * w_cell
    end

    # --- Step 2: pressure positivity ---
    # Pressure is the last entry of the primitive variable vector
    P_face = w_face[nvar]

    if P_face < eps
        P_cell = w_cell[nvar]
        if P_cell > eps
            # Binary search / bisection for theta that gives P = eps.
            # Use a simple Newton-bisection approach.
            # Since pressure is a concave function of theta for Euler-type EOS,
            # the linear estimate gives a good starting point.
            theta_P = _find_pressure_theta(law, w_cell, w_face, eps)
            w_face = theta_P * w_face + (one(theta_P) - theta_P) * w_cell
        else
            # Cell average itself has non-positive pressure; snap to cell average
            w_face = w_cell
        end
    end

    return w_face
end

"""
    _find_pressure_theta(law, w_cell, w_face, eps) -> theta

Find the largest theta in [0, 1] such that the pressure of
`theta * w_face + (1 - theta) * w_cell` is >= eps.

Uses bisection for robustness (pressure is not always linear in theta
for general EOS, though it is for ideal gas in primitive variables).
"""
@inline function _find_pressure_theta(law::AbstractConservationLaw, w_cell, w_face, eps)
    # For ideal gas in primitive variables, P is linear in theta,
    # so one step suffices. But for general EOS we use bisection.
    nvar = nvariables(law)

    # Quick linear estimate (exact for ideal gas in primitive vars)
    P_cell = w_cell[nvar]
    P_face = w_face[nvar]
    if abs(P_cell - P_face) > zero(P_cell)
        theta_est = (P_cell - eps) / (P_cell - P_face)
        theta_est = clamp(theta_est, zero(theta_est), one(theta_est))
    else
        theta_est = zero(P_cell)
    end

    # Verify the estimate works by checking pressure of blended state
    w_blend = theta_est * w_face + (one(theta_est) - theta_est) * w_cell
    u_blend = primitive_to_conserved(law, w_blend)
    w_check = conserved_to_primitive(law, u_blend)
    P_check = w_check[nvar]

    if P_check >= eps
        return theta_est
    end

    # Fall back to bisection if linear estimate was insufficient
    theta_lo = zero(theta_est)
    theta_hi = theta_est
    for _ in 1:20
        theta_mid = 0.5 * (theta_lo + theta_hi)
        w_mid = theta_mid * w_face + (one(theta_mid) - theta_mid) * w_cell
        u_mid = primitive_to_conserved(law, w_mid)
        w_mid_check = conserved_to_primitive(law, u_mid)
        P_mid = w_mid_check[nvar]
        if P_mid >= eps
            theta_lo = theta_mid
        else
            theta_hi = theta_mid
        end
    end
    return theta_lo
end

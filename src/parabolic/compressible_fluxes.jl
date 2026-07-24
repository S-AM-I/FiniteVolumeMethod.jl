# Compressible flow fluxes
# Migrated from Simu.jl SimuFVM/compressible_fluxes.jl

"""
    ideal_gas_pressure(rho, E, u, gamma)

Compute pressure for 1D ideal gas.
E is total energy per unit mass.
"""
function ideal_gas_pressure(rho, E, u, gamma)
    e = E - 0.5 * u^2
    return rho * (gamma - 1.0) * e
end

"""
    hllc_flux_1d(U_L, U_R, gamma)

Compute HLLC flux for 1D Euler equations.
U = [rho, rho*u, rho*E]
Returns flux vector F.
"""
function hllc_flux_1d(U_L::Vector{Float64}, U_R::Vector{Float64}, gamma::Float64)
    # Primitive variables
    rho_L = U_L[1]
    rhou_L = U_L[2]
    rhoE_L = U_L[3]
    u_L = rhou_L / rho_L
    p_L = ideal_gas_pressure(rho_L, rhoE_L / rho_L, u_L, gamma)
    a_L = sound_speed(IdealGasEOS(gamma), rho_L, p_L)
    H_L = (rhoE_L + p_L) / rho_L

    rho_R = U_R[1]
    rhou_R = U_R[2]
    rhoE_R = U_R[3]
    u_R = rhou_R / rho_R
    p_R = ideal_gas_pressure(rho_R, rhoE_R / rho_R, u_R, gamma)
    a_R = sound_speed(IdealGasEOS(gamma), rho_R, p_R)
    H_R = (rhoE_R + p_R) / rho_R

    # Roe-averaged state
    sq_rho_L = sqrt(rho_L)
    sq_rho_R = sqrt(rho_R)
    denom = sq_rho_L + sq_rho_R

    u_roe = (sq_rho_L * u_L + sq_rho_R * u_R) / denom
    H_roe = (sq_rho_L * H_L + sq_rho_R * H_R) / denom
    a_roe = sqrt((gamma - 1.0) * (H_roe - 0.5 * u_roe^2))

    # Wave speeds
    S_L = min(u_L - a_L, u_roe - a_roe)
    S_R = max(u_R + a_R, u_roe + a_roe)

    # Fluxes
    F_L = [rhou_L, rhou_L * u_L + p_L, u_L * (rhoE_L + p_L)]
    F_R = [rhou_R, rhou_R * u_R + p_R, u_R * (rhoE_R + p_R)]

    if S_L >= 0.0
        return F_L
    elseif S_R <= 0.0
        return F_R
    else
        # Intermediate region
        S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) /
            (rho_L * (S_L - u_L) - rho_R * (S_R - u_R))

        if S_L <= 0.0 && S_star >= 0.0
            factor = rho_L * (S_L - u_L) / (S_L - S_star)
            U_star_L = factor * [1.0, S_star, rhoE_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L)))]
            return F_L + S_L * (U_star_L - U_L)
        else # S_star <= 0.0 && S_R >= 0.0
            factor = rho_R * (S_R - u_R) / (S_R - S_star)
            U_star_R = factor * [1.0, S_star, rhoE_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R)))]
            return F_R + S_R * (U_star_R - U_R)
        end
    end
end

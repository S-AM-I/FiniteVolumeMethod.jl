# ============================================================
# Variable names for each conservation law
# ============================================================
#
# Moved here from src/dashboard_types.jl in Stage 3e: every method dispatches
# on a conservation-law type owned by this module. The generic is consumed by
# the flat parent (dashboard_types.jl, core/symbolic_indexing.jl) and by
# FVMRecipesExt through the FiniteVolumeMethod.variable_names binding.

"""
    variable_names(law) -> Vector{String}

Return human-readable names for the conserved variables of `law`.
"""
function variable_names end

variable_names(::EulerEquations{1}) = ["rho", "rho_v", "E"]
variable_names(::EulerEquations{2}) = ["rho", "rho_vx", "rho_vy", "E"]
variable_names(::EulerEquations{3}) = ["rho", "rho_vx", "rho_vy", "rho_vz", "E"]
variable_names(::IdealMHDEquations) = ["rho", "rho_vx", "rho_vy", "rho_vz", "E", "Bx", "By", "Bz"]
variable_names(::NavierStokesEquations{1}) = ["rho", "rho_v", "E"]
variable_names(::NavierStokesEquations{2}) = ["rho", "rho_vx", "rho_vy", "E"]
variable_names(::SRMHDEquations) = ["D", "Sx", "Sy", "Sz", "tau", "Bx", "By", "Bz"]
variable_names(::GRMHDEquations) = ["D", "Sx", "Sy", "Sz", "tau", "Bx", "By", "Bz"]
variable_names(::ShallowWaterEquations{1}) = ["h", "hv"]
variable_names(::ShallowWaterEquations{2}) = ["h", "hvx", "hvy"]
variable_names(::SRHydroEquations{1}) = ["D", "Sx", "tau"]
variable_names(::SRHydroEquations{2}) = ["D", "Sx", "Sy", "tau"]
variable_names(::TwoFluidEquations{1}) = ["rho_i", "rho_i_v", "E_i", "rho_e", "rho_e_v", "E_e"]
variable_names(::TwoFluidEquations{2}) = ["rho_i", "rho_i_vx", "rho_i_vy", "E_i", "rho_e", "rho_e_vx", "rho_e_vy", "E_e"]
variable_names(::ResistiveMHDEquations) = ["rho", "rho_vx", "rho_vy", "rho_vz", "E", "Bx", "By", "Bz"]
variable_names(::HallMHDEquations) = ["rho", "rho_vx", "rho_vy", "rho_vz", "E", "Bx", "By", "Bz"]

function variable_names(law::ReactiveEulerEquations{1, NS}) where {NS}
    base = ["rho", "rho_v", "E"]
    for name in law.species_names
        push!(base, "rho_Y_$(name)")
    end
    return base
end

function variable_names(law::ReactiveEulerEquations{2, NS}) where {NS}
    base = ["rho", "rho_vx", "rho_vy", "E"]
    for name in law.species_names
        push!(base, "rho_Y_$(name)")
    end
    return base
end

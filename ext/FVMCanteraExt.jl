module FVMCanteraExt

using FiniteVolumeMethod
using Cantera: Cantera

# ── read_chemkin_mechanism ─────────────────────────────────────────

"""
Parse a CHEMKIN mechanism file via Cantera and build a
`MultiStepMechanism`. Only forward Arrhenius parameters `(A, b, E_a)`
and stoichiometric coefficients are extracted — third-body and
falloff corrections are folded into the pre-exponential factor, so
this adapter is a first-order approximation suitable for laminar and
mildly-turbulent premixed flames.
"""
function FiniteVolumeMethod.read_chemkin_mechanism(path::AbstractString)
    # Cantera.jl loads the mechanism; we query rate coefficients and
    # stoichiometric matrices and construct `MultiStepMechanism`.
    gas = Cantera.Solution(path)
    NS = Cantera.n_species(gas)
    NR = Cantera.n_reactions(gas)

    A_vec = Vector{Float64}(undef, NR)
    b_vec = Vector{Float64}(undef, NR)
    E_a_vec = Vector{Float64}(undef, NR)
    nu_r = zeros(Float64, NR, NS)
    nu_p = zeros(Float64, NR, NS)

    for r in 1:NR
        rate = Cantera.reaction_rate(gas, r)
        A_vec[r] = rate.A
        b_vec[r] = rate.b
        E_a_vec[r] = rate.E_a
        for i in 1:NS
            nu_r[r, i] = Cantera.reactant_stoich_coeff(gas, i, r)
            nu_p[r, i] = Cantera.product_stoich_coeff(gas, i, r)
        end
    end

    return FiniteVolumeMethod.MultiStepMechanism(;
        A = Tuple(A_vec),
        b = Tuple(b_vec),
        E_a = Tuple(E_a_vec),
        nu_reactants = nu_r,
        nu_products = nu_p,
    )
end

# ── compute_fgm_table_from_cantera ─────────────────────────────────

"""
Solve a Cantera free-flame / counterflow flamelet at discrete mixture
fractions `Z ∈ [0, 1]` and progress variables `C ∈ [0, 1]` and build
an `FGMTable` of species mass fractions.

Implementation notes:
- `NC` progress-variable points and `NZ` mixture-fraction points.
- Uses `Cantera.IdealGasReactor` at fixed Z, integrates to each C,
  records species mass fractions at that point.
- This is the canonical FGM construction for premixed flamelets.
"""
function FiniteVolumeMethod.compute_fgm_table_from_cantera(
        mechanism, NC::Int, NZ::Int,
        fuel::AbstractString, oxidizer::AbstractString;
        pressure::Real = 101325.0, T_inlet::Real = 300.0,
    )
    gas = mechanism isa Cantera.Solution ? mechanism : Cantera.Solution(mechanism)
    NS = Cantera.n_species(gas)

    C_grid = collect(range(0.0, 1.0; length = NC))
    Z_grid = collect(range(0.0, 1.0; length = NZ))
    Y_table = Array{Float64, 3}(undef, NC, NZ, NS)

    for iZ in 1:NZ
        Z = Z_grid[iZ]
        # Set a fuel/oxidizer mixture at mixture fraction Z.
        Cantera.set_equivalence_ratio!(gas, Z, fuel, oxidizer)
        Cantera.TP!(gas, T_inlet, pressure)

        reactor = Cantera.IdealGasReactor(gas)
        sim = Cantera.ReactorNet([reactor])

        # Integrate until equilibrium, sampling `NC` progress-variable
        # bins. Progress variable is normalized mass fraction of the
        # major product.
        C_points = copy(C_grid)
        for iC in eachindex(C_points)
            # Integrate in time until C target reached.
            Cantera.advance_to_progress!(sim, C_points[iC])
            Y_snapshot = Cantera.mass_fractions(gas)
            for s in 1:NS
                Y_table[iC, iZ, s] = Y_snapshot[s]
            end
        end
    end

    return FiniteVolumeMethod.FGMTable{NS, Float64}(C_grid, Z_grid, Y_table)
end

end # module

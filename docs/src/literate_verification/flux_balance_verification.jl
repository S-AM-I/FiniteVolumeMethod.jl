using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Flux Balance (Conservation) Verification
# This example verifies that the cell-centred FVM solver conserves mass,
# momentum, and energy to machine precision when using periodic boundary
# conditions — independently for Euler, MHD, and Navier-Stokes equations.
#
# ## Mathematical Setup
# For each conservation law, we initialise a smooth state on a periodic
# domain and evolve for a fixed number of time steps. The total of each
# conserved variable (integrated over the domain) must remain constant
# to within round-off error.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)

# ============================================================
# 1. Euler Equations (1D)
# ============================================================
law_euler = EulerEquations{1}(eos)
N_euler = 100
dx_euler = 1.0 / N_euler
mesh_euler = StructuredMesh1D(0.0, 1.0, N_euler)

euler_ic(x) = SVector(1.0 + 0.2 * sin(2 * pi * x), 0.3, 1.0)
prob_euler = HyperbolicProblem(
    law_euler, mesh_euler, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), euler_ic;
    final_time = 0.2, cfl = 0.4,
)
x_e, U_e_init = let
    mesh = mesh_euler
    N = N_euler
    dx = mesh.dx
    xs = [mesh.xmin + (i - 0.5) * dx for i in 1:N]
    Us = [primitive_to_conserved(law_euler, euler_ic(x)) for x in xs]
    xs, Us
end
x_e_sol, U_e_sol, t_e = solve_hyperbolic(prob_euler)

mass_0_e = sum(u[1] for u in U_e_init) * dx_euler
mom_0_e = sum(u[2] for u in U_e_init) * dx_euler
energy_0_e = sum(u[3] for u in U_e_init) * dx_euler

mass_f_e = sum(u[1] for u in U_e_sol) * dx_euler
mom_f_e = sum(u[2] for u in U_e_sol) * dx_euler
energy_f_e = sum(u[3] for u in U_e_sol) * dx_euler

euler_drift_mass = abs(mass_f_e - mass_0_e) / abs(mass_0_e)
euler_drift_mom = abs(mom_f_e - mom_0_e) / max(abs(mom_0_e), 1.0e-15)
euler_drift_energy = abs(energy_f_e - energy_0_e) / abs(energy_0_e)

# ============================================================
# 2. MHD Equations (2D)
# ============================================================
law_mhd = IdealMHDEquations{2}(IdealGasEOS(5.0 / 3.0))
N_mhd = 16
mesh_mhd = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N_mhd, N_mhd)
dx_mhd = mesh_mhd.dx * mesh_mhd.dy

function mhd_ic(x, y)
    rho = 1.0 + 0.1 * sin(2 * pi * x) * cos(2 * pi * y)
    return SVector(rho, 0.2, 0.1, 0.0, 1.0, 0.5, 0.3, 0.0)
end

prob_mhd = HyperbolicProblem2D(
    law_mhd, mesh_mhd, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    mhd_ic; final_time = 0.05, cfl = 0.3,
)

# Initial totals
mass_0_mhd = let
    total = 0.0
    for j in 1:N_mhd, i in 1:N_mhd
        x = mesh_mhd.xmin + (i - 0.5) * mesh_mhd.dx
        y = mesh_mhd.ymin + (j - 0.5) * mesh_mhd.dy
        w = mhd_ic(x, y)
        u = primitive_to_conserved(law_mhd, w)
        total += u[1] * dx_mhd
    end
    total
end

coords_mhd, U_mhd, t_mhd = solve_hyperbolic(prob_mhd)
mass_f_mhd = sum(U_mhd[i, j][1] for j in 1:N_mhd, i in 1:N_mhd) * dx_mhd
mhd_drift_mass = abs(mass_f_mhd - mass_0_mhd) / abs(mass_0_mhd)

# ============================================================
# 3. Navier-Stokes (1D)
# ============================================================
law_ns = NavierStokesEquations{1}(eos, mu = 0.01, Pr = 0.72)
N_ns = 64
dx_ns = 1.0 / N_ns
mesh_ns = StructuredMesh1D(0.0, 1.0, N_ns)

ns_ic(x) = SVector(1.0 + 0.1 * sin(2 * pi * x), 0.1 * cos(2 * pi * x), 10.0)
prob_ns = HyperbolicProblem(
    law_ns, mesh_ns, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), ns_ic;
    final_time = 0.1, cfl = 0.3,
)

ns_init = [primitive_to_conserved(law_ns, ns_ic(mesh_ns.xmin + (i - 0.5) * dx_ns)) for i in 1:N_ns]
mass_0_ns = sum(u[1] for u in ns_init) * dx_ns

x_ns, U_ns, t_ns = solve_hyperbolic(prob_ns)
mass_f_ns = sum(u[1] for u in U_ns) * dx_ns
ns_drift_mass = abs(mass_f_ns - mass_0_ns) / abs(mass_0_ns)

# ============================================================
# Visualisation — Conservation Drift Summary
# ============================================================
fig = Figure(fontsize = 24, size = (900, 450))

labels = ["Euler mass", "Euler mom", "Euler energy", "MHD mass", "NS mass"]
drifts = [euler_drift_mass, euler_drift_mom, euler_drift_energy, mhd_drift_mass, ns_drift_mass]
colors_bar = [:blue, :cyan, :green, :orange, :red]
eps_floor = 1.0e-16
drifts_plot = max.(drifts, eps_floor)

ax = Axis(
    fig[1, 1], ylabel = "Relative conservation drift",
    yscale = log10, title = "Conservation Verification (periodic BCs)",
    xticks = (1:5, labels),
    xticklabelrotation = pi / 6,
)
barplot!(ax, 1:5, drifts_plot, color = colors_bar)
hlines!(ax, [eps(Float64)], color = :red, linestyle = :dash, linewidth = 1.5, label = "Machine ε")
axislegend(ax, position = :rt)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "flux_balance_verification.png") fig #src

# ## Test Assertions
# All conserved quantities should be preserved to within $10^{-10}$
# relative error.
@test euler_drift_mass < 1.0e-10 #src
@test euler_drift_energy < 1.0e-10 #src
@test mhd_drift_mass < 1.0e-10 #src
@test ns_drift_mass < 1.0e-10 #src
@assert euler_drift_mass < 1.0e-10 #hide
@assert euler_drift_energy < 1.0e-10 #hide
@assert mhd_drift_mass < 1.0e-10 #hide
@assert ns_drift_mass < 1.0e-10 #hide

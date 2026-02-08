# ============================================================
# 1D SRMHD Solver Specialization
# ============================================================
#
# The existing solve_hyperbolic(prob::HyperbolicProblem) works
# automatically for SRMHDEquations{1} since it dispatches through
# physical_flux, max_wave_speed, conserved_to_primitive, etc.
#
# No additional specialization needed — the generic 1D solver
# handles SRMHD through the AbstractConservationLaw interface.
#
# This file is intentionally minimal: the only SRMHD-specific
# behavior is the con2prim iteration inside conserved_to_primitive.

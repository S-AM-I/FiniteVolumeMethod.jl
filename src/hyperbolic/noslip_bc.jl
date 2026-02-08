# ============================================================
# No-Slip Wall Boundary Condition
# ============================================================

"""
    NoSlipBC <: AbstractHyperbolicBC

No-slip wall boundary condition. ALL velocity components are negated
in the ghost cells, placing zero velocity at the wall face midpoint.

Unlike `ReflectiveBC` (slip wall, which only negates the wall-normal velocity),
`NoSlipBC` enforces zero tangential velocity as well, appropriate for
viscous (Navier-Stokes) flows.
"""
struct NoSlipBC <: AbstractHyperbolicBC end

# ============================================================
# 1D NoSlipBC (NavierStokesEquations{1})
# ============================================================

function apply_bc_left!(U::AbstractVector, ::NoSlipBC, law::NavierStokesEquations{1}, ncells::Int, t)
    u1 = U[3]
    u2 = U[4]
    w1 = conserved_to_primitive(law, u1)
    w2 = conserved_to_primitive(law, u2)
    # Negate velocity
    w1_ghost = SVector(w1[1], -w1[2], w1[3])
    w2_ghost = SVector(w2[1], -w2[2], w2[3])
    U[2] = primitive_to_conserved(law, w1_ghost)
    U[1] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::NoSlipBC, law::NavierStokesEquations{1}, ncells::Int, t)
    u1 = U[ncells + 2]
    u2 = U[ncells + 1]
    w1 = conserved_to_primitive(law, u1)
    w2 = conserved_to_primitive(law, u2)
    w1_ghost = SVector(w1[1], -w1[2], w1[3])
    w2_ghost = SVector(w2[1], -w2[2], w2[3])
    U[ncells + 3] = primitive_to_conserved(law, w1_ghost)
    U[ncells + 4] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

# ============================================================
# 2D NoSlipBC (NavierStokesEquations{2})
# ============================================================

# Left wall: negate both vx and vy
function apply_bc_2d_left!(U::AbstractMatrix, ::NoSlipBC, law::NavierStokesEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[3, j])
        w2 = conserved_to_primitive(law, U[4, j])
        U[2, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], -w1[3], w1[4]))
        U[1, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], -w2[3], w2[4]))
    end
    return nothing
end

# Right wall: negate both vx and vy
function apply_bc_2d_right!(U::AbstractMatrix, ::NoSlipBC, law::NavierStokesEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[nx + 2, j])
        w2 = conserved_to_primitive(law, U[nx + 1, j])
        U[nx + 3, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], -w1[3], w1[4]))
        U[nx + 4, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], -w2[3], w2[4]))
    end
    return nothing
end

# Bottom wall: negate both vx and vy
function apply_bc_2d_bottom!(U::AbstractMatrix, ::NoSlipBC, law::NavierStokesEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, 3])
        w2 = conserved_to_primitive(law, U[i, 4])
        U[i, 2] = primitive_to_conserved(law, SVector(w1[1], -w1[2], -w1[3], w1[4]))
        U[i, 1] = primitive_to_conserved(law, SVector(w2[1], -w2[2], -w2[3], w2[4]))
    end
    return nothing
end

# Top wall: negate both vx and vy
function apply_bc_2d_top!(U::AbstractMatrix, ::NoSlipBC, law::NavierStokesEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, ny + 2])
        w2 = conserved_to_primitive(law, U[i, ny + 1])
        U[i, ny + 3] = primitive_to_conserved(law, SVector(w1[1], -w1[2], -w1[3], w1[4]))
        U[i, ny + 4] = primitive_to_conserved(law, SVector(w2[1], -w2[2], -w2[3], w2[4]))
    end
    return nothing
end

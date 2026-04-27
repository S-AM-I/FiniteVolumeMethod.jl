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

function apply_bc_left!(U::AbstractVector, ::NoSlipBC, law::NavierStokesEquations{1}, ncells::Int, ng::Int, t)
    for g in 1:ng
        u_int = U[ng + g]
        w = conserved_to_primitive(law, u_int)
        # Negate velocity
        w_ghost = SVector(w[1], -w[2], w[3])
        U[ng + 1 - g] = primitive_to_conserved(law, w_ghost)
    end
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::NoSlipBC, law::NavierStokesEquations{1}, ncells::Int, ng::Int, t)
    last_interior = ncells + ng
    for g in 1:ng
        u_int = U[last_interior + 1 - g]
        w = conserved_to_primitive(law, u_int)
        w_ghost = SVector(w[1], -w[2], w[3])
        U[last_interior + g] = primitive_to_conserved(law, w_ghost)
    end
    return nothing
end

# ============================================================
# 2D NoSlipBC (NavierStokesEquations{2})
# ============================================================

# Left wall: negate both vx and vy
function apply_bc_2d_left!(U::AbstractMatrix, ::NoSlipBC, law::NavierStokesEquations{2}, nx::Int, ny::Int, ng::Int, t)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[ng + g, j])
            U[ng + 1 - g, j] = primitive_to_conserved(law, SVector(w[1], -w[2], -w[3], w[4]))
        end
    end
    return nothing
end

# Right wall: negate both vx and vy
function apply_bc_2d_right!(U::AbstractMatrix, ::NoSlipBC, law::NavierStokesEquations{2}, nx::Int, ny::Int, ng::Int, t)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[nx + ng + 1 - g, j])
            U[nx + ng + g, j] = primitive_to_conserved(law, SVector(w[1], -w[2], -w[3], w[4]))
        end
    end
    return nothing
end

# Bottom wall: negate both vx and vy
function apply_bc_2d_bottom!(U::AbstractMatrix, ::NoSlipBC, law::NavierStokesEquations{2}, nx::Int, ny::Int, ng::Int, t)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, ng + g])
            U[i, ng + 1 - g] = primitive_to_conserved(law, SVector(w[1], -w[2], -w[3], w[4]))
        end
    end
    return nothing
end

# Top wall: negate both vx and vy
function apply_bc_2d_top!(U::AbstractMatrix, ::NoSlipBC, law::NavierStokesEquations{2}, nx::Int, ny::Int, ng::Int, t)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, ny + ng + 1 - g])
            U[i, ny + ng + g] = primitive_to_conserved(law, SVector(w[1], -w[2], -w[3], w[4]))
        end
    end
    return nothing
end

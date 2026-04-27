# ============================================================
# 3D Ghost-Cell Boundary Conditions
# ============================================================
#
# The 3D solution is stored in a padded 3D array U[i, j, k] where:
#   i = 1:ng              -> left ghost planes   (x-)
#   i = ng+1:nx+ng        -> interior            (x)
#   i = nx+ng+1:nx+2*ng   -> right ghost planes  (x+)
#   j = 1:ng              -> bottom ghost planes (y-)
#   j = ng+1:ny+ng        -> interior            (y)
#   j = ny+ng+1:ny+2*ng   -> top ghost planes    (y+)
#   k = 1:ng              -> front ghost planes  (z-)
#   k = ng+1:nz+ng        -> interior            (z)
#   k = nz+ng+1:nz+2*ng   -> back ghost planes   (z+)
#
# Interior cell (ix, iy, iz) (1-based) maps to U[ix+ng, iy+ng, iz+ng].

# ============================================================
# TransmissiveBC (3D)
# ============================================================

function apply_bc_3d_left!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[ng + 1 - g, j, k] = U[ng + 1, j, k]
        end
    end
    return nothing
end

function apply_bc_3d_right!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[nx + ng + g, j, k] = U[nx + ng, j, k]
        end
    end
    return nothing
end

function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ng + 1 - g, k] = U[i, ng + 1, k]
        end
    end
    return nothing
end

function apply_bc_3d_top!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ny + ng + g, k] = U[i, ny + ng, k]
        end
    end
    return nothing
end

function apply_bc_3d_front!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, j, ng + 1 - g] = U[i, j, ng + 1]
        end
    end
    return nothing
end

function apply_bc_3d_back!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, j, nz + ng + g] = U[i, j, nz + ng]
        end
    end
    return nothing
end

# ============================================================
# ReflectiveBC (3D Euler)
# ============================================================

# Left wall: negate vx (component 2 in 5-var Euler)
function apply_bc_3d_left!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[ng + g, j, k])
            U[ng + 1 - g, j, k] = primitive_to_conserved(law, SVector(w[1], -w[2], w[3], w[4], w[5]))
        end
    end
    return nothing
end

# Right wall: negate vx
function apply_bc_3d_right!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[nx + ng + 1 - g, j, k])
            U[nx + ng + g, j, k] = primitive_to_conserved(law, SVector(w[1], -w[2], w[3], w[4], w[5]))
        end
    end
    return nothing
end

# Bottom wall: negate vy (component 3)
function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, ng + g, k])
            U[i, ng + 1 - g, k] = primitive_to_conserved(law, SVector(w[1], w[2], -w[3], w[4], w[5]))
        end
    end
    return nothing
end

# Top wall: negate vy
function apply_bc_3d_top!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, ny + ng + 1 - g, k])
            U[i, ny + ng + g, k] = primitive_to_conserved(law, SVector(w[1], w[2], -w[3], w[4], w[5]))
        end
    end
    return nothing
end

# Front wall: negate vz (component 4)
function apply_bc_3d_front!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, j, ng + g])
            U[i, j, ng + 1 - g] = primitive_to_conserved(law, SVector(w[1], w[2], w[3], -w[4], w[5]))
        end
    end
    return nothing
end

# Back wall: negate vz
function apply_bc_3d_back!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, j, nz + ng + 1 - g])
            U[i, j, nz + ng + g] = primitive_to_conserved(law, SVector(w[1], w[2], w[3], -w[4], w[5]))
        end
    end
    return nothing
end

# ============================================================
# ReflectiveBC (3D MHD)
# ============================================================

# Left wall: negate vx (index 2)
function apply_bc_3d_left!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[ng + g, j, k])
            U[ng + 1 - g, j, k] = primitive_to_conserved(law, SVector(w[1], -w[2], w[3], w[4], w[5], w[6], w[7], w[8]))
        end
    end
    return nothing
end

# Right wall: negate vx
function apply_bc_3d_right!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[nx + ng + 1 - g, j, k])
            U[nx + ng + g, j, k] = primitive_to_conserved(law, SVector(w[1], -w[2], w[3], w[4], w[5], w[6], w[7], w[8]))
        end
    end
    return nothing
end

# Bottom wall: negate vy (index 3)
function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, ng + g, k])
            U[i, ng + 1 - g, k] = primitive_to_conserved(law, SVector(w[1], w[2], -w[3], w[4], w[5], w[6], w[7], w[8]))
        end
    end
    return nothing
end

# Top wall: negate vy
function apply_bc_3d_top!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, ny + ng + 1 - g, k])
            U[i, ny + ng + g, k] = primitive_to_conserved(law, SVector(w[1], w[2], -w[3], w[4], w[5], w[6], w[7], w[8]))
        end
    end
    return nothing
end

# Front wall: negate vz (index 4)
function apply_bc_3d_front!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, j, ng + g])
            U[i, j, ng + 1 - g] = primitive_to_conserved(law, SVector(w[1], w[2], w[3], -w[4], w[5], w[6], w[7], w[8]))
        end
    end
    return nothing
end

# Back wall: negate vz
function apply_bc_3d_back!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, j, nz + ng + 1 - g])
            U[i, j, nz + ng + g] = primitive_to_conserved(law, SVector(w[1], w[2], w[3], -w[4], w[5], w[6], w[7], w[8]))
        end
    end
    return nothing
end

# ============================================================
# DirichletHyperbolicBC (3D)
# ============================================================

function apply_bc_3d_left!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[ng + 1 - g, j, k] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_right!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[nx + ng + g, j, k] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ng + 1 - g, k] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_top!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ny + ng + g, k] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_front!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, j, ng + 1 - g] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_back!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, j, nz + ng + g] = u_bc
        end
    end
    return nothing
end

# ============================================================
# InflowBC (3D)
# ============================================================

function apply_bc_3d_left!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[ng + 1 - g, j, k] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_right!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[nx + ng + g, j, k] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ng + 1 - g, k] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_top!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ny + ng + g, k] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_front!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, j, ng + 1 - g] = u_bc
        end
    end
    return nothing
end

function apply_bc_3d_back!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, j, nz + ng + g] = u_bc
        end
    end
    return nothing
end

# ============================================================
# PeriodicHyperbolicBC (3D)
# ============================================================

function apply_bc_3d_periodic_x!(U::AbstractArray{T, 3}, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[ng + 1 - g, j, k] = U[nx + ng + 1 - g, j, k]  # left ghost from right interior
            U[nx + ng + g, j, k] = U[ng + g, j, k]            # right ghost from left interior
        end
    end
    return nothing
end

function apply_bc_3d_periodic_y!(U::AbstractArray{T, 3}, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for k in 1:(nz + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ng + 1 - g, k] = U[i, ny + ng + 1 - g, k]  # bottom ghost from top interior
            U[i, ny + ng + g, k] = U[i, ng + g, k]            # top ghost from bottom interior
        end
    end
    return nothing
end

function apply_bc_3d_periodic_z!(U::AbstractArray{T, 3}, law, nx::Int, ny::Int, nz::Int, ng::Int, t) where {T}
    for j in 1:(ny + 2 * ng), i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, j, ng + 1 - g] = U[i, j, nz + ng + 1 - g]  # front ghost from back interior
            U[i, j, nz + ng + g] = U[i, j, ng + g]            # back ghost from front interior
        end
    end
    return nothing
end

# ============================================================
# Apply all 3D boundary conditions
# ============================================================

"""
    apply_boundary_conditions_3d!(U, prob, ng, t)

Apply boundary conditions on all 6 faces of the 3D domain.
Order: left/right (x), bottom/top (y), front/back (z).
Periodic BCs are handled specially.
"""
function apply_boundary_conditions_3d!(U::AbstractArray{T, 3}, prob::HyperbolicProblem3D, ng::Int, t) where {T}
    nx = prob.mesh.nx
    ny = prob.mesh.ny
    nz = prob.mesh.nz
    law = prob.law

    # Handle periodic BCs in x
    if prob.bc_left isa PeriodicHyperbolicBC && prob.bc_right isa PeriodicHyperbolicBC
        apply_bc_3d_periodic_x!(U, law, nx, ny, nz, ng, t)
    else
        apply_bc_3d_left!(U, prob.bc_left, law, nx, ny, nz, ng, t)
        apply_bc_3d_right!(U, prob.bc_right, law, nx, ny, nz, ng, t)
    end

    # Handle periodic BCs in y
    if prob.bc_bottom isa PeriodicHyperbolicBC && prob.bc_top isa PeriodicHyperbolicBC
        apply_bc_3d_periodic_y!(U, law, nx, ny, nz, ng, t)
    else
        apply_bc_3d_bottom!(U, prob.bc_bottom, law, nx, ny, nz, ng, t)
        apply_bc_3d_top!(U, prob.bc_top, law, nx, ny, nz, ng, t)
    end

    # Handle periodic BCs in z
    if prob.bc_front isa PeriodicHyperbolicBC && prob.bc_back isa PeriodicHyperbolicBC
        apply_bc_3d_periodic_z!(U, law, nx, ny, nz, ng, t)
    else
        apply_bc_3d_front!(U, prob.bc_front, law, nx, ny, nz, ng, t)
        apply_bc_3d_back!(U, prob.bc_back, law, nx, ny, nz, ng, t)
    end

    return nothing
end

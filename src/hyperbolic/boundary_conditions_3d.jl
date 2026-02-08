# ============================================================
# 3D Ghost-Cell Boundary Conditions
# ============================================================
#
# The 3D solution is stored in a padded 3D array U[i, j, k] where:
#   i = 1:2           -> left ghost planes   (x-)
#   i = 3:nx+2        -> interior            (x)
#   i = nx+3:nx+4     -> right ghost planes  (x+)
#   j = 1:2           -> bottom ghost planes (y-)
#   j = 3:ny+2        -> interior            (y)
#   j = ny+3:ny+4     -> top ghost planes    (y+)
#   k = 1:2           -> front ghost planes  (z-)
#   k = 3:nz+2        -> interior            (z)
#   k = nz+3:nz+4     -> back ghost planes   (z+)
#
# Interior cell (ix, iy, iz) (1-based) maps to U[ix+2, iy+2, iz+2].

# ============================================================
# TransmissiveBC (3D)
# ============================================================

function apply_bc_3d_left!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), j in 1:(ny + 4)
        U[2, j, k] = U[3, j, k]
        U[1, j, k] = U[3, j, k]
    end
    return nothing
end

function apply_bc_3d_right!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), j in 1:(ny + 4)
        U[nx + 3, j, k] = U[nx + 2, j, k]
        U[nx + 4, j, k] = U[nx + 2, j, k]
    end
    return nothing
end

function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), i in 1:(nx + 4)
        U[i, 2, k] = U[i, 3, k]
        U[i, 1, k] = U[i, 3, k]
    end
    return nothing
end

function apply_bc_3d_top!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), i in 1:(nx + 4)
        U[i, ny + 3, k] = U[i, ny + 2, k]
        U[i, ny + 4, k] = U[i, ny + 2, k]
    end
    return nothing
end

function apply_bc_3d_front!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U[i, j, 2] = U[i, j, 3]
        U[i, j, 1] = U[i, j, 3]
    end
    return nothing
end

function apply_bc_3d_back!(U::AbstractArray{T, 3}, ::TransmissiveBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U[i, j, nz + 3] = U[i, j, nz + 2]
        U[i, j, nz + 4] = U[i, j, nz + 2]
    end
    return nothing
end

# ============================================================
# ReflectiveBC (3D Euler)
# ============================================================

# Left wall: negate vx (component 2 in 5-var Euler)
function apply_bc_3d_left!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[3, j, k])
        w2 = conserved_to_primitive(law, U[4, j, k])
        U[2, j, k] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5]))
        U[1, j, k] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5]))
    end
    return nothing
end

# Right wall: negate vx
function apply_bc_3d_right!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[nx + 2, j, k])
        w2 = conserved_to_primitive(law, U[nx + 1, j, k])
        U[nx + 3, j, k] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5]))
        U[nx + 4, j, k] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5]))
    end
    return nothing
end

# Bottom wall: negate vy (component 3)
function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, 3, k])
        w2 = conserved_to_primitive(law, U[i, 4, k])
        U[i, 2, k] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5]))
        U[i, 1, k] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5]))
    end
    return nothing
end

# Top wall: negate vy
function apply_bc_3d_top!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, ny + 2, k])
        w2 = conserved_to_primitive(law, U[i, ny + 1, k])
        U[i, ny + 3, k] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5]))
        U[i, ny + 4, k] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5]))
    end
    return nothing
end

# Front wall: negate vz (component 4)
function apply_bc_3d_front!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for j in 1:(ny + 4), i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, j, 3])
        w2 = conserved_to_primitive(law, U[i, j, 4])
        U[i, j, 2] = primitive_to_conserved(law, SVector(w1[1], w1[2], w1[3], -w1[4], w1[5]))
        U[i, j, 1] = primitive_to_conserved(law, SVector(w2[1], w2[2], w2[3], -w2[4], w2[5]))
    end
    return nothing
end

# Back wall: negate vz
function apply_bc_3d_back!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::EulerEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for j in 1:(ny + 4), i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, j, nz + 2])
        w2 = conserved_to_primitive(law, U[i, j, nz + 1])
        U[i, j, nz + 3] = primitive_to_conserved(law, SVector(w1[1], w1[2], w1[3], -w1[4], w1[5]))
        U[i, j, nz + 4] = primitive_to_conserved(law, SVector(w2[1], w2[2], w2[3], -w2[4], w2[5]))
    end
    return nothing
end

# ============================================================
# ReflectiveBC (3D MHD)
# ============================================================

# Left wall: negate vx (index 2)
function apply_bc_3d_left!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[3, j, k])
        w2 = conserved_to_primitive(law, U[4, j, k])
        U[2, j, k] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[1, j, k] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# Right wall: negate vx
function apply_bc_3d_right!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[nx + 2, j, k])
        w2 = conserved_to_primitive(law, U[nx + 1, j, k])
        U[nx + 3, j, k] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[nx + 4, j, k] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# Bottom wall: negate vy (index 3)
function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, 3, k])
        w2 = conserved_to_primitive(law, U[i, 4, k])
        U[i, 2, k] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, 1, k] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# Top wall: negate vy
function apply_bc_3d_top!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, ny + 2, k])
        w2 = conserved_to_primitive(law, U[i, ny + 1, k])
        U[i, ny + 3, k] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, ny + 4, k] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# Front wall: negate vz (index 4)
function apply_bc_3d_front!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for j in 1:(ny + 4), i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, j, 3])
        w2 = conserved_to_primitive(law, U[i, j, 4])
        U[i, j, 2] = primitive_to_conserved(law, SVector(w1[1], w1[2], w1[3], -w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, j, 1] = primitive_to_conserved(law, SVector(w2[1], w2[2], w2[3], -w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# Back wall: negate vz
function apply_bc_3d_back!(U::AbstractArray{T, 3}, ::ReflectiveBC, law::IdealMHDEquations{3}, nx::Int, ny::Int, nz::Int, t) where {T}
    for j in 1:(ny + 4), i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, j, nz + 2])
        w2 = conserved_to_primitive(law, U[i, j, nz + 1])
        U[i, j, nz + 3] = primitive_to_conserved(law, SVector(w1[1], w1[2], w1[3], -w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, j, nz + 4] = primitive_to_conserved(law, SVector(w2[1], w2[2], w2[3], -w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# ============================================================
# DirichletHyperbolicBC (3D)
# ============================================================

function apply_bc_3d_left!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 4), j in 1:(ny + 4)
        U[2, j, k] = u_bc
        U[1, j, k] = u_bc
    end
    return nothing
end

function apply_bc_3d_right!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 4), j in 1:(ny + 4)
        U[nx + 3, j, k] = u_bc
        U[nx + 4, j, k] = u_bc
    end
    return nothing
end

function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 4), i in 1:(nx + 4)
        U[i, 2, k] = u_bc
        U[i, 1, k] = u_bc
    end
    return nothing
end

function apply_bc_3d_top!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 4), i in 1:(nx + 4)
        U[i, ny + 3, k] = u_bc
        U[i, ny + 4, k] = u_bc
    end
    return nothing
end

function apply_bc_3d_front!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U[i, j, 2] = u_bc
        U[i, j, 1] = u_bc
    end
    return nothing
end

function apply_bc_3d_back!(U::AbstractArray{T, 3}, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U[i, j, nz + 3] = u_bc
        U[i, j, nz + 4] = u_bc
    end
    return nothing
end

# ============================================================
# InflowBC (3D)
# ============================================================

function apply_bc_3d_left!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 4), j in 1:(ny + 4)
        U[2, j, k] = u_bc
        U[1, j, k] = u_bc
    end
    return nothing
end

function apply_bc_3d_right!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 4), j in 1:(ny + 4)
        U[nx + 3, j, k] = u_bc
        U[nx + 4, j, k] = u_bc
    end
    return nothing
end

function apply_bc_3d_bottom!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 4), i in 1:(nx + 4)
        U[i, 2, k] = u_bc
        U[i, 1, k] = u_bc
    end
    return nothing
end

function apply_bc_3d_top!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for k in 1:(nz + 4), i in 1:(nx + 4)
        U[i, ny + 3, k] = u_bc
        U[i, ny + 4, k] = u_bc
    end
    return nothing
end

function apply_bc_3d_front!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U[i, j, 2] = u_bc
        U[i, j, 1] = u_bc
    end
    return nothing
end

function apply_bc_3d_back!(U::AbstractArray{T, 3}, bc::InflowBC, law, nx::Int, ny::Int, nz::Int, t) where {T}
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U[i, j, nz + 3] = u_bc
        U[i, j, nz + 4] = u_bc
    end
    return nothing
end

# ============================================================
# PeriodicHyperbolicBC (3D)
# ============================================================

function apply_bc_3d_periodic_x!(U::AbstractArray{T, 3}, law, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), j in 1:(ny + 4)
        U[2, j, k] = U[nx + 2, j, k]
        U[1, j, k] = U[nx + 1, j, k]
        U[nx + 3, j, k] = U[3, j, k]
        U[nx + 4, j, k] = U[4, j, k]
    end
    return nothing
end

function apply_bc_3d_periodic_y!(U::AbstractArray{T, 3}, law, nx::Int, ny::Int, nz::Int, t) where {T}
    for k in 1:(nz + 4), i in 1:(nx + 4)
        U[i, 2, k] = U[i, ny + 2, k]
        U[i, 1, k] = U[i, ny + 1, k]
        U[i, ny + 3, k] = U[i, 3, k]
        U[i, ny + 4, k] = U[i, 4, k]
    end
    return nothing
end

function apply_bc_3d_periodic_z!(U::AbstractArray{T, 3}, law, nx::Int, ny::Int, nz::Int, t) where {T}
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U[i, j, 2] = U[i, j, nz + 2]
        U[i, j, 1] = U[i, j, nz + 1]
        U[i, j, nz + 3] = U[i, j, 3]
        U[i, j, nz + 4] = U[i, j, 4]
    end
    return nothing
end

# ============================================================
# Apply all 3D boundary conditions
# ============================================================

"""
    apply_boundary_conditions_3d!(U, prob, t)

Apply boundary conditions on all 6 faces of the 3D domain.
Order: left/right (x), bottom/top (y), front/back (z).
Periodic BCs are handled specially.
"""
function apply_boundary_conditions_3d!(U::AbstractArray{T, 3}, prob::HyperbolicProblem3D, t) where {T}
    nx = prob.mesh.nx
    ny = prob.mesh.ny
    nz = prob.mesh.nz
    law = prob.law

    # Handle periodic BCs in x
    if prob.bc_left isa PeriodicHyperbolicBC && prob.bc_right isa PeriodicHyperbolicBC
        apply_bc_3d_periodic_x!(U, law, nx, ny, nz, t)
    else
        apply_bc_3d_left!(U, prob.bc_left, law, nx, ny, nz, t)
        apply_bc_3d_right!(U, prob.bc_right, law, nx, ny, nz, t)
    end

    # Handle periodic BCs in y
    if prob.bc_bottom isa PeriodicHyperbolicBC && prob.bc_top isa PeriodicHyperbolicBC
        apply_bc_3d_periodic_y!(U, law, nx, ny, nz, t)
    else
        apply_bc_3d_bottom!(U, prob.bc_bottom, law, nx, ny, nz, t)
        apply_bc_3d_top!(U, prob.bc_top, law, nx, ny, nz, t)
    end

    # Handle periodic BCs in z
    if prob.bc_front isa PeriodicHyperbolicBC && prob.bc_back isa PeriodicHyperbolicBC
        apply_bc_3d_periodic_z!(U, law, nx, ny, nz, t)
    else
        apply_bc_3d_front!(U, prob.bc_front, law, nx, ny, nz, t)
        apply_bc_3d_back!(U, prob.bc_back, law, nx, ny, nz, t)
    end

    return nothing
end

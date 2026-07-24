# Turbulence modeling utilities
# Migrated from Simu.jl SimuFVM/turbulence.jl
# StandardKEpsilon -> KEpsilon
# Dirichlet -> DirichletBC, Neumann -> NeumannBC, Robin -> RobinBC
# TurbulentWall kept its name (Stage 4b dropped the interim Parabolic prefix)

"""
    update_turbulent_viscosity!(mu_t, model::KEpsilon, rho, k, epsilon)

Update turbulent viscosity field based on k-epsilon model.
mu_t = rho * C_mu * k^2 / epsilon
"""
function update_turbulent_viscosity!(mu_t::Vector{Float64}, model::KEpsilon, rho::Vector{Float64}, k::Vector{Float64}, epsilon::Vector{Float64})
    return @inbounds for i in 1:length(mu_t)
        eps = max(epsilon[i], 1.0e-10)
        k_val = max(k[i], 0.0)
        mu_t[i] = rho[i] * model.C_mu * k_val^2 / eps
    end
end

"""
    compute_production_k(mesh, u, v, mu_t; bcs=nothing)

Compute production of turbulent kinetic energy Pk in 2D.
"""
function compute_production_k(mesh::Union{Mesh2D, UnstructuredMesh2D}, u::Vector{Float64}, v::Vector{Float64}, mu_t::Vector{Float64}; bcs = nothing)
    nx = length(mu_t)
    Pk = zeros(nx)

    if mesh isa UnstructuredMesh2D
        for i in 1:nx
            dudx, dudy = reconstruct_gradient_green_gauss_2d(mesh, u, i; bcs = bcs)
            dvdx, dvdy = reconstruct_gradient_green_gauss_2d(mesh, v, i; bcs = bcs)

            S_xx = dudx
            S_yy = dvdy
            S_xy = 0.5 * (dudy + dvdx)

            S_sq = 2.0 * (S_xx^2 + S_yy^2 + 2.0 * S_xy^2)

            Pk[i] = mu_t[i] * S_sq
        end
    else
        for k in 1:nx
            j = mod(k - 1, mesh.ny) + 1
            i = div(k - 1, mesh.ny) + 1

            dudx, dudy = reconstruct_gradient_green_gauss_2d(mesh, u, i, j)
            dvdx, dvdy = reconstruct_gradient_green_gauss_2d(mesh, v, i, j)

            S_xx = dudx
            S_yy = dvdy
            S_xy = 0.5 * (dudy + dvdx)

            S_sq = 2.0 * (S_xx^2 + S_yy^2 + 2.0 * S_xy^2)

            Pk[k] = mu_t[k] * S_sq
        end
    end

    return Pk
end

"""
    compute_production_k(mesh, u, v, w, mu_t; bcs=nothing)

Compute production of turbulent kinetic energy Pk in 3D.
"""
function compute_production_k(mesh::Union{Mesh3D, UnstructuredMesh3D}, u::Vector{Float64}, v::Vector{Float64}, w::Vector{Float64}, mu_t::Vector{Float64}; bcs = nothing)
    nx = length(mu_t)
    Pk = zeros(nx)

    if mesh isa UnstructuredMesh3D
        for i in 1:nx
            dudx, dudy, dudz = reconstruct_gradient_green_gauss_3d(mesh, u, i; bcs = bcs)
            dvdx, dvdy, dvdz = reconstruct_gradient_green_gauss_3d(mesh, v, i; bcs = bcs)
            dwdx, dwdy, dwdz = reconstruct_gradient_green_gauss_3d(mesh, w, i; bcs = bcs)

            S_xx = dudx; S_yy = dvdy; S_zz = dwdz
            S_xy = 0.5 * (dudy + dvdx)
            S_xz = 0.5 * (dudz + dwdx)
            S_yz = 0.5 * (dvdz + dwdy)

            S_sq = 2.0 * (S_xx^2 + S_yy^2 + S_zz^2 + 2.0 * (S_xy^2 + S_xz^2 + S_yz^2))

            Pk[i] = mu_t[i] * S_sq
        end
    else
        # Structured Mesh3D: use finite-difference gradients
        dx = mesh.Lx / mesh.nx
        dy = mesh.Ly / mesh.ny
        dz = mesh.Lz / mesh.nz
        for c in 1:nx
            # Convert linear index to (i,j,k) using column-major ordering
            # k = (kz-1)*nx*ny + (ky-1)*nx + kx  where kx,ky,kz are 1-based
            kx = mod(c - 1, mesh.nx) + 1
            ky = mod(div(c - 1, mesh.nx), mesh.ny) + 1
            kz = div(c - 1, mesh.nx * mesh.ny) + 1

            _idx(ix, iy, iz) = (iz - 1) * mesh.nx * mesh.ny + (iy - 1) * mesh.nx + ix

            # Central-difference gradients with one-sided at boundaries
            if kx == 1
                dudx = (u[_idx(2, ky, kz)] - u[c]) / dx
                dvdx = (v[_idx(2, ky, kz)] - v[c]) / dx
                dwdx = (w[_idx(2, ky, kz)] - w[c]) / dx
            elseif kx == mesh.nx
                dudx = (u[c] - u[_idx(mesh.nx - 1, ky, kz)]) / dx
                dvdx = (v[c] - v[_idx(mesh.nx - 1, ky, kz)]) / dx
                dwdx = (w[c] - w[_idx(mesh.nx - 1, ky, kz)]) / dx
            else
                dudx = (u[_idx(kx + 1, ky, kz)] - u[_idx(kx - 1, ky, kz)]) / (2 * dx)
                dvdx = (v[_idx(kx + 1, ky, kz)] - v[_idx(kx - 1, ky, kz)]) / (2 * dx)
                dwdx = (w[_idx(kx + 1, ky, kz)] - w[_idx(kx - 1, ky, kz)]) / (2 * dx)
            end

            if ky == 1
                dudy = (u[_idx(kx, 2, kz)] - u[c]) / dy
                dvdy = (v[_idx(kx, 2, kz)] - v[c]) / dy
                dwdy = (w[_idx(kx, 2, kz)] - w[c]) / dy
            elseif ky == mesh.ny
                dudy = (u[c] - u[_idx(kx, mesh.ny - 1, kz)]) / dy
                dvdy = (v[c] - v[_idx(kx, mesh.ny - 1, kz)]) / dy
                dwdy = (w[c] - w[_idx(kx, mesh.ny - 1, kz)]) / dy
            else
                dudy = (u[_idx(kx, ky + 1, kz)] - u[_idx(kx, ky - 1, kz)]) / (2 * dy)
                dvdy = (v[_idx(kx, ky + 1, kz)] - v[_idx(kx, ky - 1, kz)]) / (2 * dy)
                dwdy = (w[_idx(kx, ky + 1, kz)] - w[_idx(kx, ky - 1, kz)]) / (2 * dy)
            end

            if kz == 1
                dudz = (u[_idx(kx, ky, 2)] - u[c]) / dz
                dvdz = (v[_idx(kx, ky, 2)] - v[c]) / dz
                dwdz = (w[_idx(kx, ky, 2)] - w[c]) / dz
            elseif kz == mesh.nz
                dudz = (u[c] - u[_idx(kx, ky, mesh.nz - 1)]) / dz
                dvdz = (v[c] - v[_idx(kx, ky, mesh.nz - 1)]) / dz
                dwdz = (w[c] - w[_idx(kx, ky, mesh.nz - 1)]) / dz
            else
                dudz = (u[_idx(kx, ky, kz + 1)] - u[_idx(kx, ky, kz - 1)]) / (2 * dz)
                dvdz = (v[_idx(kx, ky, kz + 1)] - v[_idx(kx, ky, kz - 1)]) / (2 * dz)
                dwdz = (w[_idx(kx, ky, kz + 1)] - w[_idx(kx, ky, kz - 1)]) / (2 * dz)
            end

            S_xx = dudx; S_yy = dvdy; S_zz = dwdz
            S_xy = 0.5 * (dudy + dvdx)
            S_xz = 0.5 * (dudz + dwdx)
            S_yz = 0.5 * (dvdz + dwdy)

            S_sq = 2.0 * (S_xx^2 + S_yy^2 + S_zz^2 + 2.0 * (S_xy^2 + S_xz^2 + S_yz^2))
            Pk[c] = mu_t[c] * S_sq
        end
    end

    return Pk
end

"""
    assemble_k_source(model, rho, k, epsilon, Pk)

Returns LinearizedSource for k equation.
S = Pk - rho * epsilon
  = Pk - (rho * epsilon/k) * k
S_C = Pk
S_P = -rho * epsilon / k
"""
function assemble_k_source(model::KEpsilon, rho::Vector{Float64}, k::Vector{Float64}, epsilon::Vector{Float64}, Pk::Vector{Float64})
    nx = length(rho)
    sc = copy(Pk)
    sp = zeros(nx)
    for i in 1:nx
        sp[i] = -rho[i] * epsilon[i] / max(k[i], 1.0e-10)
    end
    return LinearizedSource(sc, sp)
end

"""
    assemble_epsilon_source(model, rho, k, epsilon, Pk)

Returns LinearizedSource for epsilon equation.
S = C1 * epsilon/k * Pk - C2 * rho * epsilon^2 / k
  = (C1 * epsilon/k * Pk) - (C2 * rho * epsilon / k) * epsilon
S_C = C1 * epsilon/k * Pk
S_P = -C2 * rho * epsilon / k
"""
function assemble_epsilon_source(model::KEpsilon, rho::Vector{Float64}, k::Vector{Float64}, epsilon::Vector{Float64}, Pk::Vector{Float64})
    nx = length(rho)
    sc = zeros(nx)
    sp = zeros(nx)

    C1 = model.C1_epsilon
    C2 = model.C2_epsilon

    for i in 1:nx
        k_safe = max(k[i], 1.0e-10)
        eps_by_k = epsilon[i] / k_safe

        sc[i] = C1 * eps_by_k * Pk[i]
        sp[i] = -C2 * rho[i] * eps_by_k
    end

    return LinearizedSource(sc, sp)
end

"""
    rough_wall_friction_velocity(u_tan, y, nu, roughness)

Compute friction velocity u_tau using the Law of the Wall.
"""
function rough_wall_friction_velocity(u_tan, y, nu, roughness)
    kappa = 0.41
    E = 9.8

    u_tau = sqrt(nu * u_tan / y)

    for iter in 1:10
        y_plus = y * u_tau / nu
        if y_plus < 11.225
            break
        else
            f = u_tan / u_tau - (1.0 / kappa) * log(E * y_plus)
            df = -u_tan / u_tau^2 - (1.0 / kappa) * (1.0 / u_tau)

            delta = f / df
            u_tau -= delta

            if abs(delta) < 1.0e-5
                break
            end
        end
    end
    return u_tau
end

"""
    update_wall_bcs!(bcs, mesh, u, v, k, epsilon, rho, mu, equation)

Update `TurbulentWall` boundary conditions in `bcs` dictionary with concrete linear BCs.
equation: :momentum_x, :momentum_y, :momentum_z, :k, or :epsilon
"""
function update_wall_bcs!(bcs::Dict, mesh::UnstructuredMesh2D, u, v, k, epsilon, rho, mu, equation::Symbol)
    for (f_idx, bc) in bcs
        if bc isa TurbulentWall
            face = mesh.faces[f_idx]
            owner = face.owner

            c_owner = mesh.cells[owner].center
            c_face = face.center
            dist = norm(c_face .- c_owner)

            rho_val = rho[owner]
            mu_val = mu[owner]
            nu = mu_val / rho_val

            u_val = u[owner]
            v_val = v[owner]
            u_tan = sqrt(u_val^2 + v_val^2)

            u_tau = rough_wall_friction_velocity(u_tan, dist, nu, bc.roughness)

            if equation == :k
                bcs[f_idx] = NeumannBC(0.0)
            elseif equation == :epsilon
                C_mu = 0.09
                kappa = 0.41
                k_val = k[owner]
                val = (C_mu^0.75 * k_val^1.5) / (kappa * dist)
                bcs[f_idx] = DirichletBC(val)
            elseif equation == :momentum_x || equation == :momentum_y
                tau_w = rho_val * u_tau^2
                coeff = (u_tan > 1.0e-10) ? tau_w / u_tan : 0.0
                bcs[f_idx] = RobinBC(coeff, 1.0, 0.0)
            end
        end
    end
    return
end

"""
    update_wall_bcs!(bcs, mesh, u, v, w, k, epsilon, rho, mu, equation)

Update `TurbulentWall` boundary conditions in `bcs` dictionary with concrete linear BCs for 3D.
"""
function update_wall_bcs!(bcs::Dict, mesh::UnstructuredMesh3D, u, v, w, k, epsilon, rho, mu, equation::Symbol)
    for (f_idx, bc) in bcs
        if bc isa TurbulentWall
            face = mesh.faces[f_idx]
            owner = face.owner

            c_owner = mesh.cells[owner].center
            c_face = face.center
            dist = norm(c_face .- c_owner)

            rho_val = rho[owner]
            mu_val = mu[owner]
            nu = mu_val / rho_val

            u_val = u[owner]
            v_val = v[owner]
            w_val = w[owner]
            u_tan = sqrt(u_val^2 + v_val^2 + w_val^2)

            u_tau = rough_wall_friction_velocity(u_tan, dist, nu, bc.roughness)

            if equation == :k
                bcs[f_idx] = NeumannBC(0.0)
            elseif equation == :epsilon
                C_mu = 0.09
                kappa = 0.41
                k_val = k[owner]
                val = (C_mu^0.75 * k_val^1.5) / (kappa * dist)
                bcs[f_idx] = DirichletBC(val)
            elseif equation == :momentum_x || equation == :momentum_y || equation == :momentum_z
                tau_w = rho_val * u_tau^2
                coeff = (u_tan > 1.0e-10) ? tau_w / u_tan : 0.0
                bcs[f_idx] = RobinBC(coeff, 1.0, 0.0)
            end
        end
    end
    return
end

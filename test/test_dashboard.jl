using FiniteVolumeMethod
using Test
using StaticArrays
using DelaunayTriangulation
using OrdinaryDiffEq
using JSON3
using HTTP  # triggers FVMDashboardExt alongside JSON3

# ============================================================
# Core Types
# ============================================================
@testset "FVMSnapshot and FVMSessionData" begin
    snap = FVMSnapshot(
        0.1, 5, [1.0, 2.0, 3.0], 1.0e-4,
        Dict("mass" => 1.0), 0.001, 0.5,
    )
    @test snap.time == 0.1
    @test snap.step == 5
    @test snap.residual_norm ≈ 1.0e-4
    @test snap.dt == 0.001
    @test snap.wall_time == 0.5

    session = FVMSessionData(;
        problem_type = "HyperbolicProblem",
        law_name = "EulerEquations{1}",
        mesh_info = Dict{String, Any}("type" => "StructuredMesh1D", "ncells" => 100),
        variable_names = ["rho", "rho_v", "E"],
        parameters = Dict{String, Any}("cfl" => 0.4),
    )
    @test session.problem_type == "HyperbolicProblem"
    @test length(session.snapshots) == 0
    push!(session.snapshots, snap)
    @test length(session.snapshots) == 1
end

# ============================================================
# Variable Names
# ============================================================
@testset "variable_names" begin
    eos = IdealGasEOS(1.4)
    @test variable_names(EulerEquations{1}(eos)) == ["rho", "rho_v", "E"]
    @test variable_names(EulerEquations{2}(eos)) == ["rho", "rho_vx", "rho_vy", "E"]
    @test variable_names(EulerEquations{3}(eos)) == ["rho", "rho_vx", "rho_vy", "rho_vz", "E"]
    @test variable_names(IdealMHDEquations{2}(eos)) == ["rho", "rho_vx", "rho_vy", "rho_vz", "E", "Bx", "By", "Bz"]
    @test variable_names(ShallowWaterEquations{1}(; g = 9.81)) == ["h", "hv"]
    @test variable_names(ShallowWaterEquations{2}(; g = 9.81)) == ["h", "hvx", "hvy"]
end

# ============================================================
# Mesh Serialization
# ============================================================
@testset "mesh_to_dict" begin
    mesh1d = StructuredMesh1D(0.0, 1.0, 50)
    d1 = mesh_to_dict(mesh1d)
    @test d1["type"] == "StructuredMesh1D"
    @test d1["ncells"] == 50
    @test d1["xmin"] == 0.0
    @test d1["xmax"] == 1.0
    @test d1["dx"] ≈ 0.02

    mesh2d = StructuredMesh2D(0.0, 1.0, 0.0, 2.0, 10, 20)
    d2 = mesh_to_dict(mesh2d)
    @test d2["type"] == "StructuredMesh2D"
    @test d2["nx"] == 10
    @test d2["ny"] == 20
    @test d2["dx"] ≈ 0.1
    @test d2["dy"] ≈ 0.1
end

# ============================================================
# Conserved Totals
# ============================================================
@testset "conserved_totals" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 10)
    # Uniform state: ρ=1, v=0, P=1 → u = [1, 0, 2.5]
    w = SVector(1.0, 0.0, 1.0)
    u = primitive_to_conserved(law, w)
    U = fill(u, 10)
    totals = conserved_totals(law, U, mesh)
    @test totals["rho"] ≈ 1.0  # ρ * dx * ncells = 1 * 0.1 * 10
    @test totals["rho_v"] ≈ 0.0
    @test totals["E"] ≈ 2.5  # E * 0.1 * 10 = 2.5
end

# ============================================================
# Snapshot / Session Serialization
# ============================================================
@testset "snapshot_to_dict and session_to_dict" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    w = SVector(1.0, 0.0, 1.0)
    u = primitive_to_conserved(law, w)
    U = fill(u, 5)

    snap = FVMSnapshot(
        0.1, 10, U, 1.0e-5,
        Dict("rho" => 0.5, "rho_v" => 0.0, "E" => 1.25),
        0.002, 0.3,
    )
    d = snapshot_to_dict(snap)
    @test d["time"] == 0.1
    @test d["step"] == 10
    @test d["dt"] == 0.002
    @test d["residual_norm"] ≈ 1.0e-5
    @test length(d["solution"]) == 5
    @test d["solution"][1] ≈ collect(u)

    session = FVMSessionData(;
        problem_type = "HyperbolicProblem",
        law_name = "EulerEquations{1}",
        mesh_info = Dict{String, Any}("type" => "StructuredMesh1D"),
        variable_names = ["rho", "rho_v", "E"],
        parameters = Dict{String, Any}("cfl" => 0.4),
    )
    push!(session.snapshots, snap)
    sd = session_to_dict(session)
    @test sd["version"] == "1.0"
    @test sd["problem_type"] == "HyperbolicProblem"
    @test sd["law"] == "EulerEquations{1}"
    @test length(sd["snapshots"]) == 1
end

# ============================================================
# Hyperbolic Monitor Callback (1D Sod shock tube)
# ============================================================
@testset "hyperbolic_monitor callback" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)
    ic(x) = x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), ic;
        final_time = 0.05,
    )

    session = create_session_data(prob)
    @test session.problem_type == "HyperbolicProblem"
    @test session.variable_names == ["rho", "rho_v", "E"]

    cb = hyperbolic_monitor(;
        interval = 5,
        session_data = session,
        law = law,
        mesh = mesh,
    )

    _, _, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)
    @test t_final ≈ 0.05

    # Should have recorded snapshots every 5 steps
    @test length(session.snapshots) > 0
    # Check that snapshot fields are populated
    snap = session.snapshots[1]
    @test snap.step > 0
    @test snap.time > 0.0
    @test snap.dt > 0.0
    @test snap.wall_time >= 0.0
    @test haskey(snap.conserved_totals, "rho")
    @test haskey(snap.conserved_totals, "E")
end

# ============================================================
# Hyperbolic Monitor Callback (2D)
# ============================================================
@testset "hyperbolic_monitor 2D" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 10, 10)
    ic(x, y) = x < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)
    prob = HyperbolicProblem2D(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ic;
        final_time = 0.01,
    )

    session = create_session_data(prob)
    @test session.problem_type == "HyperbolicProblem2D"

    cb = hyperbolic_monitor(;
        interval = 1,
        session_data = session,
        law = law,
        mesh = mesh,
    )

    _, _, t_final = solve_hyperbolic(prob; method = :ssprk3, callback = cb)
    @test t_final ≈ 0.01
    @test length(session.snapshots) > 0

    # 2D snapshot solution should be a matrix
    snap = session.snapshots[1]
    @test snap.solution isa AbstractMatrix
    @test size(snap.solution) == (10, 10)
end

# ============================================================
# Parabolic (FVMProblem) Monitor Callback
# ============================================================
@testset "FVMMonitorCallback" begin
    tri = triangulate_rectangle(0, 1, 0, 1, 11, 11; single_boundary = true)
    mesh = FVMGeometry(tri)
    bc = (x, y, t, u, p) -> zero(u)
    BCs = BoundaryConditions(mesh, bc, Dirichlet)
    D = (x, y, t, u, p) -> 1.0
    f = (x, y) -> sin(π * x) * sin(π * y)
    initial_condition = [f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
    prob = FVMProblem(
        mesh, BCs;
        diffusion_function = D,
        initial_condition = initial_condition,
        final_time = 0.01,
    )
    ode_prob = ODEProblem(prob)

    session = FVMSessionData(;
        problem_type = "FVMProblem",
        law_name = "DiffusionEquation",
    )
    monitor = FVMMonitorCallback(; interval = 1, session_data = session)
    sol = solve(ode_prob, Tsit5(); callback = monitor, saveat = 0.01)

    @test length(session.snapshots) > 0
    snap = session.snapshots[1]
    @test snap.time > 0.0
    @test haskey(snap.conserved_totals, "total")
end

# ============================================================
# JSON Round-Trip (via FVMDashboardExt)
# ============================================================
@testset "export_session / import_session round-trip" begin
    session = FVMSessionData(;
        problem_type = "HyperbolicProblem",
        law_name = "EulerEquations{1}",
        mesh_info = Dict{String, Any}("type" => "StructuredMesh1D", "ncells" => 50),
        variable_names = ["rho", "rho_v", "E"],
        parameters = Dict{String, Any}("cfl" => 0.4),
    )
    snap = FVMSnapshot(
        0.1, 10, [1.0, 2.0, 3.0], 1.0e-5,
        Dict("rho" => 1.0, "E" => 2.5), 0.002, 0.3,
    )
    push!(session.snapshots, snap)

    tmpfile = tempname() * ".fvm-session.json"
    try
        export_session(session, tmpfile)
        @test isfile(tmpfile)

        data = import_session(tmpfile)
        @test data["version"] == "1.0"
        @test data["problem_type"] == "HyperbolicProblem"
        @test data["law"] == "EulerEquations{1}"
        @test length(data["snapshots"]) == 1
        @test data["snapshots"][1]["time"] == 0.1
        @test data["snapshots"][1]["step"] == 10
    finally
        rm(tmpfile; force = true)
    end
end

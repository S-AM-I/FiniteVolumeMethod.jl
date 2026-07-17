"""
    FVMIO

Output, diagnostics, and dashboard-facing I/O for all solver families:
dashboard session types and extension stubs, output scheduling, CSV/TOML
utilities, conservation/flux diagnostics, VTK ASCII writers, in-situ
probes and monitors, model packaging, and the HDF5/checkpoint extension
stubs. Real HDF5/JLD2/JSON3/HTTP/WriteVTK implementations live in the
package extensions and attach methods to the stubs declared here.
"""
module FVMIO

using Dates: DateTime
using DelimitedFiles: readdlm, writedlm
using Printf: Printf
using TOML: TOML

using DelaunayTriangulation: each_solid_vertex, num_solid_triangles
using StaticArrays: SVector

using ..Geometry: CurvilinearFVMMesh, FVMGeometry, Mesh1D, Mesh2D, Mesh3D,
    StructuredFVMMesh, StructuredMesh1D, StructuredMesh2D, StructuredMesh3D,
    UnstructuredFVMMesh, UnstructuredHyperbolicMesh, UnstructuredMesh3D,
    ncells
using ..Parabolic: AbstractConfig, AbstractOutputManager,
    AdvectionDiffusion3D, Diffusion3D, ParabolicDirichlet, ParabolicNeumann,
    ParabolicRobin, VariableDiffusion3D
# Read-only: FVMIO calls these Hyperbolic generics, it never extends them.
using ..Hyperbolic: nvariables, variable_names

export FVMSnapshot, FVMSessionData, mesh_to_dict, conserved_totals,
    snapshot_to_dict, session_to_dict, add_convergence_point!,
    hyperbolic_monitor, create_session_data, FVMMonitorCallback,
    export_session, import_session, serve_dashboard
export ensure_output_dirs, write_csv, stringify_keys, write_metadata_toml,
    print_scientific, print_with_units, print_table_header, print_table_row,
    print_progress, ensure_extension, safe_filename
export OutputSchedule, OutputTarget, Diagnostic, SimulationConfig,
    Provenance, OutputManager, validate_schedule, next_write_time,
    run_diagnostics
export volume_integral, conservation_summary, boundary_fluxes, flux_inout,
    write_boundary_flux_csv, write_operator_splits_csv
export write_line_vtk, write_structured_vtk_3d
export AbstractMonitor, Probe, IntegralMonitor, find_cell_containing,
    sample_probe, compute_integral
export save_model_package, load_model_package
export write_solution_hdf5, read_solution_hdf5
export CheckpointManager, save_checkpoint, load_checkpoint

include("dashboard_types.jl")
include("utils.jl")
include("manager.jl")
include("diagnostics.jl")
include("vtk.jl")
include("insitu.jl")
include("registry.jl")
include("hdf5.jl")
include("checkpointing.jl")

end # module FVMIO

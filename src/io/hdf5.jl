# HDF5 I/O stubs — real implementations in ext/FVMHdf5Ext.jl
# These function stubs are defined here so that code can reference them without
# requiring HDF5.jl at load time.  Calling them without loading the HDF5
# extension will throw a MethodError.

"""Write solution data to an HDF5 file. Requires the HDF5 extension."""
function write_solution_hdf5 end
"""Read solution data from an HDF5 file. Requires the HDF5 extension."""
function read_solution_hdf5 end

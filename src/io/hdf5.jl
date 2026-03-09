# HDF5 I/O stubs — real implementations in ext/FVMHdf5Ext.jl
# These function stubs are defined here so that code can reference them without
# requiring HDF5.jl at load time.  Calling them without loading the HDF5
# extension will throw a MethodError.

function write_solution_hdf5 end
function read_solution_hdf5 end

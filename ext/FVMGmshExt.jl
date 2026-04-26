module FVMGmshExt

using FiniteVolumeMethod
using Gmsh: Gmsh

# Override the package's `run_gmsh_pipeline` stub once Gmsh.jl is loaded.
function FiniteVolumeMethod.run_gmsh_pipeline(
        pipeline::FiniteVolumeMethod.GmshPipeline, out_path::AbstractString,
    )
    Gmsh.gmsh.initialize()
    try
        if endswith(pipeline.script, ".geo")
            Gmsh.gmsh.open(pipeline.script)
        else
            Gmsh.gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 1.0)
        end
        Gmsh.gmsh.model.mesh.generate(3)
        Gmsh.gmsh.write(String(out_path))
    finally
        Gmsh.gmsh.finalize()
    end
    return out_path
end

end

# ============================================================
# JSON Export / Import
# ============================================================

"""
    export_session(session::FVMSessionData, filename::AbstractString)

Write the session data to a `.fvm-session.json` file.
"""
function FiniteVolumeMethod.export_session(session::FiniteVolumeMethod.FVMSessionData, filename::AbstractString)
    d = FiniteVolumeMethod.session_to_dict(session)
    open(filename, "w") do io
        JSON3.pretty(io, d)
    end
    return filename
end

"""
    import_session(filename::AbstractString) -> Dict

Read a `.fvm-session.json` file and return its contents as a Dict.
"""
function FiniteVolumeMethod.import_session(filename::AbstractString)
    return open(filename, "r") do io
        JSON3.read(io, Dict)
    end
end

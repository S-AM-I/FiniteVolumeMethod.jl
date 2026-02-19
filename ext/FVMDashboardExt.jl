module FVMDashboardExt

using FiniteVolumeMethod
using HTTP
using JSON3

# ============================================================
# JSON Export / Import
# ============================================================

"""
    export_session(session::FVMSessionData, filename::AbstractString)

Write the session data to a `.fvm-session.json` file.
Requires the JSON3 package to be loaded.
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

# ============================================================
# WebSocket Dashboard Server
# ============================================================

"""
    serve_dashboard(; port=8765, session_data=nothing)

Start a WebSocket server that pushes snapshots to connected dashboard clients.

Returns a `(server, push_snapshot!)` tuple:
- `server`: The HTTP server handle (call `close(server)` to stop).
- `push_snapshot!(snap::FVMSnapshot)`: Function to push a snapshot to all clients.

# Keyword Arguments
- `port::Int`: WebSocket port (default: 8765).
- `session_data::Union{Nothing,FVMSessionData}`: If provided, send session metadata
  to newly connected clients.
"""
function FiniteVolumeMethod.serve_dashboard(;
        port::Int = 8765,
        session_data::Union{Nothing, FiniteVolumeMethod.FVMSessionData} = nothing,
    )
    clients = Set{HTTP.WebSockets.WebSocket}()
    lock = ReentrantLock()

    function ws_handler(ws::HTTP.WebSockets.WebSocket)
        Base.lock(lock) do
            push!(clients, ws)
        end
        return try
            # Send session metadata on connect
            if session_data !== nothing
                meta = Dict(
                    "type" => "session_metadata",
                    "data" => FiniteVolumeMethod.session_to_dict(session_data),
                )
                HTTP.WebSockets.send(ws, JSON3.write(meta))
            end
            # Keep connection alive, read incoming messages
            while !HTTP.WebSockets.isclosed(ws)
                try
                    HTTP.WebSockets.receive(ws)
                catch e
                    if e isa HTTP.WebSockets.WebSocketError || e isa EOFError
                        break
                    end
                    rethrow()
                end
            end
        finally
            Base.lock(lock) do
                delete!(clients, ws)
            end
        end
    end

    server = HTTP.WebSockets.listen!(ws_handler, "0.0.0.0", port)

    function push_snapshot!(snap::FiniteVolumeMethod.FVMSnapshot)
        msg = Dict(
            "type" => "snapshot",
            "data" => FiniteVolumeMethod.snapshot_to_dict(snap),
        )
        json_msg = JSON3.write(msg)
        Base.lock(lock) do
            for ws in clients
                try
                    if !HTTP.WebSockets.isclosed(ws)
                        HTTP.WebSockets.send(ws, json_msg)
                    end
                catch
                    # Client disconnected, will be cleaned up
                end
            end
        end
        return nothing
    end

    return server, push_snapshot!
end

end # module

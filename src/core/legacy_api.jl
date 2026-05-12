function _v2_api_depwarn(method_name::Symbol, replacement::AbstractString)
    msg = "`$(method_name)` is a legacy convenience API in the v2 transition. " *
        "Prefer $(replacement) for the canonical SciML execution path."
    return Base.depwarn(msg, method_name)
end

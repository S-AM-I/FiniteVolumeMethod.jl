_merge_problem_callbacks(::Nothing, ::Nothing) = nothing
_merge_problem_callbacks(default_callback, ::Nothing) = default_callback
_merge_problem_callbacks(::Nothing, callback) = callback
_merge_problem_callbacks(default_callback, callback) = CallbackSet(default_callback, callback)

_problem_callback(prob) = haskey(prob.kwargs, :callback) ? prob.kwargs[:callback] : nothing

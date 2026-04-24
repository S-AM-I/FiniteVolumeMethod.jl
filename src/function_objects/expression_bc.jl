# function_objects/expression_bc.jl — Runtime-evaluated expression BCs.
#
# Lightweight alternative to `RuntimeGeneratedFunctions.jl`: we parse
# the supplied expression once and wrap it into an anonymous function
# via `eval` at first evaluation, then cache the result. Good enough
# for BCs and function-object probes; NOT safe to run against untrusted
# input.

"""
    ExpressionBC

Runtime-evaluated boundary condition / function-object. Holds a Julia
expression string and a kwarg bag of user-supplied constants; produces
a callable `evaluate(bc, x, y, z, t)` that returns the expression
evaluated with those arguments in scope.

```julia
bc = ExpressionBC("2 * sin(2*pi*t) * cos(pi*x/L)"; L = 1.0)
v = evaluate(bc, 0.25, 0.0, 0.0, 0.1)
```
"""
mutable struct ExpressionBC
    expression::String
    constants::Dict{Symbol, Any}
    _compiled::Any       # cached Function or nothing
end

function ExpressionBC(expression::AbstractString; constants...)
    return ExpressionBC(String(expression), Dict{Symbol, Any}(constants), nothing)
end

function _compile!(bc::ExpressionBC)
    bc._compiled !== nothing && return bc._compiled
    body = Meta.parse(bc.expression)
    consts = bc.constants
    const_exprs = Expr[]
    for (k, v) in consts
        push!(const_exprs, Expr(:(=), k, v))
    end
    full_body = Expr(:block, const_exprs..., body)
    fn_expr = :(
        (x, y, z, t) -> $(full_body)
    )
    fn = eval(fn_expr)
    bc._compiled = fn
    return fn
end

"""
    evaluate(bc::ExpressionBC, x, y, z, t) -> Number

Evaluate the cached expression. The first call compiles it; subsequent
calls reuse the cached function.
"""
function evaluate(bc::ExpressionBC, x, y, z, t)
    fn = _compile!(bc)
    return Base.invokelatest(fn, x, y, z, t)
end

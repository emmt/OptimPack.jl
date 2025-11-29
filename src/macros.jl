"""
     OptimPack.@dispatch_on_multiplier sym expr

Expand to code dispatching expression `expr` depending on the value and type of the
multiplier bound to symbol `sym`.

For example:

```julia
@dispatch_on_multiplier β unsafe_axpby!(dst, α, x, β, y)
```

expands to (with comments removed):

```julia
if !(β isa OptimPack.StaticMultiplier) && Base.iszero(β)
    unsafe_axpby!(dst, α, x, Neutrals.Neutral{0}()*Unitful.unit(β), y)
elseif !(β isa OptimPack.StaticMultiplier) && β == Base.oneunit(β)
    unsafe_axpby!(dst, α, x, Neutrals.Neutral{1}()*Unitful.unit(β), y)
elseif !(β isa OptimPack.StaticMultiplier) && TypeUtils.is_signed(β) && β == -Base.oneunit(β)
    unsafe_axpby!(dst, α, x, Neutrals.Neutral{-1}()*Unitful.unit(β), y)
else
    unsafe_axpby!(dst, α, x, β, y)
end
```

This can be checked thanks to `@macroexpand`:

```julia
@macroexpand OptimPack.@dispatch_on_multiplier β unsafe_axpby!(dst, α, x, β, y)
```

"""
macro dispatch_on_multiplier(sym::Union{Symbol,QuoteNode}, expr::Expr)
    esc(:(if !($sym isa OptimPack.StaticMultiplier) && Base.isequal($sym, Base.zero($sym))
              $(recode(expr, sym => :(Neutrals.Neutral{0}()*Unitful.unit($sym))))
          elseif !($sym isa OptimPack.StaticMultiplier) && Base.isequal($sym, Base.oneunit($sym))
              $(recode(expr, sym => :(Neutrals.Neutral{1}()*Unitful.unit($sym))))
          elseif !($sym isa OptimPack.StaticMultiplier) && TypeUtils.is_signed($sym) && Base.isequal($sym, -Base.oneunit($sym))
              $(recode(expr, sym => :(Neutrals.Neutral{-1}()*Unitful.unit($sym))))
          else
              $expr
          end))
end

"""
    OptimPack.@pass ex -> ex

Return expression `ex` unchanged.

"""
macro pass(ex)
    esc(ex)
end

"""
    OptimPack.recode(ex, a => b, ...) -> ex′

Return expression `ex` with all symbols or macros named `a` replaced by `b`. There may be
any number of replacement rules all specified by pairs. `a` may be a string or a symbol, `b`
may be anything. If `a` is a string, it is replaced by the corresponding symbol and
similarly for `b`.

"""
recode(ex::Expr, pairs::Pair...) = recode!(deepcopy(ex), pairs...)

"""
    OptimPack.recode!(ex, a => b, ...) -> ex

Return expression `ex` with all symbols or macros named `a` replaced by `b`. The operation
may be performed in-place (i.e. modifying the content of `ex`). There may be any number of
replacement rules all specified by pairs. `a` and `b` may be strings or symbols.

"""
recode!(ex::Expr, pair::Pair, pairs::Pair...) = recode!(recode!(ex, pair), pairs...)
function recode!(ex::Expr, (a, b)::Pair)
    a = _symbolic(a)
    b = _symbolic(b)
    for i in eachindex(ex.args)
        if ex.args[i] isa Expr
            recode!(ex.args[i], a => b)
        elseif ex.args[i] == a
            ex.args[i] = b
        end
    end
    return ex
end
_symbolic(x::AbstractString) = Symbol(x)
_symbolic(x::Any) = x

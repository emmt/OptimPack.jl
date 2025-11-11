function solve! end
function configure! end
function restart! end
function iterate! end

@noinline throw_bad_argument(msg::AbstractString) = throw(ArgumentError(msg))
@noinline throw_bad_argument(args...) = throw_bad_argument(string(args...))

@noinline throw_dimension_mismatch(msg::AbstractString) = throw(DimensionMismatch(msg))
@noinline throw_dimension_mismatch(args...) = throw_dimension_mismatch(string(args...))

@noinline throw_assertion_failed(msg::AbstractString) = throw(AssertionError(msg))
@noinline throw_assertion_failed(args...) = throw_assertion_failed(string(args...))

@noinline function throw_package_required(name::Symbol)
    if isdefined(Main, name)
        throw(ArgumentError("invalid argument(s)"))
    else
        error("call `using $name` first")
    end
end

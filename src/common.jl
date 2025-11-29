function solve! end
function configure! end
function restart! end
function iterate! end

function ordinal_suffix(n::Integer)
    if n > zero(n)
        ten = oftype(n, 10)
        if rem(div(n, ten), ten) != one(n)
            # Number is positive and the tens digit is not 1.
            r = rem(n, ten)
            if r == oftype(r, 1)
                return "st"
            elseif r == oftype(r, 2)
                return "nd"
            elseif r == oftype(r, 3)
                return "rd"
            end
        end
    end
    return "th"
end

function print_seconds(io::IO, t::Real)
    a = abs(t)
    if a*1e9 < 1e3
        print(io, round(t*1e9; digits=3, base = 10), " ns")
    elseif a*1e6 < 1e3
        print(io, round(t*1e6; digits=3, base = 10), " μs")
    elseif a*1e3 < 1e3
        print(io, round(t*1e3; digits=3, base = 10), " ms")
    else
        print(io, round(t; digits=3, base = 10), " s")
    end
end

@noinline throw_bad_argument(msg::AbstractString) = throw(ArgumentError(msg))
@noinline throw_bad_argument(args...) = throw_bad_argument(string(args...))

@noinline throw_dimension_mismatch(msg::AbstractString) = throw(DimensionMismatch(msg))
@noinline throw_dimension_mismatch(args...) = throw_dimension_mismatch(string(args...))

@noinline throw_incompatible_axes() =  throw_dimension_mismatch(
    "array arguments have incompatible axes")

@noinline throw_assertion_failed(msg::AbstractString) = throw(AssertionError(msg))
@noinline throw_assertion_failed(args...) = throw_assertion_failed(string(args...))

@noinline function throw_package_required(name::Symbol)
    if isdefined(Main, name)
        throw(ArgumentError("invalid argument(s)"))
    else
        error("call `using $name` first")
    end
end

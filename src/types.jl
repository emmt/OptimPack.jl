const StaticMultiplier{v} = Union{Neutral{v},AbstractQuantity{Neutral{v}}}

if !isdefined(@__MODULE__, :Memory)
    const Memory{T} = Vector{T}
end

"""
    Bounds(lower, upper) -> B

Create an object representing bounds on the variables. `lower` and `upper` are the
respective lower an upper bounds on the variables, they must have the same axes and element
types.

The object `B` is callable: `B(x)` overwrites `x` with its projection into the feasible set
defined by the bounds and returns `x`.

"""
struct Bounds{T,N,
              L<:AbstractArray{T,N},
              U<:AbstractArray{T,N}}
    lower::L
    upper::U
    function Bounds(lower::L, upper::U) where {T,N,
                                               L<:AbstractArray{T,N},
                                               U<:AbstractArray{T,N}}
        axes(lower) == axes(upper) ||  throw_dimension_mismatch(
            "`lower` and `upper` bounds must have the same axes")
        return new{T,N,L,U}(lower, upper)
    end
end

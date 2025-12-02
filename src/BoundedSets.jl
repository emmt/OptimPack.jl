"""

Module `BoundedSets` implements bounded sets which are convex sets of variables with
separable bounds constraints.

"""
module BoundedSets

export BoundedSet

using ..OptimPack
using ..OptimPack:
    @pass,
    ConvexSet,
    LoopStyle,
    LoopStyles,
    LoopStyleDot,
    LoopStyleFor,
    LoopStyleInBounds,
    LoopStyleMap,
    LoopStyleSIMD,
    PlusMinus,
    line_search_limits,
    line_search_step_max,
    project_direction!,
    project_variables!,
    recode!,
    simd_to_for,
    simd_to_inbounds,
    throw_assertion_failed,
    throw_bad_argument,
    throw_dimension_mismatch,
    throw_incompatible_axes,
    unblocked_variables!

using ArrayTools
using TypeUtils
using StructuredArrays
using StructuredArrays: value

# For a single bound value, `nothing` is assumed to be `±Inf`.
const BoundValue{T<:Number} = Union{T,Nothing}

struct BoundedSet{T,N,
                  L<:AbstractArray{T,N},
                  U<:AbstractArray{T,N}} <: ConvexSet
    lower::L
    upper::U
    # Private unsafe constructor.
    global _BoundedSet
    function _BoundedSet(lower::L, upper::U) where {T,N,
                                                    L<:AbstractArray{T,N},
                                                    U<:AbstractArray{T,N}}
        return new{T,N,L,U}(lower, upper)
    end
end

"""
    Ω = BoundedSet(x; lower=nothing, upper=nothing)

Create an object representing separable bounds on the variables `x`. The lower an upper
bounds on the variables can be respectively specified by keywords `lower` and `upper`. A
lower of upper bound may be `nothing` if there is no such bound, a scalar if the same bound
holds for all the variables, or an array of same shape as the variables.

In other words, the object `Ω` represents the convex set:

    Ω = { x ∈ ℝⁿ | lower ≤ x ≤ upper }

where `n` is the number of variables and `≤` holds element-wise.

The object `Ω` is callable: `Ω(x)` overwrites `x` with its projection into the feasible set
defined by the bounds and returns `x`.

"""
function BoundedSet(x::AbstractArray; lower = nothing, upper = nothing)
    lower = lower_bound(x, lower)
    upper = upper_bound(x, upper)
    unsafe_check_bounds(LoopStyle(lower, upper), lower, upper) || throw_bad_argument(
        "lower and upper bounds are not compatible")
    return _BoundedSet(lower, upper)
end

lower_bound(x::AbstractArray{T}, lower::Nothing) where {T} = lower_bound(x, typemin(T))
lower_bound(x::AbstractArray{T}, lower::Number) where {T} =
    UniformArray(convert(T, lower)::T, axes(x))
function lower_bound(x::AbstractArray{T}, lower::AbstractArray) where {T}
    axes(lower) == axes(x) || throw_dimension_mismatch(
        "if specified as an array, lower bounds must have the same axes as the variables")
    return convert_eltype(T, lower)
end

upper_bound(x::AbstractArray{T}, upper::Nothing) where {T} = upper_bound(x, typemax(T))
upper_bound(x::AbstractArray{T}, upper::Number)where {T} =
    UniformArray(convert(T, upper)::T, axes(x))
function upper_bound(x::AbstractArray{T}, upper::AbstractArray) where {T}
    axes(upper) == axes(x) || throw_dimension_mismatch(
        "if specified as an array, upper bounds must have the same axes as the variables")
    return convert_eltype(T, upper)
end

function unsafe_check_bounds(ls::LoopStyle, lower::AbstractArray, upper::AbstractArray)
    return all(x -> ≤(x...), zip(lower, upper))
end
function unsafe_check_bounds(ls::LoopStyle, lower::UniformArray, upper::AbstractArray)
    return all(Base.Fix1(≤, value(lower)), upper)
end
function unsafe_check_bounds(ls::LoopStyle, lower::AbstractArray, upper::UniformArray)
    return all(Base.Fix2(≤, value(upper)), lower)
end
function unsafe_check_bounds(ls::LoopStyle, lower::UniformArray, upper::UniformArray)
    return value(lower) ≤ value(upper)
end

"""
    Ω = BoundedSet(lower, upper)
    Ω = BoundedSet{T}(lower, upper)

Create an object representing separable bounds on the variables. `lower` and `upper` are the
respective lower an upper bounds on the variables. `lower` and `upper` must have the same
axes as the variables `x` to which they apply.

Optional parameter `T` is the element type of the bound values. If not specified, it is
inferred from the element types of `lower` and `upper`. Specified or inferred `T` must be
the same as the element type of the variables.

"""
function BoundedSet(lower::AbstractArray, upper::AbstractArray)
    T = promote_type(eltype(lower), eltype(upper))
    return BoundedSet{T}(lower, upper)
end

function BoundedSet{T}(lower::AbstractArray, upper::AbstractArray) where {T}
    axes(lower) == axes(upper) || throw_dimension_mismatch(
        "lower and upper bounds must have the same axes")
    unsafe_check_bounds(LoopStyle(lower, upper), lower, upper) || throw_bad_argument(
        "lower and upper bounds are not compatible")
    return _BoundedSet(convert_eltype(T, lower), convert_eltype(T, upper))
end

# Make bounded sets callable.
(Ω::BoundedSet)(x::AbstractArray) = project_variables!(x, Ω)

LoopStyles.LoopStyle(::Type{<:BoundedSet{T,N,L,U}}) where {T,N,L,U} = LoopStyle(L, U)

Base.iterate(Ω::BoundedSet, state::Int=0) =
    state == 0 ? (Ω.lower, 1) :
    state == 1 ? (Ω.upper, 2) : nothing

OptimPack.has_constraints(Ω::BoundedSet) =
    is_bounded_below_by(Ω.lower) || is_bounded_above_by(Ω.upper)

is_bounded_below_by(lower::Any) = true
is_bounded_below_by(lower::Nothing) = false
is_bounded_below_by(lower::Number) = lower > typemin(lower)
is_bounded_below_by(lower::UniformArray) = is_bounded_below_by(value(lower))

is_bounded_above_by(upper::Any) = true
is_bounded_above_by(upper::Nothing) = false
is_bounded_above_by(upper::Number) = upper < typemax(upper)
is_bounded_above_by(upper::UniformArray) = is_bounded_above_by(value(upper))

forward(x::Real) = x > zero(x)
forward(x::typeof(+)) = true
forward(x::typeof(-)) = false

#----------------------------------------------------------------------- Project variables -

function OptimPack.project_variables!(ls::LoopStyle, x::AbstractArray, Ω::BoundedSet)
    lower, upper = Ω
    axes(lower) == axes(upper) == axes(x) || throw_incompatible_axes()
    unsafe_project_variables!(ls, x, lower, upper)
    return x
end

function unsafe_project_variables!(::LoopStyleDot, x::AbstractArray,
                                   lower::AbstractArray, upper::AbstractArray)
    bounded_below = is_bounded_below_by(lower)
    bounded_above = is_bounded_above_by(upper)
    if bounded_below && bounded_above
        @. x = clamp(x, lower, upper)
    elseif bounded_below
        @. x = max(x, lower)
    elseif bounded_above
        @. x = min(x, upper)
    end
end

function unsafe_project_variables!(::LoopStyleMap, x::AbstractArray,
                                   lower::AbstractArray, upper::AbstractArray)
    bounded_below = is_bounded_below_by(lower)
    bounded_above = is_bounded_above_by(upper)
    if bounded_below && bounded_above
        map!(clamp, x, x, lower, upper)
    elseif bounded_below
        map!(max, x, x, lower)
    elseif bounded_above
        map!(min, x, x, upper)
    end
end

unsafe_project_variables!_simd() = quote
    function unsafe_project_variables!(::LoopStyleSIMD, x::AbstractArray,
                                       lower::AbstractArray, upper::AbstractArray)
        bounded_below = is_bounded_below_by(lower)
        bounded_above = is_bounded_above_by(upper)
        if bounded_below && bounded_above
            @inbounds @simd for i in eachindex(x, lower, upper)
                x[i] = clamp(x[i], lower[i], upper[i])
            end
        elseif bounded_below
            @inbounds @simd for i in eachindex(x, lower)
                x[i] = max(x[i], lower[i])
            end
        elseif bounded_above
            @inbounds @simd for i in eachindex(x, upper)
                x[i] = min(x[i], upper[i])
            end
        end
        return nothing
    end
end

@eval $(        unsafe_project_variables!_simd())
@eval $(recode!(unsafe_project_variables!_simd(), simd_to_for...))
@eval $(recode!(unsafe_project_variables!_simd(), simd_to_inbounds...))

#--------------------------------------------------------------------- Unblocked variables -

function OptimPack.unblocked_variables!(ls::LoopStyle, u::AbstractArray,
                                        x::AbstractArray, pm::PlusMinus,
                                        d::AbstractArray, Ω::BoundedSet)
    lower = Ω.lower
    upper = Ω.upper
    axes(lower) == axes(upper) == axes(x) || throw_incompatible_axes()
    return unsafe_unblocked_variables!(ls, u, x, pm, d, lower, upper)
end

function unsafe_unblocked_variables!(::LoopStyleDot, u::AbstractArray, x::AbstractArray,
                                     pm::PlusMinus, d::AbstractArray,
                                     lower::AbstractArray, upper::AbstractArray)
    z = zero(eltype(d))
    bounded_below = is_bounded_below_by(lower)
    bounded_above = is_bounded_above_by(upper)
    if bounded_below && bounded_above
        if forward(pm)
            @. u = ((x > lower)|(d > z))&((x < upper)|(d < z))
        else
            @. u = ((x > lower)|(d < z))*((x < upper)|(d > z))
        end
    elseif bounded_below
        if forward(pm)
            @. u = ((x > lower)|(d > z))
        else
            @. u = ((x > lower)|(d < z))
        end
    elseif bounded_above
        if forward(pm)
            @. u = ((x < upper)|(d < z))
        else
            @. u = ((x < upper)|(d > z))
        end
    else
        fill!(u, one(eltype(u)))
    end
    return nothing
end

@inline is_positive(x::Number) = x > zero(x)
@inline is_positive(pm::typeof(+), x::Number) = is_positive(x)
@inline is_positive(pm::typeof(-), x::Number) = is_negative(x)

@inline is_negative(x::Number) = x < zero(x)
@inline is_negative(pm::typeof(+), x::Number) = is_negative(x)
@inline is_negative(pm::typeof(-), x::Number) = is_positive(x)

@inline function is_unblocked_below(x::Number, pm::PlusMinus, d::Number, lower::Number)
    return (x > lower)|is_positive(pm, d)
end

@inline function is_unblocked_below(x::Number, pm::PlusMinus, d::Number, lower::Nothing)
    return true
end

@inline function is_unblocked_above(x::Number, pm::PlusMinus, d::Number, upper::Number)
    return (x < upper)|is_negative(pm, d)
end

@inline function is_unblocked_above(x::Number, pm::PlusMinus, d::Number, upper::Nothing)
    return true
end

@inline function is_unblocked(x::Number, pm::PlusMinus, d::Number,
                              lower::BoundValue, upper::BoundValue)
    return is_unblocked_below(x, pm, d, lower)&is_unblocked_above(x, pm, d, upper)
end

for (dir, sgn) in (:forward => :(+), :reverse => :(-))
    @eval begin
        @inline is_unblocked(::typeof($sgn)) = $(Symbol("is_unblocked_$(dir)"))
        @inline is_unblocked_below(::typeof($sgn)) = $(Symbol("is_unblocked_$(dir)_below"))
        @inline is_unblocked_above(::typeof($sgn)) = $(Symbol("is_unblocked_$(dir)_above"))
        @inline function $(Symbol("is_unblocked_$(dir)"))(x::Number, d::Number,
                                                          lower::BoundValue, upper::BoundValue)
            return is_unblocked(x, $sgn, d, lower, upper)
        end
        @inline function $(Symbol("is_unblocked_$(dir)_below"))(x::Number, d::Number,
                                                                lower::BoundValue)
            return is_unblocked_below(x, $sgn, d, lower)
        end
        @inline function $(Symbol("is_unblocked_$(dir)_above"))(x::Number, d::Number,
                                                                upper::BoundValue)
            return is_unblocked_above(x, $sgn, d, upper)
        end
    end
end

function unsafe_unblocked_variables!(::LoopStyleMap, u::AbstractArray, x::AbstractArray,
                                     pm::PlusMinus, d::AbstractArray,
                                     lower::AbstractArray, upper::AbstractArray)
    z = zero(eltype(d))
    bounded_below = is_bounded_below_by(lower)
    bounded_above = is_bounded_above_by(upper)
    if bounded_below && bounded_above
        map!(is_unblocked(pm), u, x, d, lower, upper)
    elseif bounded_below
        map!(is_unblocked_below(pm), u, x, d, lower)
    elseif bounded_above
        map!(is_unblocked_above(pm), u, x, d, upper)
    else
        fill!(u, one(eltype(u)))
    end
    return nothing
end

unsafe_unblocked_variables!_simd() = quote
    function unsafe_unblocked_variables!(::LoopStyleSIMD, u::AbstractArray, x::AbstractArray,
                                         pm::PlusMinus, d::AbstractArray,
                                         lower::AbstractArray, upper::AbstractArray)
        z = zero(eltype(d))
        bounded_below = is_bounded_below_by(lower)
        bounded_above = is_bounded_above_by(upper)
        if bounded_below && bounded_above
            @inbounds @simd for i in eachindex(u, x, d, lower, upper)
                u[i] = is_unblocked(x[i], pm, d[i], lower[i], upper[i])
            end
        elseif bounded_below
            @inbounds @simd for i in eachindex(u, x, d, lower)
                u[i] = is_unblocked_below(x[i], pm, d[i], lower[i])
            end
        elseif bounded_above
            @inbounds @simd for i in eachindex(u, x, d, upper)
                u[i] = is_unblocked_above(x[i], pm, d[i], upper[i])
            end
        else
            fill!(u, one(eltype(u)))
        end
        return nothing
    end
end

@eval $(        unsafe_unblocked_variables!_simd())
@eval $(recode!(unsafe_unblocked_variables!_simd(), simd_to_for...))
@eval $(recode!(unsafe_unblocked_variables!_simd(), simd_to_inbounds...))

#----------------------------------------------------------------------- Project direction -

function OptimPack.project_direction!(ls::LoopStyle, p::AbstractArray, x::AbstractArray,
                                      pm::PlusMinus, d::AbstractArray, Ω::BoundedSet)
    lower, upper = Ω
    axes(lower) == axes(upper) == axes(p) == axes(d) == axes(x) || throw_incompatible_axes()
    unsafe_project_direction!(ls, p, x, pm, d, lower, upper)
    return p
end

# NOTE In Julia `true` is a the neutral one for the multiplication of numbers while `false`
# is a strong zero for this operation.

function unsafe_project_direction!(::LoopStyleDot, p::AbstractArray, x::AbstractArray,
                                   pm::PlusMinus, d::AbstractArray,
                                   lower::AbstractArray, upper::AbstractArray)
    z = zero(eltype(d))
    bounded_below = is_bounded_below_by(lower)
    bounded_above = is_bounded_above_by(upper)
    if bounded_below && bounded_above
        if forward(pm)
            @. p = (((x > lower)|(d > z))&((x < upper)|(d < z)))*d
        else
            @. p = (((x > lower)|(d < z))&((x < upper)|(d > z)))*d
        end
    elseif bounded_below
        if forward(pm)
            @. p = ((x > lower)|(d > z))*d
        else
            @. p = ((x > lower)|(d < z))*d
        end
    elseif bounded_above
        if forward(pm)
            @. p = ((x < upper)|(d < z))*d
        else
            @. p = ((x < upper)|(d > z))*d
        end
    else
        copy!(p, d)
    end
    return nothing
end

@inline function project_direction(x::Number, pm::PlusMinus, d::Number,
                                   lower::BoundValue, upper::BoundValue)
    return is_unblocked(x, pm, d, lower, upper)*d
end

@inline function project_direction_below(x::Number, pm::PlusMinus, d::Number,
                                         lower::BoundValue)
    return is_unblocked_below(x, pm, d, lower)*d
end

@inline function project_direction_above(x::Number, pm::PlusMinus, d::Number,
                                         upper::BoundValue)
    return is_unblocked_above(x, pm, d, upper)*d
end

for (dir, sgn) in (:forward => :(+), :reverse => :(-))
    @eval begin
        @inline project_direction(::typeof($sgn)) = $(Symbol("project_$(dir)"))
        @inline project_direction_below(::typeof($sgn)) = $(Symbol("project_$(dir)_below"))
        @inline project_direction_above(::typeof($sgn)) = $(Symbol("project_$(dir)_above"))
        @inline function $(Symbol("project_$(dir)"))(x::Number, d::Number,
                                                     lower::BoundValue, upper::BoundValue)
            return project_direction(x, $sgn, d, lower, upper)
        end
        @inline function $(Symbol("project_$(dir)_below"))(x::Number, d::Number,
                                                           lower::BoundValue)
            return project_direction_below(x, $sgn, d, lower)
        end
        @inline function $(Symbol("project_$(dir)_above"))(x::Number, d::Number,
                                                           upper::BoundValue)
            return project_direction_above(x, $sgn, d, upper)
        end
    end
end

function unsafe_project_direction!(::LoopStyleMap, p::AbstractArray, x::AbstractArray,
                                   pm::PlusMinus, d::AbstractArray,
                                   lower::AbstractArray, upper::AbstractArray)
    bounded_below = is_bounded_below_by(lower)
    bounded_above = is_bounded_above_by(upper)
    if bounded_below && bounded_above
        map!(project_direction(pm), p, x, d, lower, upper)
    elseif bounded_below
        map!(project_direction_below(pm), p, x, d, lower)
    elseif bounded_above
        map!(project_direction_above(pm), p, x, d, upper)
    else
        copy!(p, d)
    end
    return nothing
end

unsafe_project_direction!_simd() = quote
    function unsafe_project_direction!(::LoopStyleSIMD, p::AbstractArray, x::AbstractArray,
                                       pm::PlusMinus, d::AbstractArray,
                                       lower::AbstractArray, upper::AbstractArray)
        bounded_below = is_bounded_below_by(lower)
        bounded_above = is_bounded_above_by(upper)
        if bounded_below && bounded_above
            @inbounds @simd for i in eachindex(p, x, d, lower, upper)
                p[i] = project_direction(x[i], pm, d[i], lower[i], upper[i])
            end
        elseif bounded_below
            @inbounds @simd for i in eachindex(p, x, d, lower)
                p[i] = project_direction_below(x[i], pm, d[i], lower[i])
            end
        elseif bounded_above
            @inbounds @simd for i in eachindex(p, x, d, upper)
                p[i] = project_direction_above(x[i], pm, d[i], upper[i])
            end
        else
            copy!(p, d)
        end
        return nothing
    end
end

@eval $(        unsafe_project_direction!_simd())
@eval $(recode!(unsafe_project_direction!_simd(), simd_to_for...))
@eval $(recode!(unsafe_project_direction!_simd(), simd_to_inbounds...))

#---------------------------------------------------------------------- Line-search limits -

function OptimPack.line_search_limits(ls::LoopStyle, x0::AbstractArray, pm::PlusMinus,
                                      d::AbstractArray, Ω::BoundedSet)
    lower = Ω.lower
    upper = Ω.upper
    @assert_same_axes x0 d lower upper
    return unsafe_line_search_limits(ls, x0, pm, d, lower, upper)
end

# Return `α ≥ 0` such that `x + α*d` reaches one of the bounds.
@inline function forward_step_to_bound(x::Number, d::Number, lower::Number, upper::Number)
    return (ifelse(is_negative(d), lower, upper) - x)/d
end

# Return `α ≥ 0` such that `x - α*d` reaches one of the bounds.
@inline function reverse_step_to_bound(x::Number, d::Number, lower::Number, upper::Number)
    return (x - ifelse(is_positive(d), lower, upper))/d
end

# Return `α::T ≥ 0` such that `x + α*d` reaches one of the bounds.
@inline function step_to_bound(::Type{T}, x::Number, ::typeof(+), d::Number,
                               lower::Number, upper::Number) where {T}
    return convert(T, (ifelse(is_negative(d), lower, upper) - x)/d)
end

# Return `α::T ≥ 0` such that `x - α*d` reaches one of the bounds.
@inline function step_to_bound(::Type{T}, x::Number, ::typeof(-), d::Number,
                               lower::Number, upper::Number) where {T}
    return convert(T, (x - ifelse(is_positive(d), lower, upper))/d)
end

# Return `α::T ≥ 0` such that `x + α*d` reaches the lower bound.
@inline function step_to_lower_bound(::Type{T}, x::Number, ::typeof(+), d::Number,
                                     lower::Number) where {T}
    return ifelse(is_negative(d), convert(T, (lower - x)/d), typemax(T))
end

# Return `α::T ≥ 0` such that `x - α*d` reaches the lower bound.
@inline function step_to_lower_bound(::Type{T}, x::Number, ::typeof(-), d::Number,
                                     lower::Number) where {T}
    return ifelse(is_positive(d), convert(T, (x - lower)/d), typemax(T))
end

# Return `α::T ≥ 0` such that `x + α*d` reaches the upper bound.
@inline function step_to_upper_bound(::Type{T}, x::Number, ::typeof(+), d::Number,
                                     upper::Number) where {T}
    return ifelse(is_positive(d), convert(T, (upper - x)/d), typemax(T))
end

# Return `α::T ≥ 0` such that `x - α*d` reaches the upper bound.
@inline function step_to_upper_bound(::Type{T}, x::Number, ::typeof(-), d::Number,
                                     upper::Number) where {T}
    return ifelse(is_negative(d), convert(T, (x - upper)/d), typemax(T))
end

# Callable object to be mapped, return `dnz::Bool` indicating whether `dᵢ` is non-zero and
# `α::T ≥ 0`, the step to the bound.
struct _StepToBound{T<:AbstractFloat,S<:PlusMinus} end
struct _StepToLower{T<:AbstractFloat,S<:PlusMinus} end
struct _StepToUpper{T<:AbstractFloat,S<:PlusMinus} end

for s in (:(+), :(-))
    @eval begin
        @inline function (f::_StepToBound{T,typeof($s)})(x::Number, d::Number,
                                                         lower::Number,
                                                         upper::Number) where {T}
            return (d != zero(d), step_to_bound(T, x, $s, d, lower, upper))
        end
        @inline function (f::_StepToLower{T,typeof($s)})(x::Number, d::Number,
                                                         lower::Number) where {T}
            return (d != zero(d), step_to_lower(T, x, $s, d, lower))
        end
        @inline function (f::_StepToUpper{T,typeof($s)})(x::Number, d::Number,
                                                         upper::Number) where {T}
            return (d != zero(d), step_to_upper(T, x, $s, d, upper))
        end
    end
end

@inline function reduce_limits((αₘᵢₙ,αₘₐₓ)::Tuple{T,T},
                               (dnz,α)::Tuple{Bool,T}) where {T<:AbstractFloat}
    return (ifelse(dnz & (α < αₘᵢₙ), α, αₘᵢₙ),
            ifelse(dnz & (α > αₘₐₓ), α, αₘₐₓ))
end

@inline function reduce_limits((dnz,α)::Tuple{Bool,T},
                               (αₘᵢₙ,αₘₐₓ)::Tuple{T,T}) where {T<:AbstractFloat}
    return (ifelse(dnz & (α < αₘᵢₙ), α, αₘᵢₙ),
            ifelse(dnz & (α > αₘₐₓ), α, αₘₐₓ))
end

@inline function reduce_step_max(αₘₐₓ::T,
                                 (dnz,α)::Tuple{Bool,T}) where {T<:AbstractFloat}
    return ifelse(dnz & (α > αₘₐₓ), α, αₘₐₓ)
end

@inline function reduce_step_max((dnz,α)::Tuple{Bool,T},
                                 αₘₐₓ::T) where {T<:AbstractFloat}
    return ifelse(dnz & (α > αₘₐₓ), α, αₘₐₓ)
end

function unsafe_line_search_limits(::LoopStyleMap, x0::AbstractArray,
                                   pm::PlusMinus, d::AbstractArray,
                                   lower::AbstractArray, upper::AbstractArray)
    T = get_precision(x0, d, lower, upper)
    init = (typemax(T), typemin(T))
    bounded_below = is_bounded_below_by(lower)
    bounded_above = is_bounded_above_by(upper)
    αₘᵢₙ, αₘₐₓ = if bounded_below && bounded_above
        f = _StepToBound{T,typeof(pm)}()
        mapreduce(f, reduce_limits, x0, d, lower, upper; init=init)::Tuple{T,T}
    elseif bounded_below
        f = _StepToLower{T,typeof(pm)}()
        mapreduce(f, reduce_limits, x0, d, lower; init=init)::Tuple{T,T}
    elseif bounded_above
        f = _StepToUpper{T,typeof(pm)}()
        mapreduce(f, reduce_limits, x0, d, upper; init=init)::Tuple{T,T}
    else
        init
    end
    if αₘₐₓ < zero(T)
        # No limit αₘₐₓ found. This may occur is there are no bounds or if the search
        # direction is zero everywhere.
        αₘₐₓ == typemax(T)
    end
    return (αₘᵢₙ, αₘₐₓ)
end

unsafe_line_search_limits_simd() = quote
    # NOTE This code attempt to avoid branching.
    function unsafe_line_search_limits(::LoopStyleSIMD, x0::AbstractArray,
                                       pm::PlusMinus, d::AbstractArray,
                                       lower::AbstractArray, upper::AbstractArray)
        T = get_precision(x0, d, lower, upper)
        αₘᵢₙ = typemax(T)
        αₘₐₓ = typemin(T)
        bounded_below = is_bounded_below_by(lower)
        bounded_above = is_bounded_above_by(upper)
        z = zero(eltype(d))
        if bounded_below && bounded_above
            @inbounds @simd for i in eachindex(x0, d, lower, upper)
                dᵢ = d[i]
                α = step_to_bound(T, x0[i], pm, dᵢ, lower[i], upper[i])
                αₘᵢₙ = ifelse((dᵢ != z)&(α < αₘᵢₙ), α, αₘᵢₙ)
                αₘₐₓ = ifelse((dᵢ != z)&(α > αₘₐₓ), α, αₘₐₓ)
            end
        elseif bounded_below
            @inbounds @simd for i in eachindex(x0, d, lower)
                dᵢ = d[i]
                α = step_to_lower_bound(T, x0[i], pm, dᵢ, lower[i])
                αₘᵢₙ = ifelse((dᵢ != z)&(α < αₘᵢₙ), α, αₘᵢₙ)
                αₘₐₓ = ifelse((dᵢ != z)&(α > αₘₐₓ), α, αₘₐₓ)
            end
        elseif bounded_above
            @inbounds @simd for i in eachindex(x0, d, upper)
                dᵢ = d[i]
                α = step_to_upper_bound(T, x0[i], pm, dᵢ, upper[i])
                αₘᵢₙ = ifelse((dᵢ != z)&(α < αₘᵢₙ), α, αₘᵢₙ)
                αₘₐₓ = ifelse((dᵢ != z)&(α > αₘₐₓ), α, αₘₐₓ)
            end
        end
        if αₘₐₓ < zero(T)
            # No limit αₘₐₓ found. This may occur is there are no bounds or if the search
            # direction is zero everywhere.
            αₘₐₓ == typemax(T)
        end
        return (αₘᵢₙ, αₘₐₓ)
    end
end

@eval $(        unsafe_line_search_limits_simd())
@eval $(recode!(unsafe_line_search_limits_simd(), simd_to_for...))
@eval $(recode!(unsafe_line_search_limits_simd(), simd_to_inbounds...))

#------------------------------------------------------------------- Max. line-search step -

function OptimPack.line_search_step_max(ls::LoopStyle, x0::AbstractArray,
                                        pm::PlusMinus, d::AbstractArray,
                                        Ω::BoundedSet)
    lower = Ω.lower
    upper = Ω.upper
    @assert_same_axes x0 d lower upper
    return unsafe_line_search_step_max(ls, x0, pm, d, lower, upper)
end

function unsafe_line_search_step_max(::LoopStyleMap, x0::AbstractArray,
                                     pm::PlusMinus, d::AbstractArray,
                                     lower::AbstractArray, upper::AbstractArray)
    T = get_precision(x0, d, lower, upper)
    init = typemin(T)
    bounded_below = is_bounded_below_by(lower)
    bounded_above = is_bounded_above_by(upper)
    αₘₐₓ = if bounded_below && bounded_above
        f = _StepToBound{T,typeof(pm)}()
        mapreduce(f, reduce_step_max, x0, d, lower, upper; init=init)::T
    elseif bounded_below
        f = _StepToLower{T,typeof(pm)}()
        mapreduce(f, reduce_step_max, x0, d, lower; init=init)::T
    elseif bounded_above
        f = _StepToUpper{T,typeof(pm)}()
        mapreduce(f, reduce_step_max, x0, d, upper; init=init)::T
    else
        init
    end
    if αₘₐₓ < zero(T)
        # No limit αₘₐₓ found. This may occur is there are no bounds or if the search
        # direction is zero everywhere.
        αₘₐₓ == typemax(T)
    end
    return αₘₐₓ
end

unsafe_line_search_step_max_simd() = quote
    # NOTE This code attempt to avoid branching.
    function unsafe_line_search_step_max(::LoopStyleSIMD, x0::AbstractArray,
                                         pm::PlusMinus, d::AbstractArray,
                                         lower::AbstractArray, upper::AbstractArray)
        T = get_precision(x0, d, lower, upper)
        αₘₐₓ = typemin(T)
        bounded_below = is_bounded_below_by(lower)
        bounded_above = is_bounded_above_by(upper)
        z = zero(eltype(d))
        if bounded_below && bounded_above
            @inbounds @simd for i in eachindex(x0, d, lower, upper)
                dᵢ = d[i]
                α = step_to_bound(T, x0[i], pm, dᵢ, lower[i], upper[i])
                αₘₐₓ = ifelse((dᵢ != z)&(α > αₘₐₓ), α, αₘₐₓ)
            end
        elseif bounded_below
            @inbounds @simd for i in eachindex(x0, d, lower)
                dᵢ = d[i]
                α = step_to_lower_bound(T, x0[i], pm, dᵢ, lower[i])
                αₘₐₓ = ifelse((dᵢ != z)&(α > αₘₐₓ), α, αₘₐₓ)
            end
        elseif bounded_above
            @inbounds @simd for i in eachindex(x0, d, upper)
                dᵢ = d[i]
                α = step_to_upper_bound(T, x0[i], pm, dᵢ, upper[i])
                αₘₐₓ = ifelse((dᵢ != z)&(α > αₘₐₓ), α, αₘₐₓ)
            end
        end
        if αₘₐₓ < zero(T)
            # No limit αₘₐₓ found. This may occur is there are no bounds or if the search
            # direction is zero everywhere.
            αₘₐₓ == typemax(T)
        end
        return αₘₐₓ
    end
end

@eval $(        unsafe_line_search_step_max_simd())
@eval $(recode!(unsafe_line_search_step_max_simd(), simd_to_for...))
@eval $(recode!(unsafe_line_search_step_max_simd(), simd_to_inbounds...))

end # module

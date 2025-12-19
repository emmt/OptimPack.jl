zerofill!(A::AbstractArray) = fill!(A, zero(eltype(A)))

if isdefined(Base, :copy!)
     copy!(dst::AbstractArray, src::AbstractArray) = Base.copy!(dst, src)
else
    function copy!(dst::AbstractVector, src::AbstractVector)
        if !(dst === src)
            length(dst) == length(src) || throw_dimension_mismatch(
                "source and destination have different lengths")
            firstindex(dst) == firstindex(src) || throw_dimension_mismatch(
                "source and destination have different offsets")
            copyto!(dst, src)
        end
        return dst
    end
    function copy!(dst::AbstractArray, src::AbstractArray)
        if !(dst === src)
            axes(dst) == axes(src) || throw_dimension_mismatch(
                "source and destination have different axes")
            copyto!(dst, src)
        end
        return dst
    end
end

"""
    OptimPack.adapt_multiplier_precision(α, x)
    OptimPack.adapt_multiplier_precision(α, typeof(x))
    OptimPack.adapt_multiplier_precision(eltype(x), α)

Adapt the precision of the multiplier `α` to that of the elements of array `x`.

"""
adapt_multiplier_precision(α::Number, x::AbstractArray) =
    adapt_multiplier_precision(α, typeof(x))
adapt_multiplier_precision(α::Number, ::Type{T}) where {T<:AbstractArray} =
    adapt_multiplier_precision(eltype(T), α)

adapt_multiplier_precision(::Type{T}, α::Number) where {T<:Number} =
    adapt_multiplier_precision(get_precision(T), α)

adapt_multiplier_precision(::Type{T}, α::StaticMultiplier) where {T<:AbstractFloat} = α
adapt_multiplier_precision(::Type{T}, α::Number) where {T<:AbstractFloat} =
    # TODO may not be a good idea, throw instead
    convert_floating_point_type((isconcretetype(T) ? T : Float64), α)

# Code conversion rules between different loop styles.
#
# The idea is to code each method for SIMD loops and use these rules to automatically
# produce code for other loop styles.
#
const simd_to_for = (:LoopStyleSIMD => :LoopStyleFor,
                     "@simd" => "@pass",
                     "@inbounds" => "@pass")
#
const simd_to_inbounds = (:LoopStyleSIMD => :LoopStyleInBounds,
                          "@simd" => "@pass")
#
const simd_to_turbo = (:LoopStyleSIMD => :LoopStyleTurbo,
                       "@simd" => "@pass",
                       "@inbounds" => "@turbo")

# Prefer SIMD() over Turbo() loop style.
avoid_turbo(ls::LoopStyle) = ls
avoid_turbo(ls::LoopStyleTurbo) = LoopStyles.SIMD()

# Prefer Map() over any explicit loop style.
avoid_for(ls::LoopStyle) = ls
avoid_for(ls::LoopStyleFor) = LoopStyles.Map()

"""
    OptimPack.one_norm([ls,] x) -> s

Compute the 1-norm of `x` as if it is a *vector of reals*. The dimensions of `x` are ignored
and complexes are considered as pairs of reals. The returned value is real-valued.

If `x` is an array, optional `ls::OptimPack.LoopStyle` is to explicitly choose a loop-style
for the computations. If not specified, it is automatically inferred from the type of `x`.

See also [`OptimPack.two_norm`](@ref), [`OptimPack.sup_norm`](@ref),
[`OptimPack.inner`](@ref), and [`OptimPack.LoopStyle`](@ref).

"""
@inline one_norm(x::Number) = abs(x)
@inline one_norm(x::Complex) = one_norm(x.re) + one_norm(x.im)
@inline one_norm(x::AbstractQuantity) = one_norm(ustrip(x))*unit(x)

# Automatically infer loop-style.
one_norm(x::AbstractArray) = one_norm(LoopStyle(x), x)

# Fallback method.
one_norm(::LoopStyle, x::AbstractArray) = one_norm(LoopStyles.Map(), x)

function one_norm(::LoopStyleMap, x::AbstractArray)
    return mapreduce(one_norm, +, x)
end

# This function returns the code with SIMD loops. The returned expression can be
# modified by the caller.
one_norm_simd() = quote
    function one_norm(::LoopStyleSIMD, x::AbstractArray)
        T = typeof(one_norm(zero(eltype(x)))*1)
        s = zero(T)
        @inbounds @simd for i in eachindex(x)
            s += convert(T, one_norm(x[i]))
        end
        return s
    end
end

@eval $(        one_norm_simd())
@eval $(recode!(one_norm_simd(), simd_to_for...))
@eval $(recode!(one_norm_simd(), simd_to_inbounds...))

"""
    OptimPack.two_norm([ls,] x) -> s

Compute the 2-norm (a.k.a. Euclidean norm) of `x` as if it is a *vector of reals*. The
dimensions of `x` are ignored and complexes are considered as pairs of reals. The returned
value is real-valued.

If `x` is an array, optional `ls::OptimPack.LoopStyle` is to explicitly choose a loop-style
for the computations. If not specified, it is automatically inferred from the type of `x`.

See also [`OptimPack.one_norm`](@ref), [`OptimPack.sup_norm`](@ref),
[`OptimPack.inner`](@ref), and [`OptimPack.LoopStyle`](@ref).

"""
@inline two_norm(x::Number) = abs(x)
@inline two_norm(x::Complex) = sqrt(abs2(x.re) + abs2(x.im))
@inline two_norm(x::AbstractQuantity) = two_norm(ustrip(x))*unit(x)

# Automatically infer loop-style.
two_norm(x::AbstractArray) = two_norm(LoopStyle(x), x)

# Fallback method.
two_norm(::LoopStyle, x::AbstractArray) = two_norm(LoopStyles.Map(), x)

function two_norm(::LoopStyleMap, x::AbstractArray)
    return sqrt(mapreduce(abs2, +, x))
end

two_norm_simd() = quote
    function two_norm(::LoopStyleSIMD, x::AbstractArray)
        T = typeof(abs2(zero(eltype(x)))*1)
        s = zero(T)
        @inbounds @simd for i in eachindex(x)
            s += convert(T, abs2(x[i]))
        end
        return sqrt(s)
    end
end

@eval $(        two_norm_simd())
@eval $(recode!(two_norm_simd(), simd_to_for...))
@eval $(recode!(two_norm_simd(), simd_to_inbounds...))

"""
    OptimPack.sup_norm([ls,] x) -> s

Compute the sup-norm of `x` as if it is a *vector of reals*. The dimensions of `x` are
ignored and complexes are considered as pairs of reals. The returned value is real-valued.

If `x` is an array, optional `ls::OptimPack.LoopStyle` is to explicitly choose a loop-style
for the computations. If not specified, it is automatically inferred from the type of `x`.

See also https://en.wikipedia.org/wiki/Uniform_norm, [`OptimPack.one_norm`](@ref),
[`OptimPack.sup_norm`](@ref), [`OptimPack.inner`](@ref), and [`OptimPack.LoopStyle`](@ref).

"""
@inline sup_norm(x::Number) = abs(x)
@inline sup_norm(x::Complex) = max(sup_norm(x.re), sup_norm(x.im))
@inline sup_norm(x::AbstractQuantity) = sup_norm(ustrip(x))*unit(x)

# Automatically infer loop-style.
sup_norm(x::AbstractArray) = sup_norm(avoid_turbo(LoopStyle(x)), x)

# Fallback method.
sup_norm(::LoopStyle, x::AbstractArray) = sup_norm(LoopStyles.Map(), x)

function sup_norm(::LoopStyleMap, x::AbstractArray)
    return mapreduce(sup_norm, max, x)
end

sup_norm_simd() = quote
    function sup_norm(::LoopStyleSIMD, x::AbstractArray)
        T = typeof(sup_norm(zero(eltype(x))))
        s = zero(T)
        @inbounds @simd for i in eachindex(x)
            r = convert(T, sup_norm(x[i]))
            s = max(s, r) #(isnan(r)|(r > s)) ? r : s
        end
        return s
    end
end

@eval $(        sup_norm_simd())
@eval $(recode!(sup_norm_simd(), simd_to_for...))
@eval $(recode!(sup_norm_simd(), simd_to_inbounds...))

@inline fast_min(x::Number, y::Number) = fast_min(promote(x, y)...)
@inline fast_min(x::T, y::T) where {T<:Number} = (isnan(x)|(x < y)) ? x : y

@inline fast_max(x::Number, y::Number) = fast_max(promote(x, y)...)
@inline fast_max(x::T, y::T) where {T<:Number} = (isnan(x)|(x > y)) ? x : y

"""
    OptimPack.inner([ls,] x, y) -> s
    OptimPack.inner([ls,] w, x, y) -> s

Return the inner product (a.k.a. scalar product) of `x` and `y` considering them as *simple
vectors of reals*. That is, `x` and `y` must have the same shapes and complexes are
considered as pairs of reals. The result is real-valued and computed as:

    s = Σᵢ xᵢ*yᵢ

where `xᵢ` and `yᵢ` denote the `i`-the real values in `x` and `y` respectively.

If `w`, an array of same shape as `x` and `y` is specified, the result is:

    s = Σᵢ wᵢ*xᵢ*yᵢ

Optional `ls::OptimPack.LoopStyle` is to explicitly choose a loop-style for the
computations. If not specified, it is automatically inferred from the types of `x` and `y`.

See also [`OptimPack.one_norm`](@ref), [`OptimPack.two_norm`](@ref),
[`OptimPack.sup_norm`](@ref), and [`OptimPack.LoopStyle`](@ref).

"""
function inner end

# Inner product of 2 reals or 2 complexes. NOTE Mixing complexes and reals in inner product
# is purposely not supported.
@inline inner(x::Real, y::Real) = x*y
@inline inner(x::Complex, y::Complex) = inner(x.re, y.re) + inner(x.im, y.im)

# Inner product of 1 quantity and 1 number.
@inline inner(x::Number, y::AbstractQuantity) = inner(y, x)
@inline inner(x::AbstractQuantity, y::Number) = inner(ustrip(x), y)*unit(x)

# Inner product of 2 quantities.
@inline inner(x::AbstractQuantity, y::AbstractQuantity) = inner(ustrip(x), ustrip(y))*(unit(x)*unit(y))

# Triple inner product of 3 reals or 3 complexes. NOTE Mixing complexes and reals in inner
# product is purposely not supported.
@inline inner(w::Real, x::Real, y::Real) = w*x*y
@inline inner(w::Complex, x::Complex, y::Complex) = inner(w.re, x.re, y.re) + inner(w.im, x.im, y.im)

# Triple inner product with 1 quantity and 2 numbers.
@inline inner(w::Number, x::Number, y::AbstractQuantity) = inner(y, w, x)
@inline inner(w::Number, x::AbstractQuantity, y::Number) = inner(x, w, y)
@inline inner(w::AbstractQuantity, x::Number, y::Number) = inner(ustrip(w), x, y)*unit(w)

# Triple inner product with 2 quantities and 1 number.
@inline inner(w::AbstractQuantity, x::AbstractQuantity, y::Number) = inner(y, w, x)
@inline inner(w::AbstractQuantity, x::Number, y::AbstractQuantity) = inner(x, w, y)
@inline inner(w::Number, x::AbstractQuantity, y::AbstractQuantity) =
    inner(w, ustrip(x), ustrip(y))*(unit(x)*unit(y))

# Triple inner product with 3 quantities.
@inline inner(w::AbstractQuantity, x::AbstractQuantity, y::AbstractQuantity) =
    inner(ustrip(w), ustrip(x), ustrip(y))*(unit(w)*unit(x)*unit(y))

# Automatically infer loop-style.
function inner(x::AbstractArray, y::AbstractArray)
    return inner(avoid_turbo(LoopStyle(x, y)), x, y)
end
function inner(w::AbstractArray, x::AbstractArray, y::AbstractArray)
    return inner(avoid_turbo(LoopStyle(w, x, y)), w, x, y)
end

# Check axes and call "unsafe" implementation.
function inner(ls::LoopStyle, x::AbstractArray, y::AbstractArray)
    axes(x) == axes(y) || throw_incompatible_axes()
    return unsafe_inner(ls, x, y)
end
function inner(ls::LoopStyle, w::AbstractArray, x::AbstractArray, y::AbstractArray)
    axes(w) == axes(x) == axes(y) || throw_incompatible_axes()
    return unsafe_inner(ls, w, x, y)
end

# Fallback methods.
function unsafe_inner(::LoopStyle, x::AbstractArray, y::AbstractArray)
    unsafe_inner(LoopStyles.Map(), x, y)
end
function unsafe_inner(::LoopStyle, w::AbstractArray, x::AbstractArray, y::AbstractArray)
    unsafe_inner(LoopStyles.Map(), w, x, y)
end

# Inner products based on `mapreduce`.
function unsafe_inner(::LoopStyleMap, x::AbstractArray, y::AbstractArray)
    return mapreduce(inner, +, x, y)
end
function unsafe_inner(::LoopStyleMap, w::AbstractArray, x::AbstractArray, y::AbstractArray)
    return mapreduce(inner, +, w, x, y)
end

unsafe_inner_simd() = quote
    function unsafe_inner(::LoopStyleSIMD, x::AbstractArray, y::AbstractArray)
        T = typeof(inner(zero(eltype(x)), zero(eltype(y)))*1)
        s = zero(T)
        @inbounds @simd for i in eachindex(x, y)
            s += convert(T, inner(x[i], y[i]))
        end
        return s
    end
    function unsafe_inner(::LoopStyleSIMD, w::AbstractArray, x::AbstractArray, y::AbstractArray)
        T = typeof(inner(zero(eltype(w)), zero(eltype(x)), zero(eltype(y)))*1)
        s = zero(T)
        @inbounds @simd for i in eachindex(w, x, y)
            s += convert(T, inner(w[i], x[i], y[i]))
        end
        return s
    end
end

@eval $(        unsafe_inner_simd())
@eval $(recode!(unsafe_inner_simd(), simd_to_for...))
@eval $(recode!(unsafe_inner_simd(), simd_to_inbounds...))

"""
    OptimPack.scale!([ls,] x, α::Number) -> x

In-place scaling of `x` by `α`. Do `x[i] *= α` for all valid indices `i` and optimizing the
computational burden depending on the specific values of the multiplier `α`. The convention
is that `α` is considered as a *strong zero* if `iszero(α)` holds.

Optional `ls::OptimPack.LoopStyle` is to explicitly choose a loop-style for the
computations. If not specified, it is automatically inferred from the type of `x`.

See also [`OptimPack.xpby!`](@ref), [`OptimPack.axpby!`](@ref), and
[`OptimPack.LoopStyle`](@ref).

"""
function scale!(x::AbstractArray, α::Number)
    # Automatically infer loop-style.
    return scale!(avoid_turbo(LoopStyle(x)), x, α)
end

function scale!(ls::LoopStyle, x::AbstractArray, α::Number)
    unsafe_scale!(Val(:alpha), ls, x, adapt_multiplier_precision(α, x), x)
    return x
end

"""
    OptimPack.scale!([ls,] dst, α, x) -> dst

Compute `dst[i] = α*x[i]` for all valid indices `i` and optimizing the computational burden
depending on the specific values of the multiplier `α`. The convention is that `α` is
considered as a *strong zero* if `iszero(α)` holds.

Optional `ls::OptimPack.LoopStyle` is to explicitly choose a loop-style for the
computations. If not specified, it is automatically inferred from the types of `dst` and
`x`.

See also [`OptimPack.xpby!`](@ref), [`OptimPack.axpby!`](@ref), and
[`OptimPack.LoopStyle`](@ref).

"""
function scale!(dst::AbstractArray, α::Number, x::AbstractArray)
    # Automatically infer loop-style.
    return scale!(avoid_turbo(LoopStyle(dst, x)), dst, α, x)
end

function scale!(ls::LoopStyle, dst::AbstractArray, α::Number, x::AbstractArray)
    axes(dst) == axes(x) || throw_incompatible_axes()
    unsafe_scale!(Val(:alpha), ls, dst, adapt_multiplier_precision(α, x), x)
    return dst
end

function unsafe_scale!(::Val{:alpha}, ls::LoopStyle, dst::AbstractArray,
                       α::Number, x::AbstractArray)
    @dispatch_on_value α unsafe_scale!(ls, dst, α, x)
    return nothing
end

# Fallback method.
unsafe_scale!(::LoopStyle, dst::AbstractArray, α::Number, x::AbstractArray) =
    unsafe_scale!(LoopStyles.Map(), dst, α, x)

function unsafe_scale!(::LoopStyleMap, dst::AbstractArray,
                       α::Number, x::AbstractArray)
    map!(Base.Fix1(*, α), dst, x)
    return nothing
end

function unsafe_scale!(::LoopStyleDot, dst::AbstractArray,
                       α::Number, x::AbstractArray)
    @. dst = α*x
    return nothing
end

unsafe_scale!_simd() = quote
    function unsafe_scale!(::LoopStyleSIMD, dst::AbstractArray,
                           α::Number, x::AbstractArray)
        @inbounds @simd for i in eachindex(dst, x)
            dst[i] = α*x[i]
        end
        return nothing
    end
end

@eval $(        unsafe_scale!_simd())
@eval $(recode!(unsafe_scale!_simd(), simd_to_for...))
@eval $(recode!(unsafe_scale!_simd(), simd_to_inbounds...))


"""
    OptimPack.mult!([ls,] dst, x, y) -> dst

Overwrite `dst` with the element-wise product of `x` and `y`.

Optional `ls::OptimPack.LoopStyle` is to explicitly choose a loop-style for the
computations. If not specified, it is automatically inferred from the types of `x` and `y`.

See also [`OptimPack.scale!`](@ref) and [`OptimPack.LoopStyle`](@ref).

"""
function mult!(dst::AbstractArray, x::AbstractArray, y::AbstractArray)
    return mult!(LoopStyle(dst, x, y), dst, x, y)
end
function mult!(ls::LoopStyle, dst::AbstractArray, x::AbstractArray, y::AbstractArray)
    # TODO arguments should be real-valued?
    axes(dst) == axes(x) == axes(y) || throw_incompatible_axes()
    unsafe_mult!(ls, dst, x, y)
    return dst
end
function unsafe_mult!(::LoopStyleMap, dst::AbstractArray, x::AbstractArray, y::AbstractArray)
    map!(*, dst, x, y)
    return nothing
end
function unsafe_mult!(::LoopStyleDot, dst::AbstractArray, x::AbstractArray, y::AbstractArray)
    @. dst = x*y
    return nothing
end
unsafe_mult!_simd() = quote
    function unsafe_mult!(::LoopStyleSIMD, dst::AbstractArray, x::AbstractArray, y::AbstractArray)
        @inbounds @simd for i in eachindex(dst, x, y)
            dst[i] = x[i]*y[i]
        end
        return nothing
    end
end

@eval $(        unsafe_mult!_simd())
@eval $(recode!(unsafe_mult!_simd(), simd_to_for...))
@eval $(recode!(unsafe_mult!_simd(), simd_to_inbounds...))

"""
    OptimPack.xpby!([ls,] dst, x, β, y) -> dst

Compute `dst[i] = x[i] + β*y[i]` for all valid indices `i` and optimizing the computational
burden depending on the specific values of the multiplier `β`. The convention is that `β` is
considered as a *strong zero* if `iszero(β)` holds.

Optional `ls::OptimPack.LoopStyle` is to explicitly choose a loop-style for the
computations. If not specified, it is automatically inferred from the types of `dst`, `x`,
and `y`.

See also [`OptimPack.axpby!`](@ref) [`OptimPack.scale!`](@ref), and
[`OptimPack.LoopStyle`](@ref).

"""
function xpby!(dst::AbstractArray, x::AbstractArray, β::Number, y::AbstractArray)
    # Automatically infer loop-style.
    return xpby!(avoid_turbo(LoopStyle(dst, x, y)), dst, x, β, y)
end

function xpby!(ls::LoopStyle, dst::AbstractArray, x::AbstractArray,
               β::Number, y::AbstractArray)
    axes(dst) == axes(x) == axes(y) || throw_incompatible_axes()
    unsafe_xpby!(Val(:beta), ls, dst, x, adapt_multiplier_precision(β, y), y)
    return dst
end
function unsafe_xpby!(::Val{:beta}, ls::LoopStyle, dst::AbstractArray, x::AbstractArray,
                      β::Number, y::AbstractArray)
    @dispatch_on_value β unsafe_xpby!(ls, dst, x, β, y)
    return nothing
end

# Fallback method.
function unsafe_xpby!(::LoopStyle, dst::AbstractArray, x::AbstractArray,
                      β::Number, y::AbstractArray)
    unsafe_xpby!(LoopStyles.Map(), dst, x, β, y)
end

# Simple structure to avoid a closure to implement `x + β*y`
struct XPBY{B<:Number}
    β::B
end
(f::XPBY)(x, y) = x + f.β*y

function unsafe_xpby!(::LoopStyleMap, dst::AbstractArray, x::AbstractArray,
                      β::Number, y::AbstractArray)
    map!(XPBY(β), dst, x, y)
    return nothing
end

function unsafe_xpby!(::LoopStyleDot, dst::AbstractArray, x::AbstractArray,
                      β::Number, y::AbstractArray)
    @. dst = x + β*y
    return nothing
end

unsafe_xpby!_simd() = quote
    function unsafe_xpby!(::LoopStyleSIMD,  dst::AbstractArray, x::AbstractArray,
                          β::Number, y::AbstractArray)
        @inbounds @simd for i in eachindex(dst, x, y)
            dst[i] = x[i] + β*y[i]
        end
        return nothing
    end
end

@eval $(        unsafe_xpby!_simd())
@eval $(recode!(unsafe_xpby!_simd(), simd_to_for...))
@eval $(recode!(unsafe_xpby!_simd(), simd_to_inbounds...))

"""
    OptimPack.axpby!([ls,] dst, α, x, β, y) -> dst

Compute `dst[i] = α*x[i] + β*y[i]` for all valid indices `i` and optimizing the
computational burden depending on the specific values of the multipliers `α` and `β`. The
convention is that `α` is considered as a *strong zero* if `iszero(α)` holds and similarly
for `β`.

Optional `ls::OptimPack.LoopStyle` is to explicitly choose a loop-style for the
computations. If not specified, it is automatically inferred from the types of `dst`, `x`,
and `y`.

See also [`OptimPack.xpby!`](@ref) and  [`OptimPack.scale!`](@ref).

"""
function axpby!(dst::AbstractArray,
                α::Number, x::AbstractArray,
                β::Number, y::AbstractArray)
    # Automatically infer loop-style.
    return axpby!(avoid_turbo(LoopStyle(dst, x, y)), dst, α, x, β, y)
end

function axpby!(ls::LoopStyle, dst::AbstractArray,
                α::Number, x::AbstractArray,
                β::Number, y::AbstractArray)
    axes(dst) == axes(x) == axes(y) || throw_incompatible_axes()
    unsafe_axpby!(Val(:alpha_beta), ls, dst,
                  adapt_multiplier_precision(α, x), x,
                  adapt_multiplier_precision(β, y), y)
    return dst
end

function unsafe_axpby!(::Val{:alpha_beta}, ls::LoopStyle, dst::AbstractArray,
                       α::Number, x::AbstractArray,
                       β::Number, y::AbstractArray)
    @dispatch_on_value α unsafe_axpby!(Val(:beta), ls, dst, α, x, β, y)
    return nothing
end

function unsafe_axpby!(::Val{:beta}, ls::LoopStyle, dst::AbstractArray,
                       α::Number, x::AbstractArray,
                       β::Number, y::AbstractArray)
    @dispatch_on_value β unsafe_axpby!(ls, dst, α, x, β, y)
    return nothing
end

function unsafe_axpby!(::Val{:alpha}, ls::LoopStyle, dst::AbstractArray,
                       α::Number, x::AbstractArray,
                       β::Number, y::AbstractArray)
    @dispatch_on_value α unsafe_axpby!(ls, dst, α, x, β, y)
    return nothing
end

# Fallback method.
function unsafe_axpby!(::LoopStyle, dst::AbstractArray,
                       α::Number, x::AbstractArray,
                       β::Number, y::AbstractArray)
    unsafe_axpby!(LoopStyles.Map(), dst, α, x, β, y)
end

# Simple structure to avoid a closure to implement `α*x + β*y`
struct AXPBY{A<:Number,B<:Number}
    α::A
    β::B
end
(f::AXPBY)(x, y) = f.α*x + f.β*y

function unsafe_axpby!(::LoopStyleMap, dst::AbstractArray,
                       α::Number, x::AbstractArray,
                       β::Number, y::AbstractArray)
    map!(AXPBY(α, β), dst, x, y)
    return nothing
end

function unsafe_axpby!(::LoopStyleDot, dst::AbstractArray,
                       α::Number, x::AbstractArray,
                       β::Number, y::AbstractArray)
    @. dst = α*x + β*y
    return nothing
end

unsafe_axpby!_simd() = quote
    function unsafe_axpby!(::LoopStyleSIMD,  dst::AbstractArray,
                           α::Number, x::AbstractArray,
                           β::Number, y::AbstractArray)
        @inbounds @simd for i in eachindex(dst, x, y)
            dst[i] = α*x[i] + β*y[i]
        end
        return nothing
    end
end

@eval $(        unsafe_axpby!_simd())
@eval $(recode!(unsafe_axpby!_simd(), simd_to_for...))
@eval $(recode!(unsafe_axpby!_simd(), simd_to_inbounds...))

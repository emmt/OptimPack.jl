"""

The `LoopStyles` module implement traits to represent the most efficient way to loop over
elements of arguments.

"""
module LoopStyles

# TODO Test on GPU arrays.
#
# TODO `@turbo` does not work well with neutral numbers.
#
# NOTE `@turbo` implies `@fastmath` and thus `isnan` is always false.
#
# TODO `one_norm` is 2× slower than BLAS.
#
# NOTE `two_norm` is 8× faster than BLAS (because we do not take care of avoiding overflows).
#
# NOTE `sup_norm` with any for-loop style is 10× slower with `fast_max` than BLAS, `Map`, or
#      `Dot`. With `max` and `Turbo`, `sup_norm` is nearly 3 times faster than BLAS but does
#      not correctly account for NaN. Hence, with prefer `SIMD` over `Turbo` for the
#      sup-norm which is 25% faster than BLAS.
#
# NOTE `@. expr` allocates temporaries sometimes.
#
# NOTE `map(...)` is quite efficient (owing to its generality).

# Only export abstract indexing types.
export
    LoopStyle,
    LoopStyleDot,
    LoopStyleFor,
    LoopStyleGPU,
    LoopStyleInBounds,
    LoopStyleMap,
    LoopStyleSIMD,
    LoopStyleScalar,
    LoopStyleTurbo

# Concrete indexing types are public but not exported.
using TypeUtils: @public
@public Dot
@public For
@public GPU
@public InBounds
@public Map
@public SIMD
@public Scalar
@public Turbo

@public one_norm two_norm sup_norm inner scale! xpby! axpby!
@public recode recode!  @pass

using StructuredArrays

# Singleton type to indicate undefined result.
struct Undefined end

# Abstract indexing methods for the hierarchy and for methods signatures.
abstract type LoopStyle end

"""
    LoopStyleScalar <: LoopStyle

Abstract type of traits representing scalars.

"""
abstract type LoopStyleScalar <: LoopStyle end

"""
    LoopStyleGPU <: LoopStyle

Abstract type of traits representing indexing of GPU arrays.

"""
abstract type LoopStyleGPU <: LoopStyle end

"""
    LoopStyleMap <: LoopStyle

Abstract type of traits for arguments that can only be iterated. Expressions involving such
arguments can only rely on `map`, `mapreduce`, etc.

"""
abstract type LoopStyleMap <: LoopStyle end

"""
    LoopStyleDot <: LoopStyle

Abstract type of traits for arguments that can be efficiently indexed by `@. ...` macro
calls.

"""
abstract type LoopStyleDot <: LoopStyle end

"""
    LoopStyleFor <: LoopStyle

Abstract type of traits for arguments that can be efficiently indexed in simple `for ...`
loops (with bounds checking). This should probably only used for debug.

"""
abstract type LoopStyleFor <: LoopStyle end

"""
    LoopStyleInBounds <: LoopStyleFor

Abstract type of traits for arguments that can be efficiently indexed in
`@inbounds for ...` loops.

"""
abstract type LoopStyleInBounds <: LoopStyleFor end

"""
    LoopStyleSIMD <: LoopStyleInBounds

Abstract type of traits for arguments that can be efficiently indexed in
`@inbounds @simd for ...` loops.

"""
abstract type LoopStyleSIMD <: LoopStyleInBounds end

"""
    LoopStyleTurbo <: LoopStyleSIMD

Abstract type of traits representing arrays that can be efficiently indexed in
`@turbo for ...` loops.

"""
abstract type LoopStyleTurbo <: LoopStyleSIMD end

"""
    LoopStyle(args...)

Return a singleton representing how arguments `args...` can be jointly indexed efficiently.

The hierarchy of indexing types is:

```
LoopStyle (abstract)
 │
 ├╴LoopStyleScalar (abstract) -> LoopStyles.Scalar
 │
 ├╴LoopStyleMap (abstract) -> LoopStyles.Map
 │
 ├╴LoopStyleDot (abstract) -> LoopStyles.Dot
 │
 ├╴LoopStyleGPU (abstract) -> LoopStyles.GPU{API}
 │
 ╰╴LoopStyleFor (abstract) -> LoopStyles.For
    │
    ╰╴LoopStyleInBounds (abstract) -> LoopStyles.InBounds
       │
       ╰╴LoopStyleSIMD (abstract) -> LoopStyles.SIMD
          │
          ╰╴LoopStyleTurbo (abstract) -> LoopStyles.Turbo
```

The idea is to use abstract types for method signatures so that a fallback method naturally
follows from the hierarchy.

"""
LoopStyle(x::Any) = LoopStyle(typeof(x))
LoopStyle(x::LoopStyle) = x

@inline LoopStyle(x, y, z...) = LoopStyle(LoopStyle(x, y), z...)
@inline LoopStyle(x, y) = LoopStyle(LoopStyle(x), LoopStyle(y))
@inline LoopStyle(x::LoopStyle, y::LoopStyle) =
    joint_styles_result(x, y, joint_styles(x, y), joint_styles(y, x))

@inline joint_styles_result(x, y, a::LoopStyle, b::Undefined) = a
@inline joint_styles_result(x, y, a::Undefined, b::LoopStyle) = b
@inline joint_styles_result(x, y, a::T, b::T) where {T<:LoopStyle} = a
@noinline joint_styles_result(x, y, ::Any, ::Any) = throw_bad_argument(
    "joint indexing for `$(typeof(x))` and `$(typeof(y))` is not supported")

"""
    LoopStyles.Scalar()

Trait representing scalars.

"""
struct Scalar <: LoopStyleScalar end

"""
    LoopStyles.Map()

Trait representing the most basic indexing capability: that of iterators.

"""
struct Map <: LoopStyleMap end

"""
    LoopStyles.Dot()

Trait indicating that `@. ...` macro calls can be used to evaluate element-wise expressions.

"""
struct Dot <: LoopStyleDot end

"""
    LoopStyles.For()

Trait indicating that simple `for ...` loops can be used to iterate over arguments.

"""
struct For <: LoopStyleFor end

"""
    LoopStyles.InBounds()

Trait indicating that `@inbounds for ...` loops can be used to iterate over arguments.

"""
struct InBounds <: LoopStyleInBounds end

"""
    LoopStyles.SIMD()

Trait indicating that `@inbounds @simd for ...` loops can be used to iterate over arguments.

"""
struct SIMD <: LoopStyleSIMD end

"""
    LoopStyles.Turbo()

Trait indicating that `@turbo for ...` loops can be used to iterate over arguments.

"""
struct Turbo <: LoopStyleTurbo end

"""
    LoopStyles.GPU{API}()

Trait indicating that arguments are GPU arrays. `A` is a symbolic name representing the API
of GPU arrays: e.g. `:CUDA`, `:oneAPI`, `:AMDGPU`, or `:Metal`. Arrays in CPU memory and GPU
arrays with different API cannot be mixed together.

"""
struct GPU{API} <: LoopStyleGPU end

LoopStyle(::Type{<:Number}) = Scalar()
LoopStyle(::Type{<:Any}) = Map()
LoopStyle(::Type{<:AbstractArray}) = InBounds()
LoopStyle(::Type{<:StridedArray}) = SIMD()
LoopStyle(::Type{<:Array}) = Turbo()
LoopStyle(::Type{<:AbstractUniformArray}) = Scalar()

"""
    LoopStyles.joint_styles(x::LoopStyle, y::LoopStyle)

Return most suitable trait for jointly iterating over arguments with loop styles `x` and
`y`. It is not necessary to extend the method for arguments `(x,y)` and `(y,x)`.

"""
joint_styles(x::LoopStyle, y::LoopStyle) = Undefined()

# Special rules for scalars.
joint_styles(x::LoopStyle, y::Scalar) = x

# GPU arrays can only be combined with GPU arrays.
joint_styles(x::GPU{API}, y::GPU{API}) where {API} = GPU{API}()
joint_styles(x::GPU, y::LoopStyle) = throw_bad_argument(
    "cannot mix GPU array(s) and other array(s)")
joint_styles(x::GPU{API₁}, y::GPU{API₂}) where {API₁,API₂} = throw_bad_argument(
    "cannot mix GPU array(s) with API $(API₁) and $(API₂)")

# Encode rules for other loop styles with the first operand being the most efficient and the
# second operand being the least efficient.
let styles = [:Turbo, :SIMD, :InBounds, :Dot, :For, :Map]
    for i in 1:length(styles)
        for j in i:length(styles)
            @eval joint_styles(x::$(styles[i]), y::$(styles[j])) = $(styles[j])()
        end
    end
end

@noinline throw_bad_argument(msg::AbstractString) = throw(ArgumentError(msg))
@noinline throw_bad_argument(args...) = throw_bad_argument(string(args...))

end # module

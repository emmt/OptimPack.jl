"""

Module `OptimPack.Powell` provides some of the derivative-free optimization algorithms by
M.J.D. Powell: BOBYQA, COBYLA, and NEYUOA.

"""
module Powell

export
    Bobyqa, bobyqa, bobyqa!,
    Cobyla, cobyla, cobyla!,
    Newuoa, newuoa, newuoa!,
    issuccess

using TypeUtils: @public
@public configure! solve!

using LinearAlgebra: issuccess

using Neutrals
using StructuredArrays

const rho_reduction = 1e-5
default_maxevals(n::Integer) = 5 + 30*Int(n)::Int
default_npt(n::Integer) = 2*Int(n)::Int + 1

copy_array(::Type{T}, A::AbstractArray) where {T} = copyto!(Array{T}(undef, size(A)), A)

dense_array(A::AbstractArray{T}) where {T} = dense_array(T, A)
dense_array(::Type{T}, A::DenseArray{T}) where {T} = A
dense_array(::Type{T}, A::AbstractArray) where {T} = copy_array(T, A)

# Call as dense_array(Cdouble, check_scale(arg, dims)) to ensure that a regular dense array
# is given.
function check_scale(val::typeof(𝟙), shape::Dims)
    return Array{typeof(𝟙)}(undef, shape)
end
function check_scale(val::Real, dims::Dims)
    check_scale(Bool, val) || throw_bad_argument("`scale` must be finite and strictly positive")
    return UniformArray{Cdouble}(val, dims)
end
function check_scale(arr::AbstractArray, shape::Dims)
    size(arr) == shape || throw_dimension_mismatch(
        "`scale` must have the same shape as the variables")
    check_scale(Bool, arr) || throw_bad_argument(
        "all elements of `scale` must be finite and strictly positive")
    return dense_array(Cdouble, arr)
end

check_scale(::Type{Bool}, val::Real) = isfinite(val) & (val > zero(val))
function check_scale(::Type{Bool}, arr::AbstractArray)
    flag = true
    @inbounds for i in eachindex(arr)
        flag &= check_scale(Bool, arr[i])
    end
    return flag
end

include("Bobyqa.jl")
import .Bobyqa: bobyqa, bobyqa!

include("Cobyla.jl")
import .Cobyla: cobyla, cobyla!

include("Newuoa.jl")
import .Newuoa: newuoa, newuoa!

end # module Powell

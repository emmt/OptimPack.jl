"""

`WrappedArrays` provide dense multi-dimensional arrays wrapped over a raw pointer.

"""
module WrappedArrays

export
    MutableWrappedArray,
    WrappedArray,
    rewrap!

"""
    WrappedArray{T,N}(ptr, dims...)

Create a `N`-dimensional *immutable wrapped array* with element type `T`, shape `dims...`,
and whose first element is stored at address given by `ptr`.

The number `N` of dimension can be inferred from `dims...` and may be omitted.

Element type `T` must be a bit-type. If `ptr` is of type `Ptr{T}`, the parameter `T` of the
constructor may be omitted. If the element type `T` is specified, `ptr` must be either
`Ptr{T}` or `Ptr{Nothing}`.

!!! warning
    It is the caller's responsibility to ensure that `ptr` be valid for the entire lifetime
    of `A`, that the memory at `ptr` is large enough to store all elements of `A`, and to
    free the memory at `ptr` when it is no longer needed.

See [`MutableWrappedArray`](@ref) for for building a mutable wrapped array for which a
finalizer can be declared to release the memory at `ptr`.

"""
struct WrappedArray{T,N} <: DenseArray{T,N}
    ptr::Ptr{T}
    dims::Dims{N}
    function WrappedArray(ptr::Ptr{T}, dims::Dims{N}) where {T,N}
        isbitstype(T) || throw(ArgumentError("element type is not a \"plain data\" type"))
        len = check_shape(dims)
        return new{T,N}(ptr, dims)
    end
end

"""
    MutableWrappedArray{T,N}(ptr, dims...)

Create a `N`-dimensional *mutable wrapped array* with element type `T`, shape `dims...`, and
whose first element is stored at address given by `ptr`.

The number `N` of dimension can be inferred from `dims...` and may be omitted.

Element type `T` must be a bit-type. If `ptr` is of type `Ptr{T}`, the parameter `T` of the
constructor may be omitted. If the element type `T` is specified, `ptr` must be either
`Ptr{T}` or `Ptr{Nothing}`.

!!! warning
    It is the caller's responsibility to ensure that `ptr` be valid for the entire lifetime
    of `A`, that the memory at `ptr` is large enough to store all elements of `A`, and to
    free the memory at `ptr` when it is no longer needed. A finalizer can be declared for
    `A` for that latter purpose.

See [`rewrap!`](@ref) for changing the storage address and, if there are no needs for
changing this address nor to declare a finalizer, [`WrappedArray`](@ref) for building an
immutable wrapped array.

"""
mutable struct MutableWrappedArray{T,N} <: DenseArray{T,N}
    ptr::Ptr{T}
    dims::Dims{N}
    function MutableWrappedArray(ptr::Ptr{T}, dims::Dims{N}) where {T,N}
        isbitstype(T) || throw(ArgumentError("element type is not a \"plain data\" type"))
        len = check_shape(dims)
        return new{T,N}(ptr, dims)
    end
end

const AnyWrappedArray{T,N} = Union{WrappedArray{T,N},MutableWrappedArray{T,N}}

for constr in (:WrappedArray, :MutableWrappedArray)
    @eval begin
        $constr{T,N}(ptr::Ptr, dims::Integer...) where {T,N} = $constr{T,N}(ptr, dims)
        $constr{T,N}(ptr::Ptr, dims::NTuple{N,Integer}) where {T,N} = $constr{T}(ptr, dims)
        $constr{T,N}(ptr::Ptr, dims::NTuple{L,Integer}) where {T,N,L} =
            throw(DimensionMismatch("incompatible number of dimensions"))

        $constr{T}(ptr::Ptr, dims::Integer...) where {T} = $constr{T}(ptr, dims)
        $constr{T}(ptr::Ptr{T}, dims::NTuple{N,Integer}) where {T,N} = $constr(ptr, dims)
        $constr{T}(ptr::Ptr{Nothing}, dims::NTuple{N,Integer}) where {T,N} =
            $constr(Ptr{T}(ptr), dims)
        $constr{T}(ptr::Ptr{S}, dims::NTuple{N,Integer}) where {S,T,N} =
            throw(ArgumentError("incompatible element and pointer types"))

        $constr(ptr::Ptr, dims::Integer...) = $constr(ptr, dims)
        $constr(ptr::Ptr{T}, dims::NTuple{N,Integer}) where {T,N} =
            $constr(ptr, convert(Dims{N}, dims))
    end
end

Base.length(A::AnyWrappedArray) = prod(size(A))
Base.size(A::AnyWrappedArray) = getfield(A, :dims)
Base.axes(A::AnyWrappedArray) = map(Base.OneTo, size(A))
Base.IndexStyle(::Type{<:WrappedArray}) = IndexLinear()
@inline function Base.getindex(A::AnyWrappedArray, i::Int)
    @boundscheck checkbounds(A, i)
    return GC.@preserve A unsafe_load(pointer(A), i)
end
@inline function Base.setindex!(A::AnyWrappedArray, x, i::Int)
    @boundscheck checkbounds(A, i)
    GC.@preserve A unsafe_store!(pointer(A), x, i)
    return A
end
@inline Base.pointer(A::AnyWrappedArray) = getfield(A, :ptr)
Base.unsafe_convert(::Type{Ptr{T}}, A::AnyWrappedArray{T}) where {T} = pointer(A)

"""
    rewrap!(A::MutableWrappedArray, ptr) -> A
    rewrap!(A::MutableWrappedArray, ptr, nbytes) -> A

Re-wrap a mutable wrapped array `A` to use `ptr` as the address of its first element. If
`nbytes` is provided, it is assumed to be the number of bytes available at `ptr` and it is
asserted that this is large enough to store all the elements of `A`.

Neither the element type nor the shape of `A` can be modified.

!!! warning
    It is the caller's responsibility to ensure that `ptr` be valid for the entire lifetime
    of `A` and to free the memory at `ptr` when it is no longer needed.

"""
function rewrap!(A::MutableWrappedArray{T}, ptr::Ptr{<:Union{Nothing,T}}) where {T}
    setfield!(A, :ptr, Ptr{T}(ptr))
    return A
end

function rewrap(A::MutableWrappedArray{T}, ptr::Ptr{<:Union{Nothing,T}},
                nbytes::Integer) where {T}
    nbytes ≥ length(A)*sizeof(T) || throw(ArgumentError("number of bytes too small"))
    return rewrap!(A, ptr)
end

function check_shape(dims::NTuple{N,Integer}) where {N}
    len = 1
    for dim in dims
        dim ≥ zero(dim) || throw(ArgumentError("dimension(s) must be non-negative"))
        len *= Int(dim)::Int
    end
    return len
end

end # module

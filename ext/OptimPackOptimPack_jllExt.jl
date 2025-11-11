module OptimPackOptimPack_jllExt

if isdefined(Base, :get_extension)
    using OptimPack, OptimPack_jll
    # FIXME using OptimPack: Powell
else
    using ..OptimPack, ..OptimPack_jll
    # FIXME using ..OptimPack: Powell
end

using .OptimPack:
    Powell,
    configure!,
    iterate!,
    restart!,
    solve!,
    throw_assertion_failed,
    throw_bad_argument,
    throw_dimension_mismatch

using .Powell:
    Bobyqa,
    Cobyla,
    Newuoa,
    Powell,
    check_scale,
    copy_array,
    default_maxevals,
    default_npt,
    dense_array,
    rho_reduction

using Neutrals, StructuredArrays
import LinearAlgebra

if !isdefined(@__MODULE__, :Memory)
    const Memory{T} = Vector{T}
end

include("LibOptimPack.jl")
using .LibOptimPack

include("WrappedArrays.jl")
using .WrappedArrays

struct ObjectiveFunction{F,N,M}
    func::F         # objective function
    x_dims::Dims{N} # shape of variables
    c_dims::Dims{M} # shape of constraints
    ObjectiveFunction(f::F, x_dims::Dims{N}, c_dims::Dims{M}) where {F,N,M} =
        new{F,N,M}(f, x_dims, c_dims)
end

ObjectiveFunction(f, x_dims::Dims) = ObjectiveFunction(f, x_dims, ())

struct Null{T} end
Base.pointer(x::Null{T}) where {T} = Ptr{T}(0)
Base.unsafe_convert(::Type{Ptr{T}}, x::Null{T}) where {T} = Ptr{T}(0)
Base.unsafe_convert(::Type{Ptr{Nothing}}, x::Null) = Ptr{Nothing}(0)

capitalize(s::Symbol) = Symbol(capitalize(String(s)))
function capitalize(s::AbstractString)
    start = firstindex(s)
    stop = lastindex(s)
    return start > stop ? "" :
        @inbounds uppercase(s[start])*SubString(s, nextind(s, start):stop)
end

for alg in (:bobyqa, :cobyla, :newuoa)
    mod = capitalize(alg)
    @eval begin
        # `summary(status)` returns the corresponding textual reason.
        Base.summary(status::$mod.Status) =
            unsafe_string($(Symbol(alg,"_reason"))(status))

        # `issuccess(status)` yields whether algorithm was successful.
        LinearAlgebra.issuccess(status::$mod.Status) =
            status == $mod.SUCCESS
    end
    alg === :bobyqa && continue
    type = Symbol(alg,"_context")
    @eval begin
        # Automatically convert context to a pointer of the correct type throwing an error
        # if the pointer is NULL.
        function Base.unsafe_convert(::Type{Ptr{$type}}, ctx::$mod.Context)
            ptr = getfield(ctx, :ptr)
            ptr == C_NULL && throw_bad_argument("invalid NULL pointer")
            return Ptr{$type}(ptr)
        end

        # Finalizer for the context.
        function finalize(ctx::$mod.Context)
            ptr = getfield(ctx, :ptr)
            if ptr != C_NULL
                setfield!(ctx, :ptr, C_NULL)
                $(Symbol(alg,"_delete"))(ptr)
            end
            return nothing
        end
    end
end

#---------------------------------------------------------------------------------- COBYLA -

Base.propertynames(::Cobyla.Context) = (
    :constr, :evals, :fbest, :rho, :scale, :shape, :status)

Base.getproperty(ctx::Cobyla.Context, key::Symbol) =
    key === :evals  ? Int(cobyla_get_nevals(ctx))::Int :
    key === :rho    ? cobyla_get_rho(ctx) :
    key === :shape  ? size(getfield(ctx, :scale)) :
    key === :status ? cobyla_get_status(ctx) :
    key !== :ptr    ? getfield(ctx, key) : KeyError(key)

# NOTE `c` and `scale` are shared by the context if they have the correct types.
function Cobyla.Context(x::DenseArray{Cdouble}, c::DenseArray{Cdouble};
                        rhobeg::Real, rhoend::Real = rho_reduction*rhobeg,
                        maximize::Bool = false,
                        scale::Union{Real,AbstractArray{<:Real}} = 𝟙,
                        maxevals::Integer = default_maxevals(n),
                        verbose::Integer = 0)
    scale = check_scale(scale, size(x))
    n = length(x)
    m = length(c)
    ptr = cobyla_create(n, m, rhobeg, rhoend, verbose, maxevals)
    ptr == C_NULL && error(
        Libc.errno() == Libc.ENOMEM ? "insufficient memory" : "invalid parameter(s)")
    ctx = Cobyla.Context(Ptr{Cvoid}(ptr), NaN, scale, c, maximize)
    finalizer(finalize, ctx)
    return ctx
end

"""
    OptimPack.solve!(ctx::Cobyla.Context, fc, x; kwds...) -> ctx, x

Solve a constrained optimization problem by the COBYLA algorithm. `ctx` has been created by
[`Cobyla.Context`](@ref), `fc` computes the objective function and the constraints, and `x`
specifies the initial variables overwritten with the solution on return.

If `fc` itself does not need allocations, no further allocations are required.

"""
function OptimPack.solve!(ctx::Cobyla.Context, fc, x::DenseArray{Cdouble};
                          maximize::Bool = ctx.maximize)
    axes(x) == axes(ctx.scale) || throw_dimension_mismatch(
        "variables and context have different shapes")
    setfield!(ctx, :maximize, maximize)
    c = ctx.constr
    m = length(c)
    status = restart!(ctx)
    while status == Cobyla.ITERATE
        if m > 0
            fx = fc(x, c)
            status = iterate!(ctx, fx, x, c)
        else
            fx = fc(x)
            status = iterate!(ctx, fx, x)
        end
    end
    return ctx, x
end

function OptimPack.restart!(ctx::Cobyla.Context)
    setfield!(ctx, :fbest, typemax(ctx.fbest))
    return cobyla_restart(ctx)
end

function OptimPack.iterate!(ctx::Cobyla.Context, fx::Number,
                            x::DenseArray{Cdouble},
                            c::Union{Null{Cdouble},DenseArray{Cdouble}} = Null{Cdouble}())
    fbest = ctx.fbest
    fx = oftype(fbest, fx)
    (ctx.maximize ? fx > fbest : fx < fbest) && setfield!(ctx, :fbest, fx)
    scl = ctx.scale
    scale!(x, scl)
    status = cobyla_iterate(ctx, (ctx.maximize ? -fx : fx), x, c)
    unscale!(x, scl)
    return status
end

function Cobyla.cobyla(fc, x0::AbstractArray{<:Real}, dims::Integer...; kwds...)
    return Cobyla.cobyla(fc, x0, dims; kwds...)
end

function Cobyla.cobyla(fc, x0::AbstractArray{<:Real},
                       dims::Tuple{Vararg{Integer}}; kwds...)
    x = copy_array(Cdouble, x0)
    c = Array{Cdouble}(undef, dims)
    return Cobyla.cobyla!(fc, x, c; kwds...)
end

function Cobyla.cobyla!(fc, x::DenseArray{Cdouble}, c::DenseArray{Cdouble}; kwds...)
    ctx = Cobyla.Context(x, c; kwds...)
    @assert ctx.constr === c
    solve!(ctx, fc, x)
    return ctx.status, x, c, ctx.fbest, ctx.evals, ctx.rho
end

#---------------------------------------------------------------------------------- BOBYQA -

function OptimPack.solve!(ctx::Bobyqa.Context, f, x::DenseArray{Cdouble}; kwds...)
    isempty(kwds) || configure!(ctx; kwds...)
    axes(x) == axes(ctx.scale) || throw_dimension_mismatch(
        "variables and context have different shapes")
    status = GC.@preserve ctx bobyqa_optimize(
        length(x), ctx.npt, ctx.maximize, objfun_f[], Ref(ObjectiveFunction(f, size(x))),
        x, ctx.lower, ctx.upper, unsafe_scale_pointer(ctx), ctx.rhobeg, ctx.rhoend,
        ctx.verbose, ctx.maxevals, getfield(ctx, :work))
    setfield!(ctx, :status, status)
    return ctx, x
end

function Bobyqa.bobyqa(f, x0::AbstractVector{<:Real}; kwds...)
    x = copy_array(Cdouble, x0)
    return Bobyqa.bobyqa!(f, x; kwds...)
end

function Bobyqa.bobyqa!(f, x::DenseVector{Cdouble}; kwds...)
    ctx = Bobyqa.Context(x; kwds...)
    solve!(ctx, f, x)
    return ctx.status, x, ctx.fbest, ctx.evals
end

#---------------------------------------------------------------------------------- NEWUOA -

Base.propertynames(::Newuoa.Context) = (
    :fbest, :evals, :maximize, :npt, :rho, :scale, :shape, :status)

Base.getproperty(ctx::Newuoa.Context, key::Symbol) =
    key === :evals  ? Int(newuoa_get_nevals(ctx))::Int :
    key === :rho    ? newuoa_get_rho(ctx) :
    key === :shape  ? size(getfield(ctx, :scale)) :
    key === :status ? newuoa_get_status(ctx) :
    key !== :ptr    ? getfield(ctx, key) : KeyError(key)

function Newuoa.Context(x::AbstractArray;
                        rhobeg::Real, rhoend::Real = rho_reduction*rhobeg,
                        maximize::Bool = false,
                        scale::Union{Real,AbstractArray{<:Real}} = 𝟙,
                        npt::Integer = default_npt(length(x)),
                        maxevals::Integer = default_maxevals(length(x)),
                        verbose::Integer = 0)
    scale = check_scale(scale, size(x))
    n = length(x)
    ptr = newuoa_create(n, npt, rhobeg, rhoend, verbose, maxevals)
    ptr == C_NULL && error(
        Libc.errno() == Libc.ENOMEM ? "insufficient memory" : "invalid parameter(s)")
    ctx = Newuoa.Context(Ptr{Cvoid}(ptr), NaN, scale, npt, maximize)
    finalizer(finalize, ctx)
    return ctx
end

"""
    OptimPack.solve!(ctx::Newuoa.Context, f, x; kwds...) -> ctx, x

Solve an unconstrained optimization problem by the NEWUOA algorithm. `ctx` has been created
by [`Newuoa.Context`](@ref), `f` is the objective function, and `x` specifies the initial
variables overwritten with the solution on return.

If `f` itself does not need allocations, no further allocations are required.

"""
function OptimPack.solve!(ctx::Newuoa.Context, f, x::DenseArray{Cdouble};
                          maximize::Bool = ctx.maximize)
    axes(x) == axes(ctx.scale) || throw_dimension_mismatch(
        "variables and context have incompatible shapes")
    setfield!(ctx, :maximize, maximize)
    status = restart!(ctx)
    while status == Newuoa.ITERATE
        fx = f(x)
        status = iterate!(ctx, fx, x)
    end
    return ctx, x
end

function OptimPack.restart!(ctx::Newuoa.Context)
    setfield!(ctx, :fbest, typemax(ctx.fbest))
    return newuoa_restart(ctx)
end

function OptimPack.iterate!(ctx::Newuoa.Context, fx::Number, x::AbstractArray)
    fbest = ctx.fbest
    fx = oftype(fbest, fx)
    (ctx.maximize ? fx > fbest : fx < fbest) && setfield!(ctx, :fbest, fx)
    scl = ctx.scale
    scale!(x, scl)
    status = newuoa_iterate(ctx, (ctx.maximize ? -fx : fx), x)
    unscale!(x, scl)
    return status
end

function Base.reset(ctx::Newuoa.Context)
    newuoa_restart(ctx)
    return ctx
end

function Newuoa.newuoa(f, x0::AbstractArray{<:Real}; kwds...)
    x = copy_array(Cdouble, x0)
    return Newuoa.newuoa!(f, x; kwds...)
end

function Newuoa.newuoa!(f, x::DenseArray{Cdouble}; kwds...)
    ctx = Newuoa.Context(x; kwds...)
    solve!(ctx, f, x)
    return ctx.status, x, ctx.fbest, ctx.evals, ctx.rho
end

#------------------------------------------------------------------------------- Utilities -

all_ones(A::AbstractUniformArray) = isone(StructuredArrays.value(A))
all_ones(A::AbstractArray{<:Neutral}) = eltype(A) === 𝟙
function all_ones(A::AbstractArray)
    flag = true
    @inbounds @simd for i in eachindex(A)
        flag &= isone(A[i])
    end
    return flag
end

# Return a pointer, context must be protected form being garbage collected.
function unsafe_scale_pointer(ctx)
    scale = ctx.scale
    return all_ones(scale) ? Ptr{Cdouble}(0) : pointer(scale)
end

function scale!(x::AbstractArray{T,N}, scl::AbstractArray{typeof(𝟙),N}) where {T,N}
    return nothing
end

function unscale!(x::AbstractArray{T,N}, scl::AbstractArray{typeof(𝟙),N}) where {T,N}
    return nothing
end

function scale!(x::AbstractArray{T,N}, scl::AbstractArray{S,N}) where {S,T,N}
    @inbounds @simd for i in eachindex(x, scl)
        x[i] /= scl[i]
    end
    return nothing
end

function unscale!(x::AbstractArray{T,N}, scl::AbstractArray{S,N}) where {S,T,N}
    @inbounds @simd for i in eachindex(x, scl)
        x[i] *= scl[i]
    end
    return nothing
end

#-------------------------------------------------------- Objective function callable in C -

function objfun(n::Integer, x_ptr::Ptr{T}, ref_ptr::Ptr{Cvoid}) where {T}
    # Extract objective function and dispatch on its type.
    ref = unsafe_pointer_to_objref(ref_ptr)::Ref{<:ObjectiveFunction{<:Any,<:Any,0}}
    return objfun(ref[], x_ptr, n)
end

function objfun(obj::ObjectiveFunction{F,N,0}, x_ptr::Ptr{T}, n::Integer) where {F,T,N}
    x = WrappedArray(x_ptr, obj.x_dims)::WrappedArray{T,N}
    return convert(Cdouble, obj.func(x))
end

function objfun(n::Integer, m::Integer, x_ptr::Ptr{T}, c_ptr::Ptr{T},
                ref_ptr::Ptr{Cvoid}) where {T}
    # Extract objective function and dispatch on its type.
    ref = unsafe_pointer_to_objref(ref_ptr)::Ref{<:ObjectiveFunction}
    return objfun(ref[], x_ptr, n, c_ptr, m)
end

function objfun(obj::ObjectiveFunction{F,N,M},
                x_ptr::Ptr{T}, n::Integer,
                c_ptr::Ptr{T}, m::Integer) where {F,T,N,M}
    x = WrappedArray(x_ptr, obj.x_dims)::WrappedArray{T,N}
    c = WrappedArray(c_ptr, obj.c_dims)::WrappedArray{T,M}
    return convert(Cdouble, obj.func(x, c))
end

const objfun_f  = Ref{Ptr{Cvoid}}() # unconstrained objective function
const objfun_fc = Ref{Ptr{Cvoid}}() # objective function with constraints

function __init__()
    objfun_f[] = @cfunction(
        objfun, Cdouble, (Cptrdiff_t, Ptr{Cdouble}, Ptr{Cvoid}))
    objfun_fc[] = @cfunction(
        objfun, Cdouble, (Cptrdiff_t, Cptrdiff_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
end

end # module

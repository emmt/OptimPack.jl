"""

Module `OptimPack.Powell` provides some of derivative-free optimization algorithms by
M.J.D. Powell: BOBYQA, COBYLA, and NEYUOA.

"""
module Powell

export
    # Specific methods.
    Bobyqa, bobyqa, bobyqa!,
    Cobyla, cobyla, cobyla!,
    Newuoa, newuoa, newuoa!,

    # Common API.
    configure!,
    issuccess,
    maximize, maximize!,
    minimize, minimize!,
    optimize, optimize!

using LinearAlgebra
using OptimPack_jll

const rho_reduction = 1e-5

abstract type AbstractContext end # FIXME global to OptimPack?
@inline Base.getproperty(ctx::AbstractContext, sym::Symbol) = _getproperty(ctx, Val{sym}())
_getproperty(ctx::AbstractContext, ::Val{sym}) where {sym} = throw(KeyError(sym))

abstract type AbstractStatus end # FIXME global to OptimPack?
Base.propertynames(status::AbstractStatus) = (:code, :reason)
@inline Base.getproperty(status::AbstractStatus, sym::Symbol) = _getproperty(status, Val{sym}())
_getproperty(status::AbstractStatus, ::Val{:code}) = getfield(status, :code)
_getproperty(status::AbstractStatus, ::Val{sym}) where {sym} = throw(KeyError(sym))

Base.:(==)(a::T, b::T) where {T<:AbstractStatus} = (a.code == b.code)
Base.:(==)(a::AbstractStatus, b::AbstractStatus) = false

function optimize! end # FIXME global to OptimPack?
function configure! end # FIXME global to OptimPack?

function fix_n(n::Integer)
    n ≥ 2 || throw(ArgumentError("number of variables must be ≥ 2"))
    return Int(n)
end

function fix_m(m::Integer)
    m ≥ zero(m) || throw(ArgumentError("number of constraints must be nonnegative"))
    return Int(m)
end

function fix_npt(npt::Integer, n::Int)
    npt_min = n + 2
    npt_max = div((n + 2)*(n + 1), 2)
    npt_min ≤ npt ≤ npt_max || throw(ArgumentError(
        "`$npt_min ≤ npt ≤ $npt_max` does not hold, got `npt = $npt`"))
    return Int(npt)
end

function fix_rho_parameters(rhobeg::Real, rhoend::Real)
    (isfinite(rhobeg) && rhobeg > zero(rhobeg)) || throw(ArgumentError(
        "`rhobeg` must be finite and positive"))
    (isfinite(rhoend) && rhoend ≤ rhobeg) || throw(ArgumentError(
        "`rhoend` must be finite and smaller than `rhobeg`"))
    return Cdouble(rhobeg), Cdouble(rhoend)
end

fix_verbose(verbose::Integer) = Int(verbose)

function fix_maxeval(maxeval::Integer)
    maxeval ≥ zero(maxeval) || throw(ArgumentError("`maxeval` must be nonnegative"))
    return Int(maxeval)
end

copy_variables(x::AbstractVector) = copy_variables!(Vector{Cdouble}(undef, length(x)), x)
function copy_variables!(dst::AbstractVector, src::AbstractVector)
    if dst !== src
        n = length(dst)
        length(src) == n || throw(DimensionMismatch(
            "source and destination vectors have different lengths"))
        copyto!(dst, firstindex(dst), src, firstindex(src), n)
    end
    return dst
end

copy_or_fill!(dst::AbstractVector, val::Real) = fill!(dst, val)
copy_or_fill!(dst::AbstractVector, src::AbstractVector) = copy_variables!(dst, src)

function unsafe_scale_pointer(ctx::AbstractContext)
    scale = ctx.scale
    return all(isone, scale) ? Ptr{Cdouble}(0) : pointer(scale)
end

check_scale(scale::Real, n::Int) = nothing
check_scale(scale::AbstractVector{<:Real}, n::Int) =
    scale isa Real || scale === ctx.scale || length(scale) == n || throw(DimensionMismatch(
        "`scale` must be a scalar or a vector of $n reals"))
    nothing

# Set scaling factors.
function set_scale!(ctx::AbstractContext, scale::Union{Real,AbstractVector{<:Real}}, n::Int)
    ctx_scale = getfield(ctx, :scale)
    old_n = length(ctx_scale)
    old_n == n || resize!(ctx_scale, n)
    if scale === ctx_scale
        # No new scaling factors have been specified, reset all scaling factors to 1 if
        # the number of variables has changed.
        n == old_n || fill!(ctx_scale, 1.0)
    elseif scale isa Real
        fill!(ctx_scale, Cdouble(scale))
    else
        copy_variables!(ctx_scale, scale)
    end
    return nothing
end

include("newuoa.jl")
import .Newuoa: newuoa, newuoa!

include("cobyla.jl")
import .Cobyla: cobyla, cobyla!

include("bobyqa.jl")
import .Bobyqa: bobyqa, bobyqa!

for mdl in (:Cobyla, :Bobyqa, :Newuoa)
    @eval begin
        optimize(ctx::$mdl.Context, f::Function, x0::AbstractVector{<:Real}; kwds...) =
            optimize!(ctx, f, copy_variables(x0); kwds...)

        minimize(ctx::$mdl.Context, f::Function, x0::AbstractVector{<:Real}; kwds...) =
            minimize!(ctx, f, copy_variables(x0); kwds...)

        minimize!(ctx::$mdl.Context, f::Function, x::DenseVector{Cdouble}; kwds...) =
            optimize!(ctx, f, x; kwds..., maximize=false)

        maximize(ctx::$mdl.Context, f::Function, x0::AbstractVector{<:Real}; kwds...) =
            maximize!(ctx, f, copy_variables(x0); kwds...)

        maximize!(ctx::$mdl.Context, f::Function, x::DenseVector{Cdouble}; kwds...) =
            optimize!(ctx, f, x; kwds..., maximize=true)

        LinearAlgebra.issuccess(status::$mdl.Status) = status === $mdl.SUCCESS
        LinearAlgebra.issuccess(ctx::$mdl.Context) = issuccess(ctx.status)

    end
end

end # module Powell

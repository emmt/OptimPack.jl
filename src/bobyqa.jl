"""

Module `Bobyqa` provides Mike Powell's BOBYQA algorithm to minimize a function of many
variables subject to bound constraints.

"""
module Bobyqa

export
    # Specific symbols.
    bobyqa, bobyqa!,

    # Common API.
    configure!,
    issuccess,
    maximize, maximize!,
    minimize, minimize!,
    optimize, optimize!

using ..Powell
using ..Powell: rho_reduction

const libbobyqa = Powell.OptimPack_jll.libbobyqa

"""
    bobyqa(f, x0; rhobeg, rhoend=$rho_reduction*rhobeg, lower=-Inf, upper=+Inf,
           scale=1.0, npt=2*length(x0)+1, verbose=0, maxeval=30*length(x0),
           maximize=false) -> status, x, fx

runs *BOBYQA* algorithm to find the variables `x` which solve the bound constrained
problem:

    min f(x)  subject to  lower ≤ x ≤ upper

where `x` is a vector of variables that has `n ≥ 2` components `f(x)` is an objective
function, `lower` and `upper` are bounds on the variables. The algorithm employs quadratic
approximations to the objective which interpolate the objective function at a number of
points specified by keyword `npt`, the value `npt = 2n + 1` being recommended. The
parameter `rho` controls the size of the trust region and it is reduced automatically from
`rhobeg` to `rhoend`.

The initial variables `x0` must be feasible, that is:

    lower ≤ x0 ≤ upper

must hold on entry.

The method returns a 3-tuple: `status` indicates whether the algorithm was successful, `x`
is the final value of the variables and `fx = f(x)` is the objective function at `x`.
Normally, `status` should be `Bobyqa.SUCCESS` or equivalently `issuccess(status)` should
be true; otherwise, `status.reason` yields a textual explanation about the failure.


## Precision and scaling of variables

The proper scaling of the variables is important for the success of the algorithm and the
optional `scale` keyword should be specified if the typical precision is not the same for
all variables. If `scale` is vector of same size as the variables `x`, then `scale[i]*rho`
(with `rho` the trust region radius) is the size of the trust region for the `i`-th
variable. Hence, `scale[i]*rhobeg` is the size of the initial range to explore for the
`i`-th variable while `scale[i]*rhoend` is an estimation of the precision of the `i`-th
variable for the solution. Keyword `scale` may also be set to a positive scalar to assume
the same scaling factor for all variables. In any case, all scaling factors must be finite
and strictly positive. If keyword `scale` is not specified, a unit scaling for all the
variables is assumed.

An error occurs if any of the differences `upper[i] - lower[i]` is less than
`2*rhobeg*scale[i]`.


## Keywords

The following keywords are available:

* `rhobeg` and `rhoend` are the initial and final sizes of the trust region. `0 < rhoend ≤
  rhobeg` must hold.

* `lower` and `upper` are the bounds on the variables.

* `scale` gives the typical magnitudes of the variables. It is a scalar (to have the same
  scale for all variables) or a vector with as many elements as `x`. The scaling factors
  must be all finite and strictly positive.

* `verbose` is the amount of printing.

* `maxeval` is the maximum number of calls to the objective function.

* `npt` is the number of points to use for the quadratic approximation of the objective
  function. The default setting is the recommended value: `npt = 2n + 1` with `n =
  length(x)` the number of variables.

* `maximize` specifies whether to attempt to maximize the objective function; otherwise,
  the algorithm attempts to minimize the objective function.


## Related methods

* [`bobyqa!`](@ref) is the in-place version of the algorithm.

* See [`Bobyqa.Context`](@ref) for another way to apply the algorithm which is useful to
  retrieve more information about the algorithm state or to solve several similar problems
  while avoiding allocations and thus the overhead of the garbage collector.


## References

The algorithm is described in:

* M.J.D. Powell, "The BOBYQA algorithm for bound constrained optimization without
  derivatives", Technical Report NA2009/06 of the Department of Applied Mathematics and
  Theoretical Physics, Cambridge, England (2009).

"""
bobyqa(f::Function, x0::AbstractVector{<:Real}; kwds...) =
    bobyqa!(f, Powell.copy_variables(x0); kwds...)

"""
    bobyqa!(f, x; kwds...) -> status, x, fx

runs the in-place version of **BOBYQA** algorithm to find the variables `x` which solve a
bound constrained optimization problem. On entry, argument `x` specifies the initial
variables; on return, `x` is overwritten by the solution. See [`bobyqa`](@ref) for a
description of the algorithm and of the available keywords.

"""
function bobyqa!(f::Function, x::DenseVector{Cdouble}; kwds...)
    ctx = Context(length(x); kwds...)
    optimize!(ctx, f, x)
    return ctx.status, x, ctx.fbest
end

# Status returned by BOBYQA.
struct Status <: Powell.AbstractStatus
    code::Cint
end

# Possible status values returned by BOBYQA (as defined in C header file `bobyqa.h`,
# except `INITIAL_ITERATE` which is used to mark un-started algorithm).
const INITIAL_ITERATE      = Status( 1)
const SUCCESS              = Status( 0)
const BAD_NVARS            = Status(-1)
const BAD_NPT              = Status(-2)
const BAD_RHO_RANGE        = Status(-3)
const BAD_SCALING          = Status(-4)
const TOO_CLOSE            = Status(-5)
const ROUNDING_ERRORS      = Status(-6)
const TOO_MANY_EVALUATIONS = Status(-7)
const STEP_FAILED          = Status(-8)

# Get a textual explanation of the status returned by BOBYQA.
Powell._getproperty(status::Status, ::Val{:reason}) =
    status == INITIAL_ITERATE      ? "Algorithm not yet started" :
    status == SUCCESS              ? "Algorithm converged" :
    status == BAD_NVARS            ? "Bad number of variables" :
    status == BAD_NPT              ? "NPT is not in the required interval" :
    status == BAD_RHO_RANGE        ? "Bad trust region radius parameters" :
    status == BAD_SCALING          ? "Bad scaling factor(s)" :
    status == TOO_CLOSE            ? "Insufficient space between the bounds" :
    status == ROUNDING_ERRORS      ? "Too much cancellation in a denominator" :
    status == TOO_MANY_EVALUATIONS ? "Maximum number of function evaluations exceeded" :
    status == STEP_FAILED          ? "A trust region step has failed to reduce Q" :
    "Unknown BOUNDS status: `Bobyqa.Status($(status.code))"

"""
    ctx = Bobyqa.Context(n; npt=2n+1, rhobeg, rhoend=$rho_reduction*rhobeg,
                         scale=1.0, lower=-Inf, upper=+Inf,
                         maxeval=30n, verbose=0, maximize=false)

creates a new context for solving an optimization problem with NEWUOA.

The context `ctx` can be used to solve one or several problems with COBYLA (although one
problem at a time) by:

    ctx, x = optimize(ctx, f, x0; kwds...)

where `x0` specifies the initial variables (`x0` is left unchanged) and `kwds...` are any
keywords accepted by the context constructor to change settings at will. Another
possibility is to call the in-place version:

    ctx, x = optimize!(ctx, f, x; kwds...)

where `x` contains the initial variables on entry and the solution on return. The input
context `ctx` is returned and its content is updated by `optimize` and `optimize!` to
reflect the change of settings if any `kwds...` is specified and the algorithm state on
return. If the number `n` of variables and the number `m` of constraints are unchanged, no
new allocations are needed to run the algorithm given the context.

The context `ctx` has the following properties:

```julia
ctx.n        # number of variables
ctx.npt      # number of memorized variables
ctx.rhobeg   # initial size of the trust region
ctx.rhoend   # final size of the trust region
ctx.scale    # scaling factors
ctx.lower    # lower bounds
ctx.upper    # upper bounds
ctx.maxeval  # maximum number of objective function evaluations
ctx.verbose  # verbosity level
ctx.maximize # attempt to maximize objective function?
ctx.status   # current algorithm status
ctx.fbest    # best value of the objective function
ctx.nevals   # number of evaluations of the objective function
```

"""
mutable struct Context <: Powell.AbstractContext
    n::Int
    npt::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    maxeval::Int
    verbose::Int
    lower::Vector{Cdouble}
    upper::Vector{Cdouble}
    scale::Vector{Cdouble}
    work::Vector{Cdouble}
    status::Status
    maximize::Bool
    function Context(n::Integer;
                     npt::Integer = 2*Int(n) + 1,
                     rhobeg::Real,
                     rhoend::Real = rho_reduction*rhobeg,
                     maxeval::Integer = 30*Int(n),
                     verbose::Integer = 0,
                     lower::Union{Real,AbstractVector{<:Real}} = -Inf,
                     upper::Union{Real,AbstractVector{<:Real}} = +Inf,
                     scale::Union{Real,AbstractVector{<:Real}} = 1.0,
                     maximize::Bool = false)
        # Check settings and convert them to a proper type.
        n = Powell.fix_n(n)
        npt = Powell.fix_npt(npt, n)
        rhobeg, rhoend = Powell.fix_rho_parameters(rhobeg, rhoend)
        maxeval = Powell.fix_maxeval(maxeval)
        verbose = Powell.fix_verbose(verbose)
        scale isa Real || length(scale) == n || throw(DimensionMismatch(
            "`scale` must be a scalar or a vector of $n reals"))
        lower isa Real || length(lower) == n || throw(DimensionMismatch(
            "`lower` must be a scalar or a vector of $n reals"))
        upper isa Real || length(upper) == n || throw(DimensionMismatch(
            "`upper` must be a scalar or a vector of $n reals"))

        # Create instance, setting bounds and scaling factors.
        ctx = new(n, npt, rhobeg, rhoend, maxeval, verbose,
                  #= lower =# Powell.copy_or_fill!(Vector{Cdouble}(undef, n), lower),
                  #= upper =# Powell.copy_or_fill!(Vector{Cdouble}(undef, n), upper),
                  #= scale =# Powell.copy_or_fill!(Vector{Cdouble}(undef, n), scale),
                  #= work  =# Vector{Cdouble}(undef, work_length(n, npt)),
                  INITIAL_ITERATE, maximize)
        return reset!(ctx)
    end

end

function Powell.configure!(ctx::Context;
                           n::Integer = ctx.n,
                           npt::Integer = ctx.npt,
                           rhobeg::Real = ctx.rhobeg,
                           rhoend::Real = ctx.rhoend,
                           maxeval::Integer = ctx.maxeval,
                           verbose::Integer = ctx.verbose,
                           lower::Union{Real,AbstractVector{<:Real}} = ctx.lower,
                           upper::Union{Real,AbstractVector{<:Real}} = ctx.upper,
                           scale::Union{Real,AbstractVector{<:Real}} = ctx.scale,
                           maximize::Bool = ctx.maximize)
    # Check settings and convert them to a proper type.
    n = Powell.fix_n(n)
    npt = Powell.fix_npt(npt, n)
    rhobeg, rhoend = Powell.fix_rho_parameters(rhobeg, rhoend)
    maxeval = Powell.fix_maxeval(maxeval)
    verbose = Powell.fix_verbose(verbose)
    scale isa Real || scale === ctx.scale || length(scale) == n || throw(DimensionMismatch(
        "`scale` must be a scalar or a vector of $n reals"))
    lower isa Real || lower === ctx.lower || length(lower) == n || throw(DimensionMismatch(
        "`lower` must be a scalar or a vector of $n reals"))
    upper isa Real || upper === ctx.upper || length(upper) == n || throw(DimensionMismatch(
        "`upper` must be a scalar or a vector of $n reals"))

    # Set scaling factors and bounds.
    Powell.set_scale!(ctx, scale, n)
    Powell.copy_or_fill!(resize!(getfield(ctx, :lower), n), lower)
    Powell.copy_or_fill!(resize!(getfield(ctx, :upper), n), upper)

    # Resize work array.
    resize!(getfield(ctx, :work), work_length(n, npt))

    # Set other fields.
    setfield!(ctx, :n,        n)
    setfield!(ctx, :npt,      npt)
    setfield!(ctx, :rhobeg,   rhobeg)
    setfield!(ctx, :rhoend,   rhoend)
    setfield!(ctx, :maxeval,  maxeval)
    setfield!(ctx, :verbose,  verbose)
    setfield!(ctx, :status,   INITIAL_ITERATE)
    setfield!(ctx, :maximize, maximize)
    return reset!(ctx)
end

function reset!(ctx::Context)
    getfield(ctx, :work)[1] = NaN # ctx.fbest
    getfield(ctx, :work)[2] = 0   # ctx.nevals
    return ctx
end

Powell._getproperty(ctx::Context, ::Val{:fbest})    = getfield(ctx, :work)[1]
Powell._getproperty(ctx::Context, ::Val{:nevals})   = getfield(ctx, :work)[2] |> Int
Powell._getproperty(ctx::Context, ::Val{:n})        = getfield(ctx, :n)
Powell._getproperty(ctx::Context, ::Val{:npt})      = getfield(ctx, :npt)
Powell._getproperty(ctx::Context, ::Val{:rhobeg})   = getfield(ctx, :rhobeg)
Powell._getproperty(ctx::Context, ::Val{:rhoend})   = getfield(ctx, :rhoend)
Powell._getproperty(ctx::Context, ::Val{:scale})    = getfield(ctx, :scale)
Powell._getproperty(ctx::Context, ::Val{:lower})    = getfield(ctx, :lower)
Powell._getproperty(ctx::Context, ::Val{:upper})    = getfield(ctx, :upper)
Powell._getproperty(ctx::Context, ::Val{:maxeval})  = getfield(ctx, :maxeval)
Powell._getproperty(ctx::Context, ::Val{:verbose})  = getfield(ctx, :verbose)
Powell._getproperty(ctx::Context, ::Val{:maximize}) = getfield(ctx, :maximize)
Powell._getproperty(ctx::Context, ::Val{:status})   = getfield(ctx, :status)

Base.propertynames(ctx::Context) = (
    :fbest,
    :lower,
    :maxeval,
    :maximize,
    :n,
    :nevals,
    :npt,
    :rhobeg,
    :rhoend,
    :scale,
    :status,
    :upper,
    :verbose)

function Powell.optimize!(ctx::Context, f::Function, x::DenseVector{Cdouble}; kwds...)
    isempty(kwds) || configure!(ctx; kwds...)
    length(x) == ctx.n || throw(DimensionMismatch("bad number of variables"))
    GC.@preserve ctx begin
        status = @ccall libbobyqa.bobyqa_optimize(
            ctx.n::Cptrdiff_t, ctx.npt::Cptrdiff_t, ctx.maximize::Cint,
            _objfun_c[]::Ptr{Cvoid}, f::Any, x::Ptr{Cdouble},
            ctx.lower::Ptr{Cdouble}, ctx.upper::Ptr{Cdouble},
            Powell.unsafe_scale_pointer(ctx)::Ptr{Cdouble}, ctx.rhobeg::Cdouble,
            ctx.rhoend::Cdouble, ctx.verbose::Cptrdiff_t, ctx.maxeval::Cptrdiff_t,
            getfield(ctx, :work)::Ptr{Cdouble})::Status
        setfield!(ctx, :status, status)
    end
    return ctx, x
end

# `work_length(n, npt)` yields the number of elements in BOBYQA workspace.
function work_length(n::Integer, npt::Integer)
    n = Int(n)
    npt = Int(npt)
    return (npt + 5)*(npt + n) + div(3*n*(n + 5),2) + 3n
    # FIXME + 3n is to store the scaled bounds, not always needed
end

# Wrapper for the objective function in BOBYQA, the actual objective function
# is provided by the client data as a `jl_value_t*` pointer.
function _objfun(n::Cptrdiff_t, xptr::Ptr{Cdouble}, fptr::Ptr{Cvoid})::Cdouble
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    return Cdouble(f(x))
end

# With precompilation, `__init__()` carries on initializations that must occur
# at runtime like `@cfunction` which returns a raw pointer.
const _objfun_c = Ref{Ptr{Cvoid}}()
function __init__()
    _objfun_c[] = @cfunction(_objfun, Cdouble,
                             (Cptrdiff_t, Ptr{Cdouble}, Ptr{Cvoid}))
end

end # module Bobyqa

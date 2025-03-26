"""

Module `Newuoa` provides Mike Powell's NEWUOA algorithm to minimize a function of many
variables.

"""
module Newuoa

export
    # Specific symbols.
    newuoa, newuoa!,

    # Common API.
    configure!,
    issuccess,
    maximize, maximize!,
    minimize, minimize!,
    optimize, optimize!

using ..Powell
using ..Powell: rho_reduction

const libnewuoa = Powell.OptimPack_jll.libnewuoa

"""
    newuoa(f, x0; rhobeg, rhoend=$rho_reduction*rhobeg, npt=2*length(x0)+1,
           scale=1.0, verbose=0, maxeval=30*length(x0), maximize=false) -> status, x, fx

runs *NEWUOA* algorithm to find the variables `x` which solve the unconstrained problem:

    min f(x)

where `x` is a vector of variables that has `n ≥ 2` components and `f(x)` is an objective
function. The algorithm employs quadratic approximations to the objective which
interpolates the objective function at a number of points specified by keyword `npt`, the
value `npt = 2n + 1` being recommended. The parameter `rho` controls the size of the trust
region and it is reduced automatically from `rhobeg` to `rhoend`.

The method returns a 3-tuple: `status` indicates whether the algorithm was successful, `x`
is the final value of the variables and `fx` is the objective function at `x`. Normally,
`status` should be `Newuoa.SUCCESS` or equivalently `issuccess(status)` should be true;
otherwise, `status.reason` yields a textual explanation about the failure.


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


## Keywords

The following keywords are available:

* `rhobeg` and `rhoend` are the initial and final sizes of the trust region. `0 < rhoend ≤
  rhobeg` must hold.

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

* [`newuoa!`](@ref) is the in-place version of the algorithm.

* See [`Newuoa.Context`](@ref) for another way to apply the algorithm which is useful to
  retrieve more information about the algorithm state or to solve several similar problems
  while avoiding allocations and thus the overhead of the garbage collector.


## References

The algorithm is described in:

* M.J.D. Powell, "The NEWUOA software for unconstrained minimization without derivatives,"
  in Large-Scale Nonlinear Optimization, editors G. Di Pillo and M. Roma, Springer, pp.
  255-297 (2006).

"""
newuoa(f::Function, x::AbstractVector{<:Real}; kwds...) =
    newuoa!(f, copy_variables(x); kwds...)

"""
    newuoa!(f, x; kwds...) -> status, x, fx

runs the in-place version of **NEWUOA** algorithm to find the variables `x` which solve a
bound constrained optimization problem. On entry, argument `x` specifies the initial
variables; on return, `x` is overwritten by the solution. See [`newuoa`](@ref) for a
description of the algorithm and of the available keywords.

"""
function newuoa!(f::Function, x::DenseVector{Cdouble}; kwds...)
    ctx = Context(length(x); kwds...)
    optimize!(ctx, f, x)
    return ctx.status, x, ctx.fbest
end

# Status returned by NEWUOA.
struct Status <: Powell.AbstractStatus
    code::Cint
end

# Possible status values returned by NEWUOA (as defined in C header file `newuoa.h`).
const INITIAL_ITERATE      = Status( 2)
const ITERATE              = Status( 1)
const SUCCESS              = Status( 0)
const BAD_NVARS            = Status(-1)
const BAD_NPT              = Status(-2)
const BAD_RHO_RANGE        = Status(-3)
const BAD_SCALING          = Status(-4)
const ROUNDING_ERRORS      = Status(-5)
const TOO_MANY_EVALUATIONS = Status(-6)
const STEP_FAILED          = Status(-7)
const BAD_ADDRESS          = Status(-8)
const CORRUPTED            = Status(-9)

# Get a textual explanation of the status returned by NEWUOA.
Powell._getproperty(status::Status, ::Val{:reason}) =
    status == INITIAL_ITERATE      ? "Algorithm not yet started" :
    status == ITERATE              ? "Caller is requested to evaluate the objective function" :
    status == SUCCESS              ? "Algorithm converged" :
    status == BAD_NVARS            ? "Bad number of variables" :
    status == BAD_NPT              ? "NPT is not in the required interval" :
    status == BAD_RHO_RANGE        ? "Invalid trust region parameters" :
    status == BAD_SCALING          ? "Bad scaling factor(s)" :
    status == ROUNDING_ERRORS      ? "Too much cancellation in a denominator" :
    status == TOO_MANY_EVALUATIONS ? "Maximum number of function evaluations exceeded" :
    status == STEP_FAILED          ? "Trust region step has failed to reduce quadratic approximation" :
    status == BAD_ADDRESS          ? "Illegal null address" :
    status == CORRUPTED            ? "Corrupted or misused workspace" :
    "Unknown NEWUOA status: `Newuoa.Status($(status.code))"

"""
    ctx = Newuoa.Context(n; npt=2n+1, rhobeg, rhoend=$rho_reduction*rhobeg,
                         maxeval=30n, verbose=0, scale=1.0, maximize=false)

creates a new context for solving an optimization problem with NEWUOA.

Properties:

```julia
ctx.n        # number of variables
ctx.npt      # number of memorized variables
ctx.rhobeg   # initial size of the trust region
ctx.rhoend   # final size of the trust region
ctx.scale    # scaling factors
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

        # Create instance.
        ctx = new(n, npt, rhobeg, rhoend, maxeval, verbose,
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
                           scale::Union{Real,DenseVector{<:Real}} = ctx.scale,
                           maximize::Bool = ctx.maximize)
    # Check settings and convert them to a proper type.
    n = Powell.fix_n(n)
    npt = Powell.fix_npt(npt, n)
    rhobeg, rhoend = Powell.fix_rho_parameters(rhobeg, rhoend)
    maxeval = Powell.fix_maxeval(maxeval)
    verbose = Powell.fix_verbose(verbose)
    scale isa Real || scale === ctx.scale || length(scale) == n || throw(DimensionMismatch(
        "`scale` must be a scalar or a vector of $n reals"))

    # Set scaling factors.
    Powell.set_scale!(ctx, scale, n)

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
        status = @ccall libnewuoa.newuoa_optimize(
            ctx.n::Cptrdiff_t, ctx.npt::Cptrdiff_t, ctx.maximize::Cint,
            _objfun_c[]::Ptr{Cvoid}, f::Any, x::Ptr{Cdouble},
            Powell.unsafe_scale_pointer(ctx)::Ptr{Cdouble}, ctx.rhobeg::Cdouble,
            ctx.rhoend::Cdouble, ctx.verbose::Cptrdiff_t, ctx.maxeval::Cptrdiff_t,
            getfield(ctx, :work)::Ptr{Cdouble})::Status
        setfield!(ctx, :status, status)
    end
    return ctx, x
end

# `work_length(n, npt)` yields the number of elements in NEWUOA workspace.
function work_length(n::Integer, npt::Integer)
    n = Int(n)
    npt = Int(npt)
    return (npt + 13)*(npt + n) + div(3*n*(n + 3), 2)
end

# Wrapper for the objective function in NEWUOA, the actual objective function is provided
# by the client data as a `jl_value_t*` pointer.
function _objfun(n::Cptrdiff_t, xptr::Ptr{Cdouble}, fptr::Ptr{Cvoid})::Cdouble
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    return Cdouble(f(x))
end

# With precompilation, `__init__()` carries on initializations that must occur at runtime
# like `@cfunction` which returns a raw pointer.
const _objfun_c = Ref{Ptr{Cvoid}}()
function __init__()
    _objfun_c[] = @cfunction(_objfun, Cdouble, (Cptrdiff_t, Ptr{Cdouble}, Ptr{Cvoid}))
end

end # module Newuoa

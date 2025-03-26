"""

Module `Cobyla` provides Mike Powell's COBYLA algorithm to minimize a function of many
variables subject to inequality constraints.

"""
module Cobyla

export
    # Specific symbols.
    cobyla, cobyla!,

    # Common API.
    configure!,
    issuccess,
    maximize, maximize!,
    minimize, minimize!,
    optimize, optimize!

using ..Powell
using ..Powell: rho_reduction

const libcobyla = Powell.OptimPack_jll.libcobyla

"""
   cobyla(fc, x0; m, rhobeg, rhoend=$rho_reduction*rhobeg, scale=1.0,
          verbose=0, maxeval=30*length(x0), maximize=false) -> status, x, fx

runs *COBYLA* algorithm to find the variables `x` which solve the constrained problem:

    min f(x)    subject to   c(x) ≤ 0

where `x` is a vector of variables that has `n ≥ 1` components, `f(x)` is an objective
function and `c(x)` implement `m` inequality constraints. The algorithm employs linear
approximations to the objective and constraint functions, the approximations being formed
by linear interpolation at `n+1` points in the space of the variables. We regard these
interpolation points as vertices of a simplex. The parameter `rho` controls the size of
the simplex and it is reduced automatically from `rhobeg` to `rhoend`. For each `rho`,
COBYLA tries to achieve a good vector of variables for the current size, and then `rho` is
reduced until the value `rhoend` is reached. Therefore `rhobeg` and `rhoend` should be set
to reasonable initial changes to and the required accuracy in the variables respectively,
but this accuracy should be viewed as a subject for experimentation because it is not
guaranteed. The subroutine has an advantage over many of its competitors, however, which
is that it treats each constraint individually when calculating a change to the variables,
instead of lumping the constraints together into a single penalty function. The name of
the subroutine is derived from the phrase "Constrained Optimization BY Linear
Approximations".

Argument `x0` specifies the initial variables and `fc` is a Julia function which is called
as:

    fc(x, cx) -> fx

to store in `cx` the values of the constraints at `x` and to return `fx` the value of the
objective function at `x`. If there are no constraints (i.e. `m=0`), then `fc` is called
without the `cx` argument as:

    fc(x) -> fx

The method returns a 3-tuple: `status` indicates whether the algorithm was successful, `x`
is the final value of the variables and `fx` is the objective function at `x`. Normally,
`status` should be `Cobyla.SUCCESS` or equivalently `issuccess(status)` should be true;
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

* `m` is the number of constraints.

* `rhobeg` and `rhoend` are the initial and final sizes of the trust region.

* `scale` gives the typical magnitudes of the variables. It is a scalar (to have the same
  scale for all variables) or a vector with as many elements as `x`. The scaling factors
  must be all finite and strictly positive.

* `verbose` is the amount of printing.

* `maxeval` is the maximum number of calls to the objective function.

* `maximize` specifies whether to attempt to maximize the objective function; otherwise,
  the algorithm attempts to minimize the objective function.


## References

The algorithm is described in:

* M.J.D. Powell, "A direct search optimization method that models the objective and
  constraint functions by linear interpolation," in Advances in Optimization and Numerical
  Analysis Mathematics and Its Applications, vol. 275 (eds. Susana Gomez and Jean-Pierre
  Hennart), Kluwer Academic Publishers, pp. 51-67 (1994).

"""
cobyla(fc::Function, x0::AbstractVector{<:Real}; kwds...) =
    cobyla!(fc, Powell.copy_variables(x0); kwds...)

"""
    cobyla!(fc, x; kwds...) -> (status, x, fx)

runs the in-place version of **COBYLA** algorithm to find the variables `x` which solve a
constrained optimization problem. On entry, argument `x` specifies the initial variables;
on return, `x` is overwritten by the solution. See [`cobyla`](@ref) for a description of
the algorithm and of the available keywords.

"""
function cobyla!(fc::Function, x::DenseVector{Cdouble}; m::Integer, kwds...)
    ctx = Context(length(x), m; kwds...)
    optimize!(ctx, fc, x)
    return ctx.status, x, ctx.fbest
end

# Status returned by COBYLA.
struct Status <: Powell.AbstractStatus
    code::Cint
end

# Possible status values returned by COBYLA (as defined in C header file `cobyla.h`).
const INITIAL_ITERATE      = Status( 2)
const ITERATE              = Status( 1)
const SUCCESS              = Status( 0)
const BAD_NVARS            = Status(-1)
const BAD_NCONS            = Status(-2)
const BAD_RHO_RANGE        = Status(-3)
const BAD_SCALING          = Status(-4)
const ROUNDING_ERRORS      = Status(-5)
const TOO_MANY_EVALUATIONS = Status(-6)
const BAD_ADDRESS          = Status(-7)
const CORRUPTED            = Status(-8)

# Get a textual explanation of the status returned by COBYLA.
Powell._getproperty(status::Status, ::Val{:reason}) =
    status == INITIAL_ITERATE      ? "Algorithm not yet started" :
    status == ITERATE              ? "Caller is requested to evaluate the objective function and constraints" :
    status == SUCCESS              ? "Algorithm converged" :
    status == BAD_NVARS            ? "Bad number of variables" :
    status == BAD_NCONS            ? "Bad number of constraints" :
    status == BAD_RHO_RANGE        ? "Invalid trust region parameters" :
    status == BAD_SCALING          ? "Bad scaling factor(s)" :
    status == ROUNDING_ERRORS      ? "Rounding errors prevent progress" :
    status == TOO_MANY_EVALUATIONS ? "Maximum number of function evaluations exceeded" :
    status == BAD_ADDRESS          ? "Illegal null address" :
    status == CORRUPTED            ? "Corrupted workspace" :
    "Unknown COBYLA status: `Cobyla.Status($(status.code))"

"""
    ctx = Cobyla.Context(n, m; rhobeg, rhoend=$rho_reduction*rhobeg,
                         maxeval=30n, verbose=0, scale=1.0, maximize=false)

creates a new context for solving an optimization problem with `n` variables and `m`
constraints by COBYLA algorithm. All other settings are specified by keywords and are the
same as for the [`cobyla`](@ref) method.

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
ctx.m        # number of constraints
ctx.rhobeg   # initial size of the trust region
ctx.rhoend   # final size of the trust region
ctx.scale    # scaling factors
ctx.maxeval  # maximum number of objective function evaluations
ctx.verbose  # verbosity level
ctx.maximize # attempt to maximize objective function?
ctx.status   # current algorithm status
ctx.fbest    # best value of the objective function
ctx.nevals   # number of objective function evaluations
ctx.cworst   # worst constraint
```

"""
mutable struct Context <: Powell.AbstractContext
    n::Int
    m::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    maxeval::Int
    verbose::Int
    scale::Vector{Cdouble}
    work::Vector{Cdouble}
    iact::Vector{Cptrdiff_t}
    status::Status
    maximize::Bool
    function Context(n::Integer, m::Integer;
                     rhobeg::Real,
                     rhoend::Real = rho_reduction*rhobeg,
                     maxeval::Integer = 30*Int(n),
                     verbose::Integer = 0,
                     scale::Union{Real,AbstractVector{<:Real}} = 1.0,
                     maximize::Bool = false)
        # Check settings and convert them to a proper type.
        n = Powell.fix_n(n)
        m = Powell.fix_m(m)
        rhobeg, rhoend = Powell.fix_rho_parameters(rhobeg, rhoend)
        maxeval = Powell.fix_maxeval(maxeval)
        verbose = Powell.fix_verbose(verbose)
        scale isa Real || length(scale) == n || throw(DimensionMismatch(
            "`scale` must be a scalar or a vector of $n reals"))

        # Create instance.
        ctx = new(n, m, rhobeg, rhoend, maxeval, verbose,
                  #= scale =# Powell.copy_or_fill!(Vector{Cdouble}(undef, n), scale),
                  #= work  =# Vector{Cdouble}(undef, work_length(n, m)),
                  #= iact  =# Vector{Cptrdiff_t}(undef, m + 1),
                  INITIAL_ITERATE, maximize)
        return reset!(ctx)
    end
end

function Powell.configure!(ctx::Context;
                           n::Integer = ctx.n,
                           m::Integer = ctx.m,
                           rhobeg::Real = ctx.rhobeg,
                           rhoend::Real = ctx.rhoend,
                           maxeval::Integer = ctx.maxeval,
                           verbose::Integer = ctx.verbose,
                           scale::Union{Real,AbstractVector{<:Real}} = ctx.scale,
                           maximize::Bool = ctx.maximize)
    # Check settings and convert them to a proper type.
    n = Powell.fix_n(n)
    m = Powell.fix_m(m)
    rhobeg, rhoend = Powell.fix_rho_parameters(rhobeg, rhoend)
    maxeval = Powell.fix_maxeval(maxeval)
    verbose = Powell.fix_verbose(verbose)
    scale isa Real || scale === ctx.scale || length(scale) == n || throw(DimensionMismatch(
        "`scale` must be a scalar or a vector of $n reals"))

    # Set scaling factors.
    Powell.set_scale!(ctx, scale, n)

    # Resize work arrays.
    resize!(getfield(ctx, :work), work_length(n, m))
    resize!(getfield(ctx, :iact), m + 1)

    # Set other fields.
    setfield!(ctx, :n,        n)
    setfield!(ctx, :m,        m)
    setfield!(ctx, :rhobeg,   rhobeg)
    setfield!(ctx, :rhoend,   rhoend)
    setfield!(ctx, :maxeval,  maxeval)
    setfield!(ctx, :verbose,  verbose)
    setfield!(ctx, :status,   INITIAL_ITERATE)
    setfield!(ctx, :maximize, maximize)
    return reset!(ctx)
end

function reset!(ctx::Context)
    getfield(ctx, :iact)[1] = 0   # ctx.nevals
    getfield(ctx, :work)[1] = NaN # ctx.fbest
    getfield(ctx, :work)[2] = NaN # ctx.cworst
    return ctx
end

Powell._getproperty(ctx::Context, ::Val{:nevals})   = getfield(ctx, :iact)[1]
Powell._getproperty(ctx::Context, ::Val{:fbest})    = getfield(ctx, :work)[1]
Powell._getproperty(ctx::Context, ::Val{:cworst})   = getfield(ctx, :work)[2]
Powell._getproperty(ctx::Context, ::Val{:n})        = getfield(ctx, :n)
Powell._getproperty(ctx::Context, ::Val{:m})        = getfield(ctx, :m)
Powell._getproperty(ctx::Context, ::Val{:rhobeg})   = getfield(ctx, :rhobeg)
Powell._getproperty(ctx::Context, ::Val{:rhoend})   = getfield(ctx, :rhoend)
Powell._getproperty(ctx::Context, ::Val{:scale})    = getfield(ctx, :scale)
Powell._getproperty(ctx::Context, ::Val{:maxeval})  = getfield(ctx, :maxeval)
Powell._getproperty(ctx::Context, ::Val{:verbose})  = getfield(ctx, :verbose)
Powell._getproperty(ctx::Context, ::Val{:maximize}) = getfield(ctx, :maximize)
Powell._getproperty(ctx::Context, ::Val{:status})   = getfield(ctx, :status)

Base.propertynames(ctx::Context) = (
    :cworst,
    :fbest,
    :m,
    :maxeval,
    :maximize,
    :n,
    :nevals,
    :rhobeg,
    :rhoend,
    :scale,
    :status,
    :verbose)

function Powell.optimize!(ctx::Context, fc::Function, x::DenseVector{Cdouble}; kwds...)
    isempty(kwds) || configure!(ctx; kwds...)
    length(x) == ctx.n || throw(DimensionMismatch("bad number of variables"))
    GC.@preserve ctx begin
        setfield!(ctx, :status, @ccall libcobyla.cobyla_optimize(
            ctx.n::Cptrdiff_t, ctx.m::Cptrdiff_t, ctx.maximize::Cint,
            _objfun_c[]::Ptr{Cvoid}, fc::Any, x::Ptr{Cdouble},
            Powell.unsafe_scale_pointer(ctx)::Ptr{Cdouble}, ctx.rhobeg::Cdouble,
            ctx.rhoend::Cdouble, ctx.verbose::Cptrdiff_t, ctx.maxeval::Cptrdiff_t,
            getfield(ctx, :work)::Ptr{Cdouble}, getfield(ctx,:iact)::Ptr{Cptrdiff_t})::Status)
    end
    return ctx, x
end

# `work_length(n, m)` yields the number of elements in COBYLA workspace.
function work_length(n::Integer, m::Integer)
    n = Int(n)
    m = Int(m)
    return n*(3*n + 2*m + 11) + 4*m + 6
end

# Wrapper for the objective function in COBYLA, the actual objective function is provided
# by the client data as a `jl_value_t*` pointer.
function _objfun(n::Cptrdiff_t, m::Cptrdiff_t, xptr::Ptr{Cdouble},
                 cptr::Ptr{Cdouble}, fptr::Ptr{Cvoid})
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    return (m > 0 ? Cdouble(f(x, unsafe_wrap(Array, cptr, m))) : Cdouble(f(x)))::Cdouble
end

# With precompilation, `__init__()` carries on initializations that must occur
# at runtime like `@cfunction` which returns a raw pointer.
const _objfun_c = Ref{Ptr{Cvoid}}()
function __init__()
    _objfun_c[] = @cfunction(_objfun, Cdouble,
                             (Cptrdiff_t, Cptrdiff_t, Ptr{Cdouble},
                              Ptr{Cdouble}, Ptr{Cvoid}))
end

end # module Cobyla

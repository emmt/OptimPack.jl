"""

Module `OptimPack.Bobyqa` provides Mike Powell's BOBYQA algorithm to optimize a function of
many variables subject to bound constraints.

"""
module Bobyqa

export
    bobyqa,
    bobyqa!,
    issuccess

using TypeUtils: @public
@public Context Status configure! maximize maximize! minimize minimize! solve!

using LinearAlgebra: issuccess

using Neutrals
using CEnum

using ..Powell:
    Powell,
    check_scale,
    copy_array,
    default_maxevals,
    default_npt,
    dense_array,
    rho_reduction

using ...OptimPack:
    OptimPack,
    throw_assertion_failed,
    throw_bad_argument,
    throw_dimension_mismatch

if !isdefined(@__MODULE__, :Memory)
    const Memory{T} = Vector{T}
end

# NOTE This enumeration is generated automatically, see `../gen/README.md`.
@cenum Status::Int32 begin
    SUCCESS = 0
    BAD_NVARS = -1
    BAD_NPT = -2
    BAD_RHO_RANGE = -3
    BAD_SCALING = -4
    TOO_CLOSE = -5
    ROUNDING_ERRORS = -6
    TOO_MANY_EVALUATIONS = -7
    STEP_FAILED = -8
end

mutable struct Context{N,
                       L<:DenseArray{Cdouble,N},
                       U<:DenseArray{Cdouble,N},
                       S<:DenseArray{Cdouble,N},
                       W<:DenseVector{Cdouble}}
    npt::Int
    maxevals::Int
    verbose::Int
    maximize::Bool
    status::Status
    rhobeg::Cdouble
    rhoend::Cdouble
    lower::L
    upper::U
    scale::S
    work::W
end

Base.getproperty(ctx::Context, key::Symbol) =
    key === :fbest ? getfield(ctx, :work)[1] :
    key === :evals ? getfield(ctx, :work)[2] |> Int :
    key === :n     ? length(getfield(ctx, :scale)) :
    key === :shape ? size(getfield(ctx, :scale)) :
    key !== :work  ? getfield(ctx, key) : KeyError(key)

Base.propertynames(ctx::Context) = (
    :evals, :fbest, :lower, :maxevals, :maximize, :n, :npt, :rhobeg, :rhoend,
    :scale, :shape, :status, :upper, :verbose)

"""
    using OptimPack_jll
    ctx = Bobyqa.Context(x; npt=2n+1, rhobeg, rhoend=$rho_reduction*rhobeg,
                            lower=-Inf, upper=+Inf, scale=𝟙,
                            maxevals=30n, verbose=0, maximize=false)

Create a new context for solving an optimization problem with bound constraints on the
variables by BOBYQA algorithm. Argument `x` is the array storing the variables. All other
settings are specified by keywords `kwds...` and are the same as for the [`bobyqa`](@ref)
method.

The context `ctx` can be used to solve one or several problems (although one problem at a
time) with BOBYQA algorithm by:

    ctx, x = OptimPack.solve!(ctx, f, x; kwds...)

where `x` contains the initial variables on entry and the solution on return. The keywords
of the method be specified by `kwds...`. These keywords may also be changed by calling
[`OptimPack.configure!`](@ref).

The context `ctx` has the following properties:

```julia
ctx.evals    # number of evaluations of the objective function
ctx.fbest    # best value of the objective function
ctx.lower    # lower bounds
ctx.maxevals # maximum number of evaluations of the objective function
ctx.maximize # attempt to maximize objective function?
ctx.n        # number of variables
ctx.npt      # number of memorized variables
ctx.rhobeg   # initial size of the trust region
ctx.rhoend   # final size of the trust region
ctx.scale    # scaling factors
ctx.shape    # shape of the variables
ctx.status   # current algorithm status
ctx.upper    # upper bounds
ctx.verbose  # verbosity level
```

## See also

[`bounds`](@ref), [`OptimPack.solve!`](@ref), and [`OptimPack.configure!`](@ref).

"""
function Context(x::AbstractArray;
                 rhobeg::Real, rhoend::Real = rho_reduction*rhobeg,
                 maximize::Bool = false,
                 lower::Union{Real,AbstractArray{<:Real}} = -Inf,
                 upper::Union{Real,AbstractArray{<:Real}} = +Inf,
                 scale::Union{Real,AbstractArray{<:Real}} = 𝟙,
                 npt::Integer = default_npt(length(x)),
                 maxevals::Integer = default_maxevals(length(x)),
                 verbose::Integer = 0)
    shape = size(x)
    lower = check_lower(lower, shape)
    upper = check_upper(upper, shape)
    scale = dense_array(Cdouble, check_scale(scale, shape))
    n = length(x)::Int
    npt = check_npt(npt, n)
    work = Vector{Cdouble}(undef, workspace_length(n, npt))
    return Context(npt, maxevals, verbose, maximize, SUCCESS, rhobeg, rhoend,
                   lower, upper, scale, work)
end

function check_npt(npt::Integer, n::Int)
    n+2 ≤ npt ≤ (n+1)*(n+2)÷2 || throw_bad_argument(
        "`n+2 ≤ npt ≤ (n+1)*(n+2)÷2` does not not hold, got `n=", n, " and `npt=", npt, "`")
    return Int(npt)::Int
end

# NOTE We allocate 3n extra elements in case scaling of variables is used.
workspace_length(n::Int, npt::Int) = (npt+5)*(npt+n) + 3n*(n+5)÷2 + 3n

function OptimPack.configure!(ctx::Context;
                              npt::Integer = ctx.npt,
                              rhobeg::Real = ctx.rhobeg,
                              rhoend::Real = ctx.rhoend,
                              maxevals::Integer = ctx.maxevals,
                              verbose::Integer = ctx.verbose,
                              lower::Union{Real,AbstractArray{<:Real}} = ctx.lower,
                              upper::Union{Real,AbstractArray{<:Real}} = ctx.upper,
                              scale::Union{Real,AbstractArray{<:Real}} = ctx.scale,
                              maximize::Bool = ctx.maximize)
    if npt != ctx.npt
        npt = check_npt(npt, ctx.n)
        len = workspace_length(ctx.n, npt)
        work = getfield(ctx, :work)
        if length(work) != len
            if work isa Vector
                resize!(work, len)
            else
                setfield!(ctx, :work, Memory{Cdouble}(undef, len))
            end
        end
        setfield!(ctx, :npt, npt)
    end
    if lower !== ctx.lower
        if lower isa AbstractArray
            size(lower) == ctx.shape || throw_dimension_mismatch(
                "`lower` bound must have the same shape as the variables")
            copyto!(ctx.lower, lower)
        else
            fill!(ctx.lower, lower)
        end
    end
    if upper !== ctx.upper
        if upper isa AbstractArray
            size(upper) == ctx.shape || throw_dimension_mismatch(
                "`upper` bound must have the same shape as the variables")
            copyto!(ctx.upper, upper)
        else
            fill!(ctx.upper, upper)
        end
    end
    if scale !== ctx.scale
        check_scale(Bool, scale) ||  throw_bad_argument(
            "scaling factors must all be finite and non-negative")
        if scale isa AbstractArray
            size(scale) == ctx.shape || throw_dimension_mismatch(
                "`scale` must have the same shape as the variables")
            copyto!(ctx.scale, scale)
        else
            fill!(ctx.scale, scale)
        end
    end
    setfield!(ctx, :rhobeg,   convert(Cdouble, rhobeg))
    setfield!(ctx, :rhoend,   convert(Cdouble, rhoend))
    setfield!(ctx, :maxevals, convert(Int, maxevals))
    setfield!(ctx, :verbose,  convert(Int, verbose))
    setfield!(ctx, :maximize, maximize)
    return ctx
end

for bound in (:lower, :upper)
    func = Symbol("check_", bound)
    @eval begin
        function $func(val::Real, shape::Dims)
            return fill!(Array{Cdouble}(undef, shape), val)
        end
        function $func(arr::AbstractArray, shape::Dims)
            size(arr) == shape || throw_dimension_mismatch(
                "`$($bound)` bound must have the same shape as the variables")
            return dense_array(Cdouble, arr)
        end
    end
end

"""
    using OptimPack_jll
    bobyqa(f, x0; rhobeg, rhoend=$rho_reduction*rhobeg, lower=-Inf, upper=+Inf, scale=𝟙,
           npt=2n+1, verbose=0, maxevals=30n+5, maximize=false) -> status, x, fx, nf

Run *BOBYQA* algorithm to minimize (or maximize) the objective function `f(x)` starting at
variables `x0` with `n ≥ 2` components and subject to the bound constraints:

    lower ≤ x ≤ upper

The initial variables `x0` must be feasible.

The algorithm employs quadratic approximations to the objective which interpolate the
objective function at a number of points specified by keyword `npt`, the value `npt = 2n +
1` being recommended. The parameter `rho` controls the size of the trust region and it is
reduced automatically from `rhobeg` to `rhoend`.

`bobyqa` returns a 4-tuple: `status` indicates whether the algorithm was successful, `x` is
the approximate solution, `fx = f(x)`, and `nf` is the number of evaluations of the
objective function. In case of success, `status == Bobyqa.SUCCESS` or, equivalently,
`issuccess(status)` hold. In any case,`summary(status)` yields a textual explanation about
the termination of the algorithm.


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
  rhobeg` must hold. At least `rhobeg` must be specified.

* `lower` and `upper` are the bounds on the variables.

* `scale` gives the typical magnitudes of the variables. It is a scalar (to have the same
  scale for all variables) or an array of same shape as `x0`. The scaling factors must be
  all finite and strictly positive.

* `verbose` is the amount of printing.

* `maxevals` is the maximum number of calls to the objective function.

* `npt` is the number of points to use for the quadratic approximation of the objective
  function. The default setting is the recommended value: `npt = 2n + 1` with `n =
  length(x0)` the number of variables.

* `maximize` specifies whether to attempt to maximize the objective function; otherwise,
  the algorithm attempts to minimize the objective function.


## See also

* [`bobyqa!`](@ref) is the in-place version of the algorithm.

* See [`Bobyqa.Context`](@ref) for another way to apply the algorithm which is useful to
  retrieve more information about the algorithm state or to solve several similar problems
  while avoiding allocations and thus the overhead of the garbage collector.

* M.J.D. Powell, *"The BOBYQA algorithm for bound constrained optimization without
  derivatives"*, Technical Report NA2009/06 of the Department of Applied Mathematics and
  Theoretical Physics, Cambridge, England (2009).

"""
function bobyqa(f, x0::AbstractVector{<:Real}; kwds...)
    x = copy_array(Cdouble, x0)
    return bobyqa!(f, x; kwds...)
end

"""
    using OptimPack_jll
    bobyqa!(f, x; kwds...) -> status, x, fx, nf

Run the in-place version of **BOBYQA** algorithm to minimize (or maximize) the multi-variate
objective function `f(x)` subject to bound constraints on `x`. On entry, argument `x`
specifies the initial variables; on return, `x` is overwritten by the solution. See
[`bobyqa`](@ref) for a description of the algorithm and available keywords.

"""
function bobyqa!(f, x::AbstractArray; kwds...)
    x isa DenseArray{Cdouble} || throw_bad_argument("`x` must be a `DenseArray{$Cdouble}`")
    error("`OptimPack_jll` is not loaded, call `using OptimPack_jll` first")
end

# TODO doc. needed
for func in (:maximize, :minimize)
    func! = Symbol(func,"!")
    @eval begin
        $func(f, x0::AbstractArray; kwds...) =
            bobyqa(f, x0; kwds..., maximize=$(func === :maximize))
        $func!(f, x::DenseArray{Cdouble}; kwds...) =
            bobyqa!(f, x; kwds..., maximize=$(func === :maximize))
    end
end

end # module Bobyqa

"""

Module `OptimPack.Newuoa` provides Mike Powell's NEWUOA algorithm to optimize a function of
many variables.

"""
module Newuoa

export
    newuoa,
    newuoa!,
    issuccess

using TypeUtils: @public
@public Context Status configure! maximize maximize! minimize minimize! solve!

using LinearAlgebra: issuccess

using Neutrals
using CEnum

using ..Powell:
    Powell,
    default_maxevals,
    default_npt,
    rho_reduction

using ...OptimPack:
    OptimPack,
    throw_assertion_failed,
    throw_bad_argument,
    throw_dimension_mismatch

# NOTE This enumeration is generated automatically, see `../gen/README.md`.
@cenum Status::Int32 begin
    INITIAL_ITERATE = 2
    ITERATE = 1
    SUCCESS = 0
    BAD_NVARS = -1
    BAD_NPT = -2
    BAD_RHO_RANGE = -3
    BAD_SCALING = -4
    ROUNDING_ERRORS = -5
    TOO_MANY_EVALUATIONS = -6
    STEP_FAILED = -7
    BAD_ADDRESS = -8
    CORRUPTED = -9
end

"""
    using OptimPack_jll
    ctx = Newuoa.Context(x; rhobeg, rhoend=$rho_reduction*rhobeg,
                            maximize=false, scale=𝟙,
                            npt=2n+1, maxevals=30n+5, verbose=0)

Create a new context for solving an optimization problem with NEWUOA via
[`OptimPack.solve!`](@ref).

Properties:

```julia
ctx.evals     # number of evaluations of the objective function
ctx.fbest     # best value of the objective function
ctx.maximize  # attempt to maximize objective function?
ctx.npt       # number of memorized variables
ctx.scale     # scaling factors
ctx.shape     # shape of the variables
ctx.status    # current algorithm status
```

## See also

[`newuoa`](@ref), [`OptimPack.solve!`](@ref), and [`OptimPack.configure!`](@ref).

"""
mutable struct Context{S<:AbstractArray}
    ptr::Ptr{Cvoid}
    fbest::Cdouble
    scale::S # scaling factors
    npt::Int
    maximize::Bool
    # TODO maxevals::Int
    # TODO verbose::Int
    # TODO rhobeg::Cdouble
    # TODO rhoend::Cdouble
end

# TODO NEWUOA library must be updated to have more accessors and mutators
#      on the context.
"""
    OptimPack.configure!(ctx::Newuoa.Context; kwds...) -> ctx

Configure context `cxt` created by [`Newuoa.Context`](@ref) for the NEWUOA algorithm.

"""
function OptimPack.configure!(ctx::Context;
                              scale::Union{Real,AbstractVector{<:Real}} = ctx.scale,
                              maximize::Bool = ctx.maximize)
    if scale !== ctx.scale
        if scale isa Real
            check_scale(scl) || throw_bad_argument(
                "`scale` must be finite and strictly positive")
            fill!(ctx.scale, scale) # FIXME only works if ctx.scale is a writable array
        else
            size(scl) == size(ctx.scale) || throw_dimension_mismatch(
                "`scale` must have the same shape as the variables")
            check_scale(scl) || throw_bad_argument(
                "all elements of `scale` must be finite and strictly positive")
             copyto!(ctx.scale, scale) # FIXME only works if ctx.scale is a writable array
        end
    end
    setfield!(ctx, :maximize, maximize)
    return ctx
end

"""
    using OptimPack_jll
    newuoa(f, x0;  rhobeg, rhoend=$rho_reduction*rhobeg, scale=𝟙, npt=2n+1,
           verbose=0, maxevals=30n+5, maximize=false) -> status, x, fx, nf, rho

Run *NEWUOA* algorithm to minimize (or maximize) the objective function `f(x)`
starting at variables `x0` with `n ≥ 2` components.

The algorithm employs quadratic approximations to the objective which interpolates the
objective function at a number of points specified by keyword `npt`, the value `npt = 2n +
1` being recommended. The parameter `rho` controls the size of the trust region and it is
reduced automatically from `rhobeg` to `rhoend`.

`newuoa` returns a 5-tuple: `status` indicates whether the algorithm was successful, `x` is
the approximate solution, `fx = f(x)`, `nf` is the number of evaluations of the objective
function, and `rho` is the final radius of the trust region. In case of success, `status ==
Newuoa.SUCCESS` or, equivalently, `issuccess(status)` hold. In any case,`summary(status)`
yields a textual explanation about the termination of the algorithm.


## Precision and scaling of variables

Suitable scaling of the variables is essential for the success of the algorithm and the
`scale` keyword should be specified if the typical precision is not the same for all
variables. If `scale` is an array of same shape as the variables `x`, then `scale[i]*rho`
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
  rhobeg` must hold. At least `rhobeg` must be specified.

* `scale` gives the typical magnitudes of the variables. It is a scalar (to have the same
  scale for all variables) or an array of same shape elements as `x0`. The scaling factors
  must be all finite and strictly positive.

* `verbose` is the amount of printing.

* `maxevals` is the maximum number of calls to the objective function.

* `npt` is the number of points to use for the quadratic approximation of the objective
  function. The default setting is the recommended value: `npt = 2n + 1` with `n =
  length(x0)` the number of variables.

* `maximize` specifies whether to attempt to maximize the objective function; otherwise,
  the algorithm attempts to minimize the objective function.


## See also

* [`newuoa!`](@ref) is the in-place version of the algorithm.

* See [`Newuoa.Context`](@ref) and [`Newuoa.solve!`](@ref) for another way to apply the
  algorithm which is useful to solve several similar problems while avoiding allocations and
  thus the overhead of the garbage collector.

* M.J.D. Powell, *"The NEWUOA software for unconstrained minimization without derivatives"*,
  in Large-Scale Nonlinear Optimization, editors G. Di Pillo and M. Roma, Springer, pp. 2
  55-297 (2006).

"""
function newuoa end

"""
    using OptimPack_jll
    newuoa!(f, x; kwds...) -> status, x, fx, nf, rho

Run the in-place version of **NEWUOA** algorithm to minimize (or maximize) the multi-variate
objective function `f(x)`. On entry, argument `x` specifies the initial variables; on
return, `x` is overwritten by the solution. See [`newuoa`](@ref) for a description of the
algorithm and available keywords.

"""
function newuoa! end

# TODO doc. needed
for func in (:maximize, :minimize)
    func! = Symbol(func,"!")
    @eval begin
        $func(f, x0::AbstractArray; kwds...) =
            newuoa(f, x0; kwds..., maximize=$(func === :maximize))
        $func!(f, x::DenseArray{Cdouble}; kwds...) =
            newuoa!(f, x; kwds..., maximize=$(func === :maximize))
    end
end

end # module Newuoa

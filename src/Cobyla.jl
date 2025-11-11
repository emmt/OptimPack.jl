"""

Module `OptimPack.Cobyla` provides Mike Powell's COBYLA algorithm to optimize a function of
many variables subject to inequality constraints.

"""
module Cobyla

export
    cobyla,
    cobyla!,
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
    throw_dimension_mismatch,
    throw_package_required

# NOTE This enumeration is generated automatically, see `../gen/README.md`.
@cenum Status::Int32 begin
    INITIAL_ITERATE = 2
    ITERATE = 1
    SUCCESS = 0
    BAD_NVARS = -1
    BAD_NCONS = -2
    BAD_RHO_RANGE = -3
    BAD_SCALING = -4
    ROUNDING_ERRORS = -5
    TOO_MANY_EVALUATIONS = -6
    BAD_ADDRESS = -7
    CORRUPTED = -8
end

"""
    using OptimPack_jll
    ctx = Cobyla.Context(x, c; rhobeg, rhoend=$rho_reduction*rhobeg,
                               scale=𝟙, maximize=false, maxevals=30n+5, verbose=0)

Creates a new context for solving an optimization problem by COBYLA algorithm. Arguments `x`
and `c` are the arrays storing the variables and the constraints. All other settings are
specified by keywords `kwds...` and are the same as for the [`cobyla`](@ref) method.

!!! note
    `x` and `c` must be dense arrays with elements of type `Cdouble`. The returned context
    shares the array `c` to store the constraints. Make a copy if this is undesirable.

The context `ctx` can be used to solve one or several problems (although one problem at a
time) with COBYLA algorithm by:

    ctx, x = OptimPack.solve!(ctx, f, x; kwds...)

where `x` contains the initial variables on entry and the solution on return. A few keywords
of the method such as `maximize` and `scale` may be specified by `kwds...`. These keywords
may also be changed by calling [`OptimPack.configure!`](@ref).

The context `ctx` has the following properties:

```julia
ctx.constr   # worst constraints
ctx.evals    # number of evaluations of the objective function
ctx.fbest    # best value of the objective function
ctx.maximize # attempt to maximize objective function?
ctx.rho      # final radius of the trust region
ctx.scale    # scaling factors
ctx.shape    # shape of the variables
ctx.status   # current algorithm status
```

## See also

[`cobyla`](@ref), [`OptimPack.solve!`](@ref), and [`OptimPack.configure!`](@ref).

"""
mutable struct Context{S<:AbstractArray{<:Union{typeof(𝟙),Cdouble}},
                       C<:DenseArray{Cdouble}}
    ptr::Ptr{Cvoid} # address of allocated context
    fbest::Cdouble
    scale::S
    constr::C
    maximize::Bool
    # TODO rhobeg::Cdouble
    # TODO rhoend::Cdouble
    # TODO maxevals::Int
    # TODO verbose::Int
end

"""
    OptimPack.configure!(ctx::Cobyla.Context; kwds...) -> ctx

Configure context `cxt` created by [`Cobyla.Context`](@ref) for the COBYLA algorithm.

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
    return reset!(ctx)
end

"""
    using OptimPack_jll
    cobyla(fc, x0, dims...; rhobeg, rhoend=$rho_reduction*rhobeg, scale=𝟙,
           verbose=0, maxevals=30n+5, maximize=false) -> status, x, cx, fx, nf, rho

Run *COBYLA* algorithm to minimize (or maximize) an objective function `f(x)` starting at
variables `x0` with `n ≥ 1` components and subject to`m = prod(dims...)` inequality
constraints `c(x) ≤ 0`.

The algorithm employs linear approximations to the objective and constraint functions, the
approximations being formed by linear interpolation at `n+1` points in the space of the
variables. We regard these interpolation points as vertices of a simplex. The parameter
`rho` controls the size of the simplex and it is reduced automatically from `rhobeg` to
`rhoend`. For each `rho`, COBYLA tries to achieve a good set of variables for the current
size, and then `rho` is reduced until the value `rhoend` is reached. Therefore `rhobeg` and
`rhoend` should be set to reasonable initial changes to and the required accuracy in the
variables respectively, but this accuracy should be viewed as a subject for experimentation
because it is not guaranteed. The subroutine has an advantage over many of its competitors,
however, which is that it treats each constraint individually when calculating a change to
the variables, instead of lumping the constraints together into a single penalty function.
The name of the subroutine is derived from the phrase *"Constrained Optimization BY Linear
Approximations"*.

Argument `x0` specifies the initial variables and `fc` is a Julia function which is called
as:

    fc(x, cx) -> fx

to store in `cx` the values of the constraints at `x` and to return `fx` the value of the
objective function at `x`. Arguments `dims...` specify the shape of the array `cx`. If there
are no constraints (i.e. `m=0`), then `fc` is called without the `cx` argument as:

    fc(x) -> fx

`cobyla` returns a 6-tuple: `status` indicates the reason of the termination of the
algorithm, `x` is the approximate solution, `cx = c(c)`, `fx = f(x)`, `nf` is the number of
evaluations of the objective function, and `rho` is the final radius of the trust region. In
case of success, `status == Cobyla.SUCCESS` or, equivalently, `issuccess(status)` hold. In
any case,`summary(status)` yields a textual explanation about the termination of the
algorithm.


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
  scale for all variables) or an array of same shape as `x0`. The scaling factors must be
  all finite and strictly positive.

* `verbose` is the amount of printing.

* `maxevals` is the maximum number of calls to the objective function.

* `maximize` specifies whether to attempt to maximize the objective function; otherwise,
  the algorithm attempts to minimize the objective function.


## See also

* [`cobyla!`](@ref) is the in-place version of the algorithm.

* See [`Cobyla.Context`](@ref) and [`Cobyla.solve!`](@ref) for another way to apply the
  algorithm which is useful to solve several similar problems while avoiding allocations and
  thus the overhead of the garbage collector.

* M.J.D. Powell, *"A direct search optimization method that models the objective and
  constraint functions by linear interpolation"*, in Advances in Optimization and Numerical
  Analysis Mathematics and Its Applications, vol. **275** (eds. Susana Gomez and Jean-Pierre
  Hennart), Kluwer Academic Publishers, pp. 51-67 (1994).

"""
cobyla(args...; kwds...) = throw_package_required(:OptimPack_jll)

"""
    using OptimPack_jll
    cobyla!(fc, x, dims...; kwds...) -> status, x, cx, fx, nf, rho

Run the in-place version of **COBYLA** algorithm to minimize (or maximize) objective
function `f(x)` subject to constraints. On entry, argument `x` specifies the initial
variables; on return, `x` is overwritten by the solution. See [`cobyla`](@ref) for a
description of the algorithm and available keywords.

"""
cobyla!(args...; kwds...) = throw_package_required(:OptimPack_jll)

# TODO doc. needed
for func in (:maximize, :minimize)
    func! = Symbol(func,"!")
    @eval begin
        $func(fc, x0::AbstractArray{<:Real}, args...; kwds...) =
            cobyla(fc, x0, args...; kwds..., maximize=$(func === :maximize))
        $func!(fc, x::DenseArray{Cdouble}, c::DenseArray{Cdouble}; kwds...) =
            cobyla!(fc, x, c; kwds..., maximize=$(func === :maximize))
    end
end

end # module Cobyla

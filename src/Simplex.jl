"""

Module `Simplex` implements Nelder-Mead *Simplex* algorithm to find the minimum of an
unconstrained multivariate function.

# Description

The algorithm is a *direct search* method (derivative free) which is called `amoeba` in
Numerical Recipes (1) and `fminsearch` in MATLAB.

This implementation shall reflect the state-of-the-art versions of the algorithm and is
based on the papers by Singer & Singer (2004) and Singer & Nelder (2009).

This implementation shall be usable for real-time applications. This is achieved by avoiding
allocations with all required workspaces stored in a context that needs to be created once
and that is reusable to solve any number of similar problems.

# Remarks

  (1) The Numerical Recipes and GSL (http://www.gnu.org/software/gsl/) versions of the
      Nelder-Mead algorithm do not perform inside contraction (only outside contraction).

  (2) In 1-D, Brent's method for smooth functions or golden search for non-smooth functions
      are probably better than Nelder-Mead algorithm -- that is: faster and guaranteed
      convergence.

  (3) Luersen and Le Riche have proposed a modified Nelder-Mead algorithm to account for
      bound constraints and to perform global optimization.

# References

* J.A. Nelder and R. Mead, "*A simplex method for function minimization*" in Computer
  Journal, vol. 7, pp. 308-313, 1965.

* J.C. Lagarias, J.A. Reeds, M.H. Wright, and P.E. Wright, "*Convergence properties of the
  Nelder-Mead simplex method in low dimensions*", SIAM Journal of Optimization, vol. 9, pp.
  112-147, 1998.

* K.I.M. McKinnon, "*Convergence of the Nelder-Mead simplex method to a nonstationary
  point*", SIAM Journal on Optimization, vol. 9, pp. 148-158, 1998.

* M.A. Luersen and R.L. Le Riche, "*Globalized Nelder-Mead method for engineering
  optimization*", Computers & Structures, vol. 82, pp. 2251-2260, 2004.

* J. Nocedal, J. and S.J. Wright, "*Numerical optimization*", Springer Verlag, 2006.

* S. Singer and S. Singer, "*Efficient implementation of the Nelder–Mead search algorithm*"
  in Applied Numerical Analysis & Computational Mathematics, vol. 1, pp. 524-534, 2004.

* S. Singer, and J.A. Nelder, "*Nelder-Mead algorithm*" in Scholarpedia, vol. 4, p. 2928,
  2009. http://www.scholarpedia.org/article/Nelder-Mead_algorithm

"""
module Simplex

# TODO Here, χ > 1, while χ > max(1, ρ) in Singer et al. papers

export
    TotalMax,
    TotalMin,
    simplex,
    issuccess

using TypeUtils: @public
# Public symbols that are not exported.
@public Context,
        build_simplex!,
        configure!,
        instantiate!,
        properties,
        solve,
        solve!

using Base.Order: Ordering
using Base: @propagate_inbounds
using LinearAlgebra
using Neutrals
using Printf
using QuickHeaps: TotalMin, TotalMax
using TypeUtils
using ..OptimPack
using ..OptimPack:
    ordinal_suffix,
    throw_bad_argument,
    throw_dimension_mismatch,
    throw_assertion_failed

if !isdefined(@__MODULE__, :Memory)
    const Memory{T} = Vector{T}
end

const VectorOfArrays{T,N} = AbstractVector{<:AbstractArray{T,N}}

# Simplex status is a simple wrapper around a symbol.
struct Status
    sym::Symbol
end

# Accessor.
Base.Symbol(status::Status) = status.sym

# Conversion and comparison.
Base.convert(::Type{Status}, x::Status) = x
Base.convert(::Type{Status}, x::Symbol) = Status(x)
Base.convert(::Type{Symbol}, x::Status) = Symbol(x)

for func in (:(==), :isequal)
    @eval begin
        Base.$func(x::Status, y::Symbol) = $func(Symbol(x), y)
        Base.$func(x::Symbol, y::Status) = $func(x,         Symbol(y))
        Base.$func(x::Status, y::Status) = $func(Symbol(x), Symbol(y))
    end
end

Base.summary(status::Status) = status_summary(Symbol(status))

status_summary(sym::Symbol) =
    sym == :allocating           ? "Allocation of resources in progress" :
    sym == :initializing         ? "Initialization in progress" :
    sym == :searching            ? "Search in progress" :
    sym == :convergence_in_f     ? "Convergence in the objective function" :
    sym == :convergence_in_x     ? "Convergence in the variable(s)" :
    sym == :rounding_errors      ? "Rounding errors prevent progress" :
    sym == :too_many_iterations  ? "Too many algorithm iterations" :
    sym == :too_many_evaluations ? "Too many evaluations of the objective function" :
    unknown_status(sym)

@noinline unknown_status(sym::Symbol) = "Unknown status `:$sym`"

# Default parameters for updating the simplex, but see the reference below for other values.
#
#     J. M. Parkinson and D. Hutchinson, An investigation into the efficiency of variants on
#     the simplex method, in: Numerical Methods for Nonlinear Optimization, edited by F. A.
#     Lootsma (Academic Press, New York, 1972), pp. 115–135.
#
const default_rho   = 1    # default reflection factor, ρ > 0
const default_chi   = 2    # default expansion factor, χ > 1
const default_gamma = 1//2 # default contraction factor, 0 < γ < 1
const default_sigma = 1//2 # default shrinkage factor, 0 < γ < 1

# Fine tuning of the algorithm.
const default_greedy_expansion   = false
const default_recompute_centroid = false

# Stopping criteria.
const default_maxiters = typemax(Int)
const default_maxevals = typemax(Int)
const default_ftol     = 1e-8
const default_xtol     = 1e-5

# Default is to minimize.
const default_order = TotalMin

"""
    ctx = Simplex.Context{T,F,X}(undef, n; order::Ordering=TotalMin, kwds...)

Create a context for optimizing a multivariate function by the Nelder-Mead *Simplex* method.
`T` is the floating-point type for computations, `F` is the type returned by the objective
function, `X` is the type of the variables, `n` is the number of variables, and `order` is
the ordering of function values (the default amounts to minimizing the objective function).

The storage for the points needed by the algorithm (the `n + 1` vertices of the simplex plus
additional work points) is not created by this method as this requires to have an instance
(or a constructor) of the variables. In other words, all entries of `ctx.points` are
unassigned. These resources will be allocated the first time an optimization problem is
solved with the context by:

    Simplex.solve!(ctx, f, x0, args...)

with `f` the objective function, `x0` initial variables, and `args...` additional arguments
to build the initial simple.

## See also

[`simplex`](@ref) for a description of the Nelder-Mead *Simplex* method.

[`Simplex.configure!`](@ref) for allowed keywords `kwds...`.

[`Simplex.solve!`](@ref) for solving an optimization problem with a given context.

[`Simplex.properties`](@ref) for the properties of a given context.

"""
mutable struct Context{T<:AbstractFloat,F<:Number,X<:AbstractArray,O<:Ordering}
    # The objective function at the n+1 vertices of the simplex.
    costs::Memory{F}

    # Ordering of function values.
    order::O

    # The n + 3 points used by the algorithm. The vertices of the simplex are the n + 1
    # first points, 2 more points are needed: one to store the centroid, the other to store
    # the reflection or the contraction point. The expansion point being stored in-place of
    # the worst simplex point. See the comments in the `solve!` method for detailed
    # explanations.
    points::Memory{X}

    # The size of the problem.
    n::Int

    # The indices of the best, worst, and second worst points.
    j_best::Int
    j_worst::Int
    j_2nd_worst::Int

    # Stopping criteria.
    maxiters::Int
    maxevals::Int
    ftol::T
    xtol::T

    # Counters.
    evaluations::Int
    iterations::Int
    reflections::Int
    expansions::Int
    inside_contractions::Int
    outside_contractions::Int
    shrinkages::Int

    # Simplex transform factors.
    rho::T   # reflection parameter, ρ > 0
    chi::T   # expansion factor, χ > 1
    gamma::T # contraction factor, 0 < γ < 1
    sigma::T # shrinkage factor, 0 < σ < 1

    # Following Singer & Singer (2004), the "linearized volume ratio" (LVR) of the simplex
    # is initially set to 1 and is updated in O(1) floating-point operations according to
    # the type of transform undergone by the simplex. Convergence in the variables is
    # assumed when the condition `LVR ≤ xtol` holds. Most other implementations take O(n²)
    # operations to check convergence in the variables.
    LVR::T

    status::Status

    # In the original algorithm (Nelder & Mead, 1965), the expansion point is accepted if it
    # is better than the former best point even though it may be worst than the reflection
    # point. This strategy is known as "greedy expansion" (Singer & Singer, 2004) as it
    # keeps the simplex as large as possible which may be favorable for non-smooth
    # functions. Most implementations apply a "greedy minimization" strategy where the
    # expansion point is accepted if it is better than the reflection point.
    greedy_expansion::Bool

    # If `n > 2`, some operations are saved by updating the centroid rather than recompute
    # it from scratch.
    recompute_centroid::Bool

    # The inner constructor
    function Context{T,F,X}(::UndefInitializer, n::Integer; order::O = default_order,
                            kwds...) where {T<:AbstractFloat, F<:Number,
                                            X<:AbstractArray, O<:Ordering}
        # Check arguments.
        n = convert(Int, n)
        n ≥ 2 || throw_assertion_failed("number of variables should be ≥ 2, got ", n)
        isconcretetype(T) || throw_assertion_failed(
            "floating-point type must be concrete, got `", T, "`")
        isconcretetype(X) || throw_assertion_failed(
            "type of variables must be concrete, got `", X, "`")
        E = eltype(X)
        float(E) === E || throw_assertion_failed(
            "variables must have floating-point element type, got `", E, "`")

        # Create the structure with sensible values (so that most properties can be used).
        # The next stage will be to allocate the points.
        ctx = new{T,F,X,O}()
        ctx.order              = order
        ctx.n                  = n
        ctx.costs              = Memory{F}(undef, n + 1)
        ctx.points             = Memory{X}(undef, n + 3)
        ctx.j_best             = 1
        ctx.j_worst            = 1
        ctx.j_2nd_worst        = 1
        ctx.maxiters           = default_maxiters
        ctx.maxevals           = default_maxevals
        ctx.ftol               = default_ftol
        ctx.xtol               = default_xtol
        ctx.rho                = default_rho
        ctx.chi                = default_chi
        ctx.gamma              = default_gamma
        ctx.sigma              = default_sigma
        ctx.status             = :allocating
        ctx.greedy_expansion   = default_greedy_expansion
        ctx.recompute_centroid = default_recompute_centroid
        reset!(ctx)

        # Apply options if any.
        isempty(kwds) || configure!(ctx; kwds...)

        return ctx
    end
end

# Traits.
float_type(x::Context) = float_type(typeof(x))
float_type(::Type{<:Context{T,F,X}}) where {T,F,X} = T
objfunc_type(x::Context) = objfunc_type(typeof(x))
objfunc_type(::Type{<:Context{T,F,X}}) where {T,F,X} = F
vertex_type(x::Context) = vertex_type(typeof(x))
vertex_type(::Type{<:Context{T,F,X}}) where {T,F,X} = X
vertex_ndims(x::Union{Context,Type{<:Context}}) = ndims(vertex_type(x))
vertex_eltype(x::Union{Context,Type{<:Context}}) = eltype(vertex_type(x))

# Accessors.
@propagate_inbounds vertex(ctx::Context, j::Int) = ctx.points[j]
Base.Order.Ordering(ctx::Context) = ctx.order

# Extend methods provided by other packages or modules.
OptimPack.configure!(ctx::Context; kwds...) = configure!(ctx; kwds...)
OptimPack.solve!(ctx::Context, args...; kwds...) = solve!(ctx, args...; kwds...)
LinearAlgebra.issuccess(ctx::Context) =
    ctx.status == :convergence_in_f || ctx.status == :convergence_in_x

"""
    ctx = Simplex.Context(f, x0, args...; order::Ordering=TotalMin, kwds...)
    ctx = Simplex.Context(f(x0), x0, args...; order::Ordering=TotalMin, kwds...)

Create a context for optimizing a multivariate function `f` by the Nelder-Mead *Simplex*
method and with an initial simplex build according to the initial variables `x0` and
arguments `args...` (see [`Simplex.build_simplex!`](@ref)). Such a context can be directly
used to optimize the objective function by calling:

    Simplex.solve!(ctx, f) -> ctx

"""
function Context(f, x0::AbstractArray, args...; kwds...)
    # Early detection of missing arguments.
    isempty(args) && throw_bad_argument("too few arguments to define the initial simplex")

    # Number of variables.
    n = length(x0)

    # Element type of the points.
    E = float(eltype(x0))

    # Allocate the first point to have the concrete type of points.
    x1 = copy!(similar(x0, E), x0)
    X = typeof(x1)

    # Evaluate the function at the starting point to have its type and value.
    f1, nf = f isa Number ? (f, 0) : (f(x1), 1)
    F = typeof(f1)

    # Floating-point type for computations, at least double precision.
    T = get_precision(Float64, E)

    # Create a context with inferred type parameters and given options.
    ctx = Context{T,F,X}(undef, n; kwds...)

    # Store the first vertex of the simplex.
    ctx.points[1] = x1
    ctx.costs[1] = f1
    ctx.evaluations = nf

    # Build the initial simplex.
    build_simplex!(ctx, x1, args...)

    return ctx
end

# Reset the context for solving a first or a new optimization problem.
function reset!(ctx::Context)
    # Reset the counters.
    ctx.evaluations          = 0
    ctx.iterations           = 0
    ctx.reflections          = 0
    ctx.expansions           = 0
    ctx.inside_contractions  = 0
    ctx.outside_contractions = 0
    ctx.shrinkages           = 0

    # Reset the "linearized volume ratio".
    ctx.LVR = 𝟙

    # Reset the objective function values.
    fill_with_NaNs!(ctx.costs)

    # Reset the status.
    ctx.status = :initializing
    return ctx
end

#
"""
    Simplex.instantiate!(ctx::Simplex.Context, x0::AbstractArray)

Check the consistency of `ctx`, make sure that all internal points are allocated with the
same axes and linear indices as `x0`, and copy `x0` in the first vertex of the simplex. This
method shall be called by all implementations of [`Simplex.build_simplex!`](@ref).

"""
function instantiate!(ctx::Context, x0::AbstractArray)
    n = ctx.n
    n ≥ 2 || throw_assertion_failed("number of variables should be ≥ 2, got ", n)
    points = ctx.points
    costs = ctx.costs
    length(points) == n + 3 || throw_assertion_failed(
        "invalid number of stored points, should be ", n + 3, ", got ", length(points))
    length(costs) == n + 1 || throw_assertion_failed(
        "invalid number of objective function values, should be ", n + 1, ", got ",
        length(costs))
    length(x0) == n || throw_dimension_mismatch(
        "simplex vertices have ", n, " entries while given variables have ", length(x0))
    ndims(x0) == vertex_ndims(ctx) || throw_dimension_mismatch(
        "simplex vertices have ", vertex_ndims(ctx),
        " dimension(s) while given variables have ", ndims(x0))
    shape = axes(x0)
    start = firstindex(x0)
    for j in eachindex(points)
        if !isassigned(points, j)
            points[j] = similar(x0, vertex_eltype(ctx))
        end
        length(points[j]) == n || throw_dimension_mismatch(
            j, ordinal_suffix(j), " simplex vertex should have ", n, " entries, got ",
            length(points[j]))
        axes(points[j]) == shape || throw_dimension_mismatch(
            j, ordinal_suffix(j),
            " simplex vertex and variables must have the same axes")
        firstindex(points[j]) == start || throw_dimension_mismatch(
            j, ordinal_suffix(j),
            " simplex vertex and variables must have the same first linear index")
    end
    points[1] === x0 || copy!(points[1], x0)
    return ctx
end

"""
    Simplex.build_simplex!(ctx::Simplex.Context, x0::AbstractArray, args...)

Build the initial simplex in context `ctx` given initial variables `x0` for the optimization
problem and arguments `args...`. This method can be specialized in `args...` to implement
different strategies.

"""
@noinline function build_simplex!(ctx::Context, x0::AbstractArray, args...)
    len = length(args)
    buf = IOBuffer()
    print(buf, len < 1 ? "too few arguments" : "no strategy is implemented")
    print(buf, " for building an initial simplex with `x0` of type `", typeof(x0), "'")
    if len > 0
        print(buf, " and `args...` of type", len < 2 ? " " : "s ")
        for i in 𝟙:len
            print(buf,
                  if i == 1
                      "`"
                  elseif i < len
                      "`, `"
                  elseif len < 3
                       "` and `"
                  else
                      "`, and `"
                  end, typeof(args[i]))
        end
        print(buf, "`")
    end
    throw_bad_argument(String(take!(buf)))
end

"""
    Simplex.build_simplex!(ctx::Simplex.Context, x0::AbstractArray,
                           siz::Union{Number,AbstractArray})

Build the initial simplex in context `ctx` given initial variables `x0` for the optimization
problem and simplex size `siz`. If `siz` is a scalar, it is assumed to give the simplex size
along all variables; otherwise, `siz` must be an array of same shape as `x0`.

The `j`-th vertex of the initial simplex is built as:

```julia
s = siz isa Number ? siz : siz[j]
for k in eachindex(x0)
    if j == k
        xj[k] = x0[k] + s
    else
        xj[k] = x0[k]
    end
end
```

"""
function build_simplex!(ctx::Context, x0::AbstractArray, siz::Union{Number,AbstractArray})
    # Allocate vertices and check structure.
    instantiate!(ctx, x0)

    # Retrieve first vertex of simplex.
    points = ctx.points
    x1 = points[1]
    start = firstindex(x1)

    # Check given simplex size if it is an array.
    if siz isa AbstractArray
        axes(siz) == axes(x1) || throw_dimension_mismatch(
            "array specifying initial simplex size and variables must have the same axes")
        firstindex(siz) == start || throw_dimension_mismatch(
            "array specifying initial simplex size and variables must have the same first linear index")
    end

    # Build the other simplex vertices than the first one.
    for j in 2:ctx.n+1
        k = start + j - 2 # the linear index of the vertex entry to perturb
        s = get_simplex_size(siz, k) # the size of the perturbation
        isfinite(s) || throw_bad_argument(
            "`siz", (siz isa AbstractArray ? "[$k]" : ""), " = ", s, "` is non-finite")
        xj = copy!(points[j], x1)
        xj[k] += s # add the perturbation
        xj[k] != x1[k] || throw_bad_argument(
            "`siz", (siz isa AbstractArray ? "[$k]" : ""), " = ", s,
            "` is too small in magnitude compared to `x0[", k, "] = ", x1[k], "`")
    end

    return nothing
end

# Get the `k`-th initial vertex size.
get_simplex_size(siz::Number, k::Int) = siz
get_simplex_size(siz::AbstractArray, k::Int) = siz[k]

"""
    Simplex.configure!(ctx; kwds...) -> ctx

Configure the properties specified by the keywords `kwds...` in context `ctx` and return the
context. Unspecified properties are left unchanged. An exception is thrown if the properties
of `ctx` are not valid. This method can thus be called with no keywords to check the
validity of the configuration in `ctx`.

The following keywords specify the stopping rules for the algorithm:

* `xtol` is a relative tolerance for the convergence in the variables. By default, `xtol =
  $default_xtol`.

* `ftol` is a relative tolerance for the convergence in the objective function. By default,
  `ftol = $default_ftol`.

* `maxiters` sets the maximum number of iterations which is virtually unlimited by default.

* `maxevals` sets the maximum number of evaluations of the objective function which is
  virtually unlimited by default.

The following keywords specify the transformation undergone by the simplex:

* `rho > 0` is the *reflection* factor. By default, `rho = $default_rho`.

* `chi > 1` is the *expansion* factor. By default, `chi = $default_chi`.

* `0 < gamma < 1` is the *contraction* factor. By default, `gamma = $default_gamma`.

* `0 < sigma < 1` is the *shrinkage* factor. By default, `sigma = $default_sigma`.

The following keywords may be used to tune the behavior of the algorithm:

* `greedy_expansion` indicates whether to accept the expansion point if it is better than
  the best point even though it may be worst than the reflection point. This corresponds to
  the original algorithm by Nelder & Mead (1965) and is known as *greedy expansion* strategy
  as it keeps the simplex as large as possible which may be favorable for non-smooth
  functions. Otherwise, a *greedy minimization* strategy is applied where the expansion
  point is accepted only if it is better than the reflection point. By default,
  `greedy_expansion = $default_greedy_expansion`.

* `recompute_centroid` indicates whether to always recompute the centroid of the simplex
  from scratch in `O(n²)` operations rather than update it in `O(n)` operations with `n` the
  number of variables. By default, `recompute_centroid = $default_recompute_centroid`.

"""
function configure!(ctx::Context;
                    rho::Real                = ctx.rho,
                    chi::Real                = ctx.chi,
                    gamma::Real              = ctx.gamma,
                    sigma::Real              = ctx.sigma,
                    ftol::Real               = ctx.ftol,
                    xtol::Real               = ctx.xtol,
                    maxevals::Integer        = ctx.maxevals,
                    maxiters::Integer        = ctx.maxiters,
                    greedy_expansion::Bool   = ctx.greedy_expansion,
                    recompute_centroid::Bool = ctx.recompute_centroid)
    # Check settings.
    rho > zero(rho) || throw_bad_argument(
        "bad reflection factor, `rho > 0` must hold, got `rho = ", rho, "`")
    chi > one(chi) || throw_bad_argument(
        "bad expansion factor, `chi > 1` must hold, got `chi = ", chi, "`")
    zero(gamma) < gamma < one(gamma) || throw_bad_argument(
        "bad contraction factor, `0 < gamma < 1` must hold, got `gamma = ", gamma, "`")
    zero(sigma) < sigma < one(sigma) || throw_bad_argument(
        "bad shrinkage factor, `0 < sigma < 1` must hold, got `sigma = ", sigma, "`")
    if ! isnan(ftol)
        ftol ≥ zero(ftol) || throw_bad_argument("`ftol` must be nonnegative, got ", ftol)
        ftol ≤  one(ftol) || throw_bad_argument("`ftol` must be ≤ 1, got ", ftol)
    end
    if ! isnan(xtol)
        xtol ≥ zero(xtol) || throw_bad_argument("`xtol` must be nonnegative, got ", xtol)
        xtol ≤  one(xtol) || throw_bad_argument("`xtol` must be ≤ 1, got ", xtol)
    end

    # Only updates after full checking.
    ctx.rho                = rho
    ctx.chi                = chi
    ctx.gamma              = gamma
    ctx.sigma              = sigma
    ctx.ftol               = ftol
    ctx.xtol               = xtol
    ctx.maxiters           = maxiters
    ctx.maxevals           = maxevals
    ctx.greedy_expansion   = greedy_expansion
    ctx.recompute_centroid = recompute_centroid
    return ctx
end

const _PROPERTIES_DOCSTRING = """
# Properties of Simplex context

The context `ctx` of Nelder-Mead's *Simplex* method has a number of properties (listed
below) that can be queried by the `ctx.key` syntax.

!!! warning
    All properties should be considered as being **read-only** by the end-user. Directly
    setting a property may break the assumptions made by the algorithm or the consistency of
    the context. Call [`Simplex.configure!`](@ref) to safely change the configurable
    options.

## Vertices and objective function values

| Property                   | Description                                        |
|:---------------------------|:---------------------------------------------------|
| `ctx.n`                    | Number of variables of the problem                 |
| `ctx.points`               | All points used by the algorithm                   |
| `ctx.vertices`             | Vertices of the simplex                            |
| `ctx.costs`                | Objective function at the vertices of the simplex  |
| `ctx.order`                | Ordering of objective function values              |
|                            |                                                    |
| `ctx.x_best`               | Best point (i.e. 1st one according to ordering)    |
| `ctx.f_best`               | Objective function at the best point               |
| `ctx.j_best`               | Index of the best point                            |
|                            |                                                    |
| `ctx.x_worst`              | Worst point                                        |
| `ctx.f_worst`              | Objective function at the worst point              |
| `ctx.j_worst`              | Index of the worst point                           |
|                            |                                                    |
| `ctx.x_2nd_worst`          | Second worst point                                 |
| `ctx.f_2nd_worst`          | Objective function at the second worst point       |
| `ctx.j_2nd_worst`          | Index of the second worst point                    |

## Simplex transform factors

| Property                   | Description                                        |
|:---------------------------|:---------------------------------------------------|
| `ctx.rho`                  | Reflection factor, ρ > 0                           |
| `ctx.chi`                  | Expansion factor, χ > 1                            |
| `ctx.gamma`                | Contraction factor, 0 < γ < 1                      |
| `ctx.sigma`                | Shrinkage factor, 0 < σ < 1                        |

## Counters

| Property                   | Description                                           |
|:---------------------------|:------------------------------------------------------|
| `ctx.iterations`           | Number of algorithm iterations                        |
| `ctx.evaluations`          | Number of evaluations of the objective function       |
| `ctx.reflections`          | Number of reflections applied to the simplex          |
| `ctx.expansions`           | Number of expansions applied to the simplex           |
| `ctx.outside_contractions` | Number of outside contractions applied to the simplex |
| `ctx.inside_contractions`  | Number of inside contractions applied to the simplex  |
| `ctx.shrinkages`           | Number of shrinkages applied to the simplex           |

## Algorithm variants

| Property                   | Description                                          |
|:---------------------------|:-----------------------------------------------------|
| `ctx.greedy_expansion`     | Whether to apply the "greedy expansion" strategy     |
| `ctx.recompute_centroid`   | Whether to recompute rather than update the centroid |

## Stopping criterion

| Property                   | Description                                                      |
|:---------------------------|:-----------------------------------------------------------------|
| `ctx.ftol`                 | Relative tolerance for the convergence in the objective function |
| `ctx.xtol`                 | Relative tolerance for the convergence in the variables          |
| `ctx.maxiters`             | Maximum number of allowed algorithm iterations                   |
| `ctx.maxevals`             | Maximum number of allowed evaluations of the objective function  |
| `ctx.LVR`                  | Linearized Volume Ratio of the simplex                           |
| `ctx.status`               | Status of the algorithm                                          |

The status of the algorithm is one of`⁽¹⁾`:

* `:initializing` if the algorithm has not yet started;

* `:searching` if the algorithm has not yet converged;

* `:convergence_in_f` if the algorithm has converged in the objective function;

* `:convergence_in_x` if the algorithm has converged in the variables;

* `:rounding_errors` if rounding errors prevent further progress;

* `:too_many_evaluations` if the maximum number of calls for the objective function
  have been exceeded;

* `:too_many_iterations` if the maximum number of algorithm iterations have been
  exceeded.

`⁽¹⁾` The status may also be set to another symbolic value of by the observer if the caller
opts to this possibility. During the search, the algorithm is stopped if its status becomes
different from `:searching`.

"""

# Extract the list of properties from their docstring.
const properties = let v = Set{Symbol}(), i = 1
    while true
        m = match(r"^ *\| *`ctx\.(\w+)` *\| *(.*?) *\| *$()"m, _PROPERTIES_DOCSTRING, i)
        m === nothing && break
        push!(v, Symbol(m.captures[1]))
        i = last(m.offsets) + 1
    end
    Tuple(sort(collect(v))) # result of this block
end
@doc _PROPERTIES_DOCSTRING properties

Base.propertynames(ctx::Context) = properties

Base.getproperty(ctx::Context, key::Symbol) = _getproperty(ctx, Val(key))
_getproperty(ctx::Context, ::Val{key}) where {key} = getfield(ctx, key)
_getproperty(ctx::Context, ::Val{:x_best})      = ctx.points[ctx.j_best]
_getproperty(ctx::Context, ::Val{:f_best})      = ctx.costs[ctx.j_best]
_getproperty(ctx::Context, ::Val{:x_worst})     = ctx.points[ctx.j_worst]
_getproperty(ctx::Context, ::Val{:f_worst})     = ctx.costs[ctx.j_worst]
_getproperty(ctx::Context, ::Val{:x_2nd_worst}) = ctx.points[ctx.j_2nd_worst]
_getproperty(ctx::Context, ::Val{:f_2nd_worst}) = ctx.costs[ctx.j_2nd_worst]
_getproperty(ctx::Context, ::Val{:vertices})    = @inbounds view(ctx.points, 1:ctx.n+1)

#-----------------------------------------------------------------------------------------
# ALGORITHM

# The predicate `is_better(ctx, fa, fb)` yields whether objective function `fa` is strictly
# better than `fb` according to the ordering `ctx.order` of objective objective function
# values.
is_better(ctx::Context, fa, fb) = Base.lt(ctx.order, fa, fb)

"""
    simplex(f, x0, args...; kwds...) -> x, fx, status, nf
    Simplex.solve(f, x0, args...; kwds...) -> x, fx, status, nf

Optimize the objective function `f` by the Nelder-Mead method with an initial simplex built
according to initial variables `x0` and argument(s) `args...`. In the most simple strategy
to build an initial simplex, `args...` is a single argument specifying the size of the
initial simplex size as a scalar or as an array of same shape as `x0` (see
[`Simplex.build_simplex!`](@ref)).

The result is a 4-tuple: `x` is the best solution found by the algorithm, `fx = f(x)` is the
corresponding objective function value, `status` is the final status of the algorithm, and
`nf` is the number of evaluations of the objective function. Call `issuccess(status)` to
figure out whether algorithm has converged.

In order to retrieve the complete algorithm state, call one of:

    simplex(Val(:context), f, x0, args...; kwds...) -> ctx
    Simplex.solve(Val(:context), f, x0, args...; kwds...) -> ctx

which yields a context `ctx` of type [`Simplex.Context`](@ref) that can be reused for
solving other similar problems (saving allocations) and whose content is available by the
`ctx.key` syntax (see [`Simplex.properties`](@ref)).

## Keywords

* `order` specifies the ordering of objective function values. With the default, `order =
  TotalMin`, the best function value is the smallest one while `NaN` and then `missing` are
  the worst values.

* `observer` can be set with a user defined function which is called as `observer(ctx, f,
  t)` at every iteration of the algorithm with `ctx` an instance of
  [`Simplex.Context`](@ref), `f` the objective function, and `t` the elapsed time in
  seconds. The observer may may return a symbolic status other than `:searching` to
  cleanly terminate the algorithm (any other type of result returned by the observer is
  silently ignored).

Other possible keywords are configurable options of the *Simplex* method (see
[`Simplex.configure!`](@ref)).

## See also

[`Simplex.Context`](@ref), [`Simplex.configure!`](@ref), [`Simplex.build_simplex!`](@ref),
and [`Simplex.solve!`](@ref).

"""
function simplex(::Val{:context}, f, x0::AbstractArray, args...; observer=nothing, kwds...)
    ctx = Context(f, x0, args...; kwds...)
    return solve!(ctx, f; observer=observer)
end

function simplex(f, x0::AbstractArray, args...; kwds...)
    ctx = simplex(Val(:context), f, x0, args...; kwds...)
    return ctx.status, ctx.x_best, ctx.f_best, ctx.evaluations
end

const solve = simplex

"""
    solve!(ctx::Simplex.Context, f, x0, args...; kwds...) -> ctx

Run the Nelder-Mead *Simplex* algorithm using context `ctx` to optimize the multivariate
objective function `f`. Initial variables `x0` and arguments `args...` are used to build the
initial simplex.

## See also

[`Simplex.Context`](@ref), [`Simplex.configure!`](@ref), [`Simplex.build_simplex!`](@ref),
and [`simplex`](@ref).

"""
function solve!(ctx::Context, f, x0::AbstractArray, args...; kwds...)
    # Reset the context because a new optimization will be performed.
    reset!(ctx)

    # Build the initial simplex.
    build_simplex!(ctx, x0, args...)

    # Optimize starting with the newly defined simplex.
    return solve!(ctx, f; kwds...)
end

"""
    solve!(ctx::Simplex.Context, f; restart::Bool=false, kwds...) -> ctx

Continue the optimization of the objective function `f` using the simplex currently defined
in context `ctx` and perhaps other configurable settings specified by `kwds...`. This method
may also be useful to improve the current solution in `ctx`.

Keyword `restart` indicates whether to re-evaluate the value of the objective function for
every points of the existing simplex. Other possible keywords `kwds...` are described in the
documentation of [`simplex`](@ref).

!!! warning
    Specify `restart=true` if `f` is another objective function than the one previously used
    with this context.

"""
function solve!(ctx::Context{T,F}, f; observer=nothing, restart::Bool=false, kwds...) where {T,F}
    # Set configurable parameters if any keywords specified.
    isempty(kwds) || configure!(ctx; kwds...)

    # Retrieve constant parameters.
    rho    = ctx.rho
    chi    = ctx.chi
    gamma  = ctx.gamma
    sigma  = ctx.sigma
    points = ctx.points
    costs  = ctx.costs
    n      = ctx.n

    # Check that all points have been allocated. This is a simple mean to detect that
    # context has been properly instantiated and initialized with a simplex.
    @inbounds for j in eachindex(points)
        isassigned(points, j) || throw_assertion_failed("build an initial simplex first")
    end

    # Starting time.
    t0 = time()

    # Evaluate the objective function at the vertices of the initial simplex.
    @inbounds for j in eachindex(costs)
        if restart || isnan(costs[j])
            costs[j] = f(points[j])
            ctx.evaluations += 1
        end
    end

    # To determine the transformation of the simplex, the reflection point is first computed
    # which a dedicated requires storage. Then, there are 2 cases:
    #
    # - If the reflection point is strictly better than the second worst point, the worst
    #   point will be replaced either by the reflection point or by the expansion point and
    #   the expansion point can thus override the worst point but not the reflection point.
    #
    # - If the reflection point is not strictly better than the second worst point, a
    #   contraction point (inside or outside) is computed and either the contraction point
    #   is accepted or the simplex is shrunk. The contraction point can thus override the
    #   reflection point but none of the points of the simplex.
    #
    # In addition to the n + 1 points of the simplex, the storage for only 2 other points is
    # thus needed: one for the centroid, the other for the reflection point and for the
    # contraction point.
    j_centroid = n + 2 # index of the centroid
    j_reflect  = n + 3 # index of the reflection (or contraction) point

    # Main loop of the algorithm.
    must_sort      = true   # force (partially) sorting the vertices
    must_recompute = true   # force recomputing the centroid from scratch
    ctx.status = :searching # status is used to track algorithm termination
    while true
        # Determine the indices and objective function of the best, worst, and second worst
        # vertices.
        j_worst_old = ctx.j_worst # remember index of worst point prior to update
        must_sort && partialsort!(ctx)
        j_best = ctx.j_best
        f_best = costs[j_best]
        j_worst = ctx.j_worst
        f_worst = costs[j_worst]
        j_2nd_worst = ctx.j_2nd_worst
        f_2nd_worst = costs[j_2nd_worst]

        # Check for termination in O(1) operations.
        if ctx.LVR ≤ ctx.xtol
            # Assume convergence in the variables if the "linearized volume ratio" of the
            # simplex is smaller than `xtol`.
            ctx.status = :convergence_in_x
        elseif abs(f_best - f_worst) ≤ ctx.ftol*max(abs(f_best), abs(f_worst))
            # Assume convergence in the objective function if the relative difference
            # between the best and worst objective function is smaller than `ftol`.
            ctx.status = :convergence_in_f
        elseif iszero(ctx.LVR) || iszero(f_best - f_worst)
            ctx.status = :rounding_errors
        elseif ctx.iterations ≥ ctx.maxiters
            ctx.status = :too_many_iterations
        elseif ctx.evaluations ≥ ctx.maxevals
            ctx.status = :too_many_evaluations
        end
        if observer != nothing
            # Call the user defined observer which may return another status which takes
            # priority.
            let status = observer(ctx, f, time() - t0)
                if status isa Symbol
                    ctx.status = status
                end
            end
        end
        ctx.status == :searching || return ctx

        # Update or recompute the centroid of the simplex vertices but the worst one.
        if must_recompute
            # Compute the centroid from scratch in O(n²) operations.
            compute_centroid!(ctx)
        elseif j_worst != j_worst_old
            # The centroid can be updated in O(n) operations. If the index of the worst
            # point has not changed, nothing has to be done.
            update_centroid!(ctx, j_worst_old)
        end

        # For the next iteration, assume that sorting cannot be skipped and, unless
        # configuration says otherwise, that the centroid can be simply updated. Skipping
        # the sorting is possible if just the worst point change (hence, not in case of a
        # shrinkage) and has an objective function value still strictly worse than that of
        # the second worst point. This can only occur for a contraction.
        must_sort = true
        must_recompute = ctx.recompute_centroid

        # To update the simplex, a few points may be tried:
        #
        #     x_reflect          = c - ρ⋅(x_worst   - c)
        #     x_expand           = c + χ⋅(x_reflect - c) = c - ρ⋅χ⋅(x_worst - c)
        #     x_contract_outside = c + γ⋅(x_reflect - c) = c - ρ⋅γ⋅(x_worst - c)
        #     x_contract_inside  = c + γ⋅(x_worst   - c)
        #
        # where `c` is the centroid of the simplex face opposite to the worst point. During
        # a shrinkage, each vertex `j` becomes (the shrinkage leaves the best vertex
        # unchanged):
        #
        #     x[j] <- x_best + σ⋅(x[j] - x_best)
        #
        # All these transforms are done by the `new_point!` method.
        #
        # The "linearized volume ratio" (LVR) was proposed by Singer and Singer (2004) as a
        # fast criterion -- computed in O(1) floating point operations -- to decide upon
        # convergence in the variables. This criterion is updated as:
        #
        #     LVR <- LVR⋅ρ^(1/n)     after a reflection
        #            LVR⋅(ρ⋅χ)^(1/n) after an expansion
        #            LVR⋅(ρ⋅γ)^(1/n) after an outside contraction
        #            LVR⋅γ^(1/n)     after an inside contraction
        #            LVR⋅σ           after a shrinkage
        #

        # Compute reflection point.
        x_reflect = new_point!(ctx, j_reflect, -rho, j_worst, j_centroid)
        f_reflect = convert(F, f(x_reflect))
        ctx.evaluations += 1
        if is_better(ctx, f_reflect, f_best)
            # Attempt expansion if the reflection point is strictly better than the best
            # point.
            j_expand = j_worst # the expansion point can override the worst point
            x_expand = new_point!(ctx, j_expand, chi, j_reflect, j_centroid)
            f_expand = convert(F, f(x_expand))
            ctx.evaluations += 1
            if is_better(ctx, f_expand, (ctx.greedy_expansion ? f_best : f_reflect))
                # Accept expansion point.
                replace_worst!(ctx, j_expand, f_expand)
                ctx.expansions += 1
                ctx.LVR *= nthroot(rho*chi, n)
            else
                # Accept reflection point.
                replace_worst!(ctx, j_reflect, f_reflect)
                ctx.reflections += 1
                ctx.LVR *= nthroot(rho, n)
            end
        elseif is_better(ctx, f_reflect, f_2nd_worst)
            # Accept reflection point.
            replace_worst!(ctx, j_reflect, f_reflect)
            ctx.reflections += 1
            ctx.LVR *= nthroot(rho, n)
        else
            # Contract between the centroid and the best of the worst and the reflection
            # point if the reflection point is not better than the second worst one. The
            # contraction is called an "outside contraction" if the reflection point is
            # strictly better than the worst point, an "inside contraction" otherwise.
            outside = is_better(ctx, f_reflect, f_worst)
            j_other = outside ? j_reflect : j_worst
            j_contract = j_reflect # the contraction point can override the reflection point
            x_contract = new_point!(ctx, j_contract, gamma, j_other, j_centroid)
            f_contract = convert(F, f(x_contract))
            ctx.evaluations += 1
            if (outside ? is_better(ctx, f_reflect, f_contract) : ! is_better(ctx, f_contract, f_worst))
                # Shrink simplex around the best vertex.
                for j in 1:n+1
                    j == j_best && continue # shrinkage leaves the best point unchanged
                    x = new_point!(ctx, j, sigma, j, j_best)
                    costs[j] = f(x)
                    ctx.evaluations += 1
                end
                ctx.shrinkages += 1
                ctx.LVR *= sigma
                must_recompute = true # force recomputing the centroid from scratch
            else
                # Accept contraction point. If the objective function at the contraction
                # point is worse than that of the second worst point, no sorting is
                # necessary.
                replace_worst!(ctx, j_contract, f_contract)
                must_sort = !is_better(ctx, f_2nd_worst, f_contract)
                if outside
                    ctx.outside_contractions += 1
                    ctx.LVR *= nthroot(rho*gamma, n)
                else
                    ctx.inside_contractions += 1
                    ctx.LVR *= nthroot(gamma, n)
                end
            end
        end
        ctx.iterations += 1
    end
end

# See https://github.com/JuliaLang/julia/issues/47565
nthroot(x::Real, n::Integer) =
    isodd(n) || x ≥ zero(x) ? copysign(abs(x)^(one(n)//n), x) : throw(DomainError(
        "Exponentiation yielding a complex result requires a complex argument. Replace `nthroot(x, n)` with `complex(x)^(1//n)`."))

#-----------------------------------------------------------------------------------------
# OPERATIONS ON THE SIMPLEX

# Find the best, worst, and second worst vertices in O(n) operations.
#
# Sorting is necessary after any simplex change except when the new point replacing the
# worst one is worst than the former second worst one.
function Base.partialsort!(ctx::Context)
    f = ctx.costs
    j_first = firstindex(f)
    j_last = lastindex(f)
    j_worst = j_2nd_worst = j_best = j_first
    for j in j_first+1:j_last
        if is_better(ctx, f[j], f[j_best])
            j_best = j
        elseif is_better(ctx, f[j_worst], f[j])
            j_2nd_worst = j_worst
            j_worst = j
        elseif is_better(ctx, f[j_2nd_worst], f[j])
            j_2nd_worst = j
        end
    end
    ctx.j_best = j_best
    ctx.j_worst = j_worst
    ctx.j_2nd_worst = j_2nd_worst
    return nothing
end

# Replace worst point by new one.
function replace_worst!(ctx::Context, j_new::Int, f_new::Real)
    # Replace the worst point by the new one if the index where is stored the new point is
    # different from that of the worst point.
    j_worst = ctx.j_worst
    if j_new != j_worst
        points = ctx.points
        if points isa VectorOfArrays
            # Just swap the points.
            temp = points[j_worst]
            points[j_worst] = points[j_new]
            points[j_new] = temp
        elseif points isa Matrix
            # Must copy values between columns.
            copy!(view(points, :, j_worst), view(points, :, j_new))
        else
            error("unknown storage for the vertices of the simplex")
        end
    end

    # Replace the objective function value.
    f = ctx.costs[j_worst] = f_new
    return nothing
end

# Compute the centroid of vertices but the worst one from scratch in O(n²) operations.
function compute_centroid!(ctx::Context)
    n        = ctx.n
    points   = ctx.points
    j_worst  = ctx.j_worst
    centroid = points[n + 2]
    init     = true
    for j in 1:n+1
        j == j_worst && continue
        if init
            copy!(centroid, points[j])
            init = false
        else
            add!(centroid, points[j])
        end
    end
    scale!(centroid, 1//n)
    return nothing
end

# Update the simplex centroid in O(n) operations. `j_worst_old` is the index of the worst
# point prior to its replacement.
function update_centroid!(ctx::Context, j_worst_old::Int)
    # Nothing to do if the worst vertex is at the same index as before.
    if j_worst_old != ctx.j_worst # FIXME redundant test
        n      = ctx.n
        points = ctx.points
        x_cen  = points[n + 2]
        x_new  = points[ctx.j_worst]
        x_old  = points[j_worst_old]
        scl    = one(floating_point_type(vertex_eltype(ctx)))/n
        @inbounds for i in eachindex(x_cen, x_old, x_new)
            x_cen[i] -= scl*(x_new[i] - x_old[i])
        end
    end
    return nothing
end

# Copy values.TODO Use OptimPack/LazyAlgebra vcopy!
copy!(ctx::Context, jdst::Int, jsrc::Int) = copy!(ctx.points[jdst], ctx.points[jsrc])
function copy!(dst::AbstractArray{D,N},
               src::AbstractArray{S,N}) where {D,S,N}
    if dst !== src
        axes(dst) == axes(src) || throw_dimension_mismatch("arrays have different axes")
        @inbounds for i in eachindex(dst, src)
            dst[i] = src[i]
        end
    end
    return dst
end

# In-place addition.TODO Use axpby!
add!(ctx::Context, jdst::Int, jsrc::Int) = add!(ctx.points[jdst], ctx.points[jsrc])
function add!(dst::AbstractArray{D,N},
              src::AbstractArray{S,N}) where {D,S,N}
    axes(dst) == axes(src) || throw_dimension_mismatch("arrays have different axes")
    @inbounds for i in eachindex(dst, src)
        dst[i] += src[i]
    end
    return dst
end

# Scaling of values. TODO Use OptimPack/LazyAlgebra vscale!.
scale!(ctx::Context, j::Int, s::Real) = scale!(ctx.points[j], s)
function scale!(A::AbstractArray, s::Real)
    if iszero(s)
        fill_with_zeros!(A)
    elseif !isone(s)
        s = convert_multiplier(eltype(A), s)
        @inbounds for i in eachindex(A)
            A[i] *= s
        end
    end
    return A
end

# TODO Use OptimPack/LazyAlgebra vzeros! and vnans!
fill_with_zeros!(A::AbstractArray) = fill!(A, zero(eltype(A)))
fill_with_NaNs!(A::AbstractArray) = fill!(A, NaN*zero(eltype(A)))

"""
    Simplex.new_point!(dst, alpha, pnt, org) -> dst

Compute a new point in `dst` by contracting or expanding the position of the point `pnt` by
a factor `alpha` relative to the point `org`. This amounts to do for all indices `i`:

    dst[i] = org[i] + alpha*(pnt[i] - org[i])

"""
new_point!(ctx::Context, jdst::Int, alpha::Real, jpnt::Int, jcen::Int) =
    new_point!(vertex(ctx, jdst), alpha, vertex(ctx, jpnt), vertex(ctx, jcen))

# TODO Use axpby!
function new_point!(dst::AbstractArray{T,N}, alpha::Real,
                    pnt::AbstractArray{T,N},
                    org::AbstractArray{T,N}) where {T,N}
    alpha = convert_multiplier(T, alpha)
    axes(dst) == axes(org) == axes(pnt) || throw_dimension_mismatch(
        "arrays must have the same axes")
    @inbounds for i in eachindex(dst, pnt, org)
        dst[i] = org[i] + alpha*(pnt[i] - org[i])
    end
    return dst
end

# Convert a scalar multiplier `s` to the correct floating-point type given the type `E` of
# the elements to be multiplied by `s`. TODO Use LazyAlgebra/OptimPack version of this.
convert_multiplier(::Type{E}, s::Real) where {E<:Number} = as(floating_point_type(E), s)

function simple_observer(ctx::Context)
    if ctx.iterations == 0
        println("#   ITER   EVAL     LVR              F_BEST                F_WORST")
        println("# ------ ------ ----------- ----------------------- -----------------------")
    end
    # ±X.FFFe±EEE witdh is number of F's + 8
    @printf("  %6d %6d %11.3e %23.15e %23.15e\n", ctx.iterations, ctx.evaluations,
            ctx.LVR, ctx.f_best, ctx.f_worst)
    status = ctx.status
    if runlevel > :searching
        color =
            runlevel == SUCCESS ? :green  :
            runlevel == WARNING ? :yellow : :red
        print("# Termination status: `")
        printstyled(":", ctx.status; color=color)
        println("`")
    end
    nothing
end


                                                                      end # module

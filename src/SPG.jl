"""
    OptinPack.SPG

This module implements the *Spectral Projected Gradient Method* (Version 2: "continuous
projected gradient direction") to find a local minimum of a multi-variate objective function
with convex constraints. The algorithm is described in:

- E. G. Birgin, J. M. Martinez, and M. Raydan, *"Nonmonotone spectral projected gradient
  methods on convex sets"*, SIAM Journal on Optimization **10**, pp. 1196-1211 (2000).

- E. G. Birgin, J. M. Martinez, and M. Raydan, *"SPG: software for convex-constrained
  optimization"*, ACM Transactions on Mathematical Software (TOMS) **27**, pp. 340-349
  (2001).

Original code can be found at www.ime.usp.br/~egbirgin/tango/.

"""
module SPG

export
    spg, spg!

using TypeUtils: @public
@public Context, configure!, solve!

using Printf
using LinearAlgebra
using Neutrals
using TypeUtils
using ArrayTools

using ..OptimPack
using ..OptimPack:
    Memory,
    adapt_multiplier_precision,
    axpby!,
    copy!,
    inner,
    isnothing,
    issomething,
    ordinal_suffix,
    print_seconds,
    scale!,
    tiny,
    throw_bad_argument,
    throw_dimension_mismatch,
    throw_assertion_failed,
    xpby!

# Default settings. All values are dimensionless.
const default_m = 100
const default_lmin = 1e-30
const default_lmax = 1e+30
const default_eps1 = 1e-5
const default_maxiters =  50_000
const default_maxevals = 100_000
const default_gamma = 1.0e-4
const default_sigma1 = 0.1
const default_sigma2 = 0.9

const _DOC_PROPERTIES = """
"""

"""
    SPG.Context{T,Xa,F,Ga}(undef) -> ctx

Create a context object for solving problems with the SPG algorithm. Parameter `T` is the
floating-point type for scalar computations. Parameter `Xa` is the type of the variables of
the problem. Parameter `F` is the type of the objective function value. Parameter `Ga` is
the type of the gradient.

## Properties

Context `ctx` has many properties accessible with the `ctx.key` syntax.

* Current variables:
  * `ctx.x`: Current point.
  * `ctx.f`: Objective function at current point.
  * `ctx.g`: Gradient at current point.
  * `ctx.gpsupn`: Sup-norm of projected gradient.

* Candidate solution:
  * `ctx.x_best`: Best point.
  * `ctx.f_best`: Objective function at best point.

* New iterate found by the line-search:
  * `ctx.x_new`: Trial point.
  * `ctx.f_new`: Objective function at trial point.
  * `ctx.g_new`: Gradient at trial point.

* Line-search:
  * `ctx.d`: Search direction or projected gradient.
  * `ctx.costs`: Memorized previous function values.
  * `ctx.gamma`: Parameter for Armijo's criterion.
  * `ctx.lambda`: Spectral step-length.
  * `ctx.lmin`: Lower bound for `ctx.lambda`, `0 < ctx.lmin < 1` must hold.
  * `ctx.lmax`: Upper bound for `ctx.lambda`, `1 < lmax < +∞` must hold.
  * `ctx.alpha`: Backtracking step-length, `0 < ctx.alpha ≤ 1` must hold.
  * `ctx.sigma1`: Lower absolute threshold for `ctx.alpha`.
  * `ctx.sigma2`: Upper relative threshold for `ctx.alpha`.

* Stopping criteria.:
  * `ctx.eps1`: Threshold for the sup-norm of the projected gradient.
  * `ctx.maxiters`: Maximum number of iterations.
  * `ctx.maxevals`: Maximum number of functional evaluations.
  * `ctx.status`: Algorithm status.

* Scaling factors:
  * `ctx.xscl`: Scaling factor for the variables.
  * `ctx.fscl`: Scaling factor for the objective function.

*  Counters:
  * `ctx.iterations`: Number of iterations.
  * `ctx.evaluations`: Number of objective function calls.
  * `ctx.gradients`: Number of gradient calls.
  * `ctx.projections`: Number of projections.
  * `ctx.Δt`: Elapsed time (seconds).

* Optional storage:
  * `ctx.save_memory`: Use optional work-spaces?
  * `ctx.s`: Storage for `ctx.x_new - ctx.x`, unused if `ctx.save_memory` is `true`.
  * `ctx.y`: Storage for `ctx.g_new - ctx.g`, unused if `ctx.save_memory` is `true`.

"""
mutable struct Context{T<:AbstractFloat,Xa,F,Ga,Xs}
    # Current variables
    x::Xa             # Current point
    f::F              # Objective function at current point
    g::Ga             # Gradient at current point
    gpsupn::Xs        # Sup-norm of projected gradient

    # Candidate solution
    x_best::Xa        # Best point
    f_best::F         # Objective function at best point

    # New iterate found by the line-search
    x_new::Xa          # Trial point
    f_new::F           # Objective function at trial point
    g_new::Ga          # Gradient at trial point

    # Line-search
    d::Xa             # Search direction or projected gradient
    costs::Memory{F}  # Memorized previous function values
    gamma::T          # Parameter for Armijo's criterion
    lambda::T         # Spectral step-length
    lmin::T           # Lower bound for lambda, 0 < lmin < 1
    lmax::T           # Upper bound for lambda, 1 < lmax < +∞
    alpha::T          # Backtracking step-length
    sigma1::T         # Lower absolute threshold for alpha
    sigma2::T         # Upper relative threshold for alpha

    # Stopping criteria
    eps1::T           # Threshold for the sup-norm of the projected gradient
    maxiters::Int     # Maximum number of iterations
    maxevals::Int     # Maximum number of functional evaluations
    status::Symbol    # Algorithm status

    # Scaling factors
    xscl::Xs          # Scaling factor for the variables
    fscl::F           # Scaling factor for the objective function

    # Counters
    iterations::Int   # Number of iterations
    evaluations::Int  # Number of objective function calls
    gradients::Int    # Number of gradient calls
    projections::Int  # Number of projections
    Δt::Float64       # Elapsed time (seconds)

    # Optional storage
    save_memory::Bool # Use optional work-spaces?
    s::Xa             # Storage for x_new - x, unused if save_memory is true
    y::Ga             # Storage for g_new - g, unused if save_memory is true

    function Context{T,Xa,F,Ga}(::UndefInitializer) where {T<:AbstractFloat,
                                                           Xa<:AbstractArray{<:Number},
                                                           F<:Number,
                                                           Ga<:AbstractArray{<:Number}}
        Xs = eltype(Xa)
        isconcretetype(Xs) && float(Xs) == Xs || error(
            "variables must have concrete floating-point elements")
        isconcretetype(F) && float(F) == F || error(
            "objective function must have concrete floating-point value")
        Gs = eltype(Ga)
        isconcretetype(Gs) && float(Gs) == Gs || error(
            "gradient of objective function must have concrete floating-point elements")
        get_precision(Gs) == get_precision(Xs) || error(
            "gradient and variables must have the same numerical precision")
        ndims(Ga) == ndims(Xa) || error(
            "gradient and variables must have the same number of dimensions")
        ctx = new{T,Xa,F,Ga,Xs}()
        ctx.f = NaN*oneunit(F)
        ctx.gpsupn = NaN*oneunit(Xs)
        ctx.f_best = NaN*oneunit(F)
        ctx.f_new = NaN*oneunit(F)
        ctx.gamma = default_gamma
        ctx.lambda = 𝟘
        ctx.lmin = default_lmin
        ctx.lmax = default_lmax
        ctx.alpha = 𝟘
        ctx.sigma1 = default_sigma1
        ctx.sigma2 = default_sigma2
        ctx.eps1 = default_eps1
        ctx.maxiters = default_maxiters
        ctx.maxevals = default_maxevals
        ctx.status = :undefined
        ctx.xscl = oneunit(eltype(Xs))
        ctx.fscl = oneunit(F)
        ctx.iterations = 0
        ctx.evaluations = 0
        ctx.gradients = 0
        ctx.projections = 0
        ctx.Δt = 𝟘
        ctx.save_memory = true

        return ctx
    end
end

"""
    SPG.Context(x, fx[, gx]; kwds...)
    SPG.Context{T}(x, fx[, gx]; kwds...)

Create a new context for the SPG method with initial variables `x`, objective function value
`fx`, and, if specified, gradient `gx`. The returned structure will share its storage with
`x` and `g` (use `copy` if that's not desirable). `T` is the floating-point type for scalar
computations; if not specified, it is automatically inferred from the arguments.

## Keywords

See [`SPG.configure!`](@ref) for possible keywords to set algorithms configurable
parameters.

In addition to these, the following keywords are available (to save some computations):

* `projections` specifies the initial number of projections onto the feasible set. If
  unspecified, `projections = 0` is assumed; otherwise, if `projections ≥ 1`, it will be
  assumed by the next call to `SPG.solve!(ctx)` that `x` are feasible.

* `evaluations` specifies the initial number of evaluations of the objective function. If
  unspecified, `evaluations = 0` is assumed; otherwise, if `evaluations ≥ 1`, it will be
  assumed by the next call to `SPG.solve!(ctx)` that `fx = f(x)` holds.

* `gradients` specifies the initial number of computations of the gradient of the objective
  function. If unspecified, `gradients = 0` is assumed; otherwise, if `gradients ≥ 1`, it
  will be assumed by the next call to `SPG.solve!(ctx)` that `gx = ∇f(x)` holds.

"""
function Context(x::AbstractArray, fx::Number; kwds...)
    # Numerical precision for scalars is at least double-precision.
    T = get_precision(Float64, eltype(x), typeof(fx))
    return Context{T}(x, fx; kwds...)
end

function Context(x::AbstractArray, fx::Number, gx::AbstractArray; kwds...)
    # Numerical precision for scalars is at least double-precision.
    T = get_precision(Float64, eltype(x), typeof(fx), eltype(gx))
    return Context{T}(x, fx, gx; kwds...)
end

function Context{T}(x::AbstractArray, fx::Number; kwds...) where {T<:AbstractFloat}
    # Check element type of variables.
    Xs = eltype(x)
    (isconcretetype(Xs) && Xs == float(Xs)) || throw_bad_argument(
        "variables must have floating-point elements")

    # Gradient has the same numerical precision as variables but may have different units.
    Gs = adapt_precision(get_precision(Xs), typeof(oneunit(fx)/oneunit(Xs)))
    gx = similar(x, Gs)

    return Context{T}(x, fx, gx; kwds...)
end

function Context{T}(x::Xa, fx::F, gx::Ga;
                    m::Integer = default_m,
                    evaluations::Integer = 0,
                    gradients::Integer = 0,
                    projections::Integer = 0,
                    kwds...) where {T<:AbstractFloat,
                                    Xa<:AbstractArray{<:Number},
                                    F<:Number,
                                    Ga<:AbstractArray{<:Number}}
    axes(x) == axes(gx) || throw_dimension_mismatch(
        "variables and gradient must have the same axes")
    firstindex(x) == firstindex(gx) || throw_dimension_mismatch(
        "variables and gradient must have the same first linear index")
    m ≥ 1 || throw_bad_argument("`m ≥ 1` must hold, got `m = $m`")
    ctx = Context{T,Xa,F,Ga}(undef)
    ctx.x = x
    ctx.f = fx
    ctx.g = gx
    ctx.costs = Memory{F}(undef, m)
    ctx.evaluations = evaluations
    ctx.gradients   = gradients
    ctx.projections = projections
    return configure!(ctx; kwds...)
end

"""
    SpectralProjectedGradient.Context(func, grad!, proj!, x0; kwds...)

Create a new context for the SPG method with initial variables `x0` and given callable
objects `func`, `grad!`, and `proj!` to respectively compute the objective function, compute
the gradient of the objective function, and project the variables onto the feasible set.

"""
function Context(func, grad!, proj!, x0::AbstractArray; kwds...)
    x = similar(x0, float(eltype(x0)))
    copy!(x, x0)
    proj!(x)
    fx = func(x)
    return Context(x, fx; kwds..., evaluations=1, projections=1, gradients=0)
end

# Extend methods in other modules.
OptimPack.configure!(ctx::Context; kwds...) = configure!(ctx::Context; kwds...)
OptimPack.solve!(ctx::Context, args...; kwds...) = solve!(ctx::Context, args...; kwds...)
LinearAlgebra.issuccess(ctx::Context) = ctx.status == :convergence

status_summary(ctx::Context) = status_summary(ctx.status)
status_summary(sym::Symbol) =
    sym == :undefined            ? "Algorithm net yet started" :
    sym == :searching            ? "Search in progress" :
    sym == :convergence          ? "Convergence in the sup-norm of the projected gradient" :
    sym == :rounding_errors      ? "Rounding errors prevent progress" :
    sym == :too_many_iterations  ? "Too many algorithm iterations" :
    sym == :too_many_evaluations ? "Too many evaluations of the objective function" :
    sym == :value_error          ? "Unexpected value computed by the algorithm" :
    unknown_status(sym)

@noinline unknown_status(sym::Symbol) = "Unknown status `:$sym`"

function Base.show(io::IO, ::MIME"text/plain", ctx::Context)
    @lock io begin
        print(io, "\n• Algorithm: Spectral Projected Gradient (SPG) method")
        print(io, "\n\n• Problem size")
        print(io, "\n  Memorized functional values: ", isdefined(ctx, :costs) ? length(ctx.costs) : 0)
        print(io, "\n  Number of variables:         ", isdefined(ctx, :x) ? length(ctx.x) : 0)
        print(io, "\n\n• Status: ")
        print(io, status_summary(ctx), " (")
        if issuccess(ctx)
            printstyled(io, "success"; color=:green)
        elseif ctx.status == :undefined
            printstyled(io, Symbol(ctx.status); color=:yellow)
        else
            printstyled(io, "failure"; color=:red)
        end
        print(io, ")")
        if ctx.evaluations > 0
            print(io, "\n\n• Candidate solution")
            print(io, "\n  Best f(x):                      ", ctx.f_best)
            print(io, "\n  Sup-norm of projected gradient: ", ctx.gpsupn)
            print(io, "\n\n• Counters")
            print(io, "\n  Iterations:   ", ctx.iterations)
            print(io, "\n  f(x) calls:   ", ctx.evaluations)
            print(io, "\n  ∇f(x) calls:  ", ctx.gradients)
            print(io, "\n  P_Ω(x) calls: ", ctx.projections)
            print(io, "\n  Elapsed time: "); print_seconds(io, ctx.Δt)
        end
    end
end

function instantiate!(ctx::Context)
    # First, create all needed work-spaces.
    isdefined(ctx, :x) || error("work-space for variables must be defined in the context")
    isdefined(ctx, :g) || error("work-space for gradient must be defined in the context")
    for key in (:d, :x_new, :x_best, :s,)
        key == :s && ctx.save_memory && continue
        if !isdefined(ctx, :key)
            setfield!(ctx, key, similar(ctx.x))
        end
    end
    for key in (:g_new, :y)
        key == :y && ctx.save_memory && continue
        if !isdefined(ctx, :key)
            setfield!(ctx, key, similar(ctx.g))
        end
    end
    # Second, check axes and indices of all defined work-spaces.
    start = firstindex(ctx.x)
    shape = axes(ctx.x)
    for key in (:d, :x_new, :x_best, :g, :g_new, :s, :y)
        isdefined(ctx, :key) || continue
        axes(getfield(ctx, key)) == shape || throw_dimension_mismatch(
            "work-space `$key` have incompatible axes")
        firstindex(getfield(ctx, key)) == start || throw_dimension_mismatch(
            "work-space `$key` have incompatible first linear index")
    end
    return ctx
end

"""
    SPG.configure!(ctx::SPG.Context; kwds...) -> ctx

Configure the parameters of context `ctx` for the *Spectral Projected Gradient* (SPG)
method for minimizing a multi-variate objective function under convex constraints.

The configurable parameters correspond to the following keywords:

* `xscl` and `fscl` are the respective scaling factors for the variables `x` and the
  objective function `f(x)` of the problem. `xscl` must have the same units, if any, as the
  elements of `x` and `fscl` must have the same units, if any, as `f(x)` so that the other
  parameters of the SPG method are dimensionless reals. Initially, `xscl =
  oneunit(eltype(x))` and `fscl = oneunit(f(x))`.

* `eps1` specifies the threshold for the convergence criterion `‖pg‖_∞ ≤ eps1*xscl` with
  `pg` the projected gradient. Initially, `eps1 = $default_eps1`.

* `gamma` specifies the parameter of the non-monotone Armijo-like stopping criterion.
  Initially, `gamma = 1e-4`.

* `lmin` and `lmax` specify safeguard bounds for the spectral step-length. Initially, `lmin
  = $default_lmin` and `lmax = $default_lmax`.

* `sigma1` and `sigma2` specify safeguard bounds for the quadratic interpolation step.
  Initially, `sigma1 = $default_sigma1` and `sigma2 = $default_sigma2`.

* `maxiters` specifies the maximum number of iterations. Initially, `maxiters` is
  practically unlimited.

* `maxevals` specifies the maximum number of evaluations of the objective function.
  Initially, `maxevals` is practically unlimited.

* `save_memory` specifies whether to not explicitly store `x - x_new` and `g - g_new` to
  save storage. Initially, `save_memory = true`. It may be required to choose `save_memory =
  false`, if the variables (and the gradient) are stored in non-conventional memory like GPU
  arrays.

## See also

[`SPG.Context`](@ref).

"""
function configure!(ctx::Context;
                    eps1::Real = ctx.eps1,
                    maxiters::Integer = ctx.maxiters,
                    maxevals::Integer = ctx.maxevals,
                    xscl::Number = ctx.xscl,
                    fscl::Number = ctx.xscl,
                    lmin::Real = ctx.lmin,
                    lmax::Real = ctx.lmax,
                    gamma::Real = ctx.gamma,
                    sigma1::Real = ctx.sigma1,
                    sigma2::Real = ctx.sigma2,
                    save_memory::Bool = ctx.save_memory)
    # Check settings.
    eps1 ≥ zero(eps1)                   || throw_bad_argument("`eps1` must be non-negative")
    maxiters ≥ 0                        || throw_bad_argument("`maxiters` must be non-negative")
    maxevals ≥ 1                        || throw_bad_argument("`maxevals` must be at least 1")
    isfinite(xscl) && xscl > zero(xscl) || throw_bad_argument("`xscl` must be finite and strictly positive")
    isfinite(fscl) && fscl > zero(fscl) || throw_bad_argument("`fscl` must be finite and strictly positive")
    𝟘 < lmin < 𝟙                        || throw_bad_argument("`0 < lmin < 1` must hold")
    lmax > 𝟙                            || throw_bad_argument("`lmax > 1` must hold")
    zero(gamma) < gamma < 1//2          || throw_bad_argument("`0 < gamma < 1/2` must hold")
    𝟘 < sigma1 < sigma2 < 𝟙             || throw_bad_argument("`0 < sigma1 < sigma2 < 1` must hold")

    # Instantiate context.
    ctx.eps1 = eps1
    ctx.maxiters = maxiters
    ctx.maxevals = maxevals
    ctx.xscl = xscl
    ctx.fscl = fscl
    ctx.lmin = lmin
    ctx.lmax = lmax
    ctx.gamma = gamma
    ctx.sigma1 = sigma1
    ctx.sigma2 = sigma2
    ctx.save_memory = save_memory
    return ctx
end

"""
    spg(func, grad!, proj!, x0; kwds...) -> ctx

Run the *"Spectral Projected Gradient"* method to solve the constrained problem:

    minₓ f(x)    subject to    x ∈ Ω ⊆ ℝ^n

with `f(x)` a multivariate objective function, `Ω` a convex set, and `n` the number of
variables.


## Arguments

* `func` is a callable object implementing the objective function. `func(x)` shall yield
  `f(x)`, the value taken by the objective function for feasible variables `x ∈ Ω`.

* `grad!` is a callable object implementing the gradient of the objective function.
  `grad!(g, x)` with `x ∈ Ω` shall store `∇f(x)` in provided `g`.

* `proj!` is a callable object implementing the orthogonal projection onto the convex set
  `Ω`. `proj!(x)` with `x ∈ ℝ^n` shall overwrite `x` with it orthogonal projection onto `Ω`.

* `x0 ∈ ℝ^n` is the starting point, it may not be feasible. It is left unchanged by the
  algorithm.


## Result

`ctx` is an instance of [`SPG.Context`](@ref) storing the algorithm state, in particular:
`ctx.x_best` is the best feasible point found by the algorithm, `ctx.f_best` is the
corresponding value of the objective function, and `ctx.status` is a symbolic constant
indicating the reason of the termination of the algorithm. `issuccess(ctx)` yields whether
algorithm was successful.

!!! note
    The returned context may be used to solve similar problems with [`SPG.solve!`](@ref) but
    with no further allocations.


## See also

[`spg!`](@ref) is an in-place version of the algorithm.

[`SPG.configure!`](@ref) and [`SPG.solve!`](@ref) for possible keywords.


## Running tests with `CUTEst` and `NLPModels`

`spg` may be used on one of the tests provided by `CUTEst` and `NLPModels` as follows:

```julia
using CUTEst, NLPModels, Revise, OptimPack
nlp = CUTEstModel{Float64}("BDEXP")
try
    ctx = spg(x -> obj(nlp, x), (g, x) -> copy!(g, grad(nlp, x)),
              Box(nlp.meta.lvar, nlp.meta.uvar), nlp.meta.x0)
finally
    finalize(nlp)
end
```

"""
function spg(func, grad!, proj!, x0::AbstractArray; kwds...)
    # Variables are similar to x0 but with floating-point elements.
    x = similar(x0, float(eltype(x0)))
    copy!(x, x0)

    # Call the in-place version of the algorithm.
    return spg!(func, grad!, proj!, x; kwds...)
end

"""
    spg!(func, grad!, proj!, x; kwds...) -> ctx

In-place version of [`spg`](@ref) to run the *"Spectral Projected Gradient"* method. The
only difference is that, `x` provides the starting point (possibly unfeasible) and is
overwritten by the best point found by the algorithm.

## See also

[`spg`](@ref) is an out-of-place version of the algorithm.

[`SPG.configure!`](@ref) and [`SPG.solve!`](@ref) for possible keywords.

"""
function spg!(func, grad!, proj!, x::AbstractArray;
              reset::Bool=false, observer=nothing, kwds...)
    # Project initial guess and compute the corresponding objective function value whose
    # type is needed to create the context.
    proj!(x)
    fx = func(x)
    ctx = Context(x, fx; kwds..., projections=1, evaluations=1)
    return solve!(ctx, func, grad!, proj!; reset=reset, observer=observer)
end

"""
    SPG.solve!(ctx, func, grad!, proj![, x0]; kwds...) -> ctx

Solve a constrained optimization problem by the *"Spectral Projected Gradient"* method.

## Arguments

* `ctx` is an instance of [`SPG.Context`](@ref) with all parameters and storage needed by
  the method. The content of `ctx` is updated to store the result of the algorithm.

* `func`, `grad!`, and `proj!` are callable to compute the objective function, its
  gradient, and to project the variables onto the feasible set.

* If a starting point `x0` is provided, it is copied into `ctx.x`; otherwise, it is assumed
  that `ctx.x` is the starting point.


## Keywords

See [`SPG.configure!`](@ref) for possible keywords to set algorithms configurable
parameters.

The following additional keyword is available:

* `observer` specifies a callable object to be called as `observer(ctx)` at every iteration
  of the algorithm. This may be used to print information or to implement other stopping
  criteria. To that end, `observer` may set `ctx.status` to another symbolic constant than
  `:searching`; another possibility is to have `observer` return `:searching` to continue
  the iterations or any other symbol to stop the algorithm (a returned value that is not a
  `Symbol` is ignored).

If a starting point `x0` *is not provided*, it is assumed that `ctx.x` is the starting
point and the following additional keywords are available:

* `projections` specifies the initial number of projections onto the feasible set. If
  `projections ≥ 1`, it is assumed that the starting point in `ctx.x` is feasible.

* `evaluations` specifies the initial number of evaluations of the objective function. If
  `evaluations ≥ 1`, it is assumed that `ctx.f = f(ctx.x)` holds.

* `gradients` specifies the initial number of computations of the gradient of the objective
  function. If `gradients ≥ 1`, it is assumed that `ctx.g = ∇f(ctx.x)` holds.

* `reset = true` may be specified to reset the initial number of projections, of function
  and of gradient evaluations to `0` and thus avoid the above assumptions. This is useful if
  the problem has changed (that is `func`, `grad!`, and/or `proj!` are different) or if
  `ctx.x` has been modified to store a starting point which is not the solution found by a
  previous run of the algorithm.


## See also

[`SPG.configure!`](@ref) for configurable parameters.

[`SPG.Context`](@ref) for creating a context.

[`spg`](@ref) for a description of the *"Spectral Projected Gradient"* method.

"""
function solve!(ctx::Context, func, grad!, proj!, x0::AbstractArray; kwds...)
    copy!(ctx.x, x0)
    return solve!(ctx, func, grad!, proj!; kwds..., reset=true)
end

function solve!(ctx::Context{T}, func, grad!, proj!;
                reset::Bool=false, observer=nothing, kwds...) where {T<:AbstractFloat}
    # Apply settings and allocate missing work-spaces.
    isempty(kwds) || configure!(ctx; kwds...)
    instantiate!(ctx)

    # Reset memorized objective function values.
    fill!(ctx.costs, typemin(eltype(ctx.costs)))

    # Scaling factors.
    fscl = ctx.fscl
    xscl = ctx.xscl
    gscl = convert(eltype(ctx.g), fscl/xscl)

    # Process initial point.
    if reset || ctx.projections < 1
        # Make the initial variables feasible.
        proj!(ctx.x)
        ctx.projections = 1
        ctx.evaluations = 0
        ctx.gradients = 0
    end
    if reset || ctx.evaluations < 1
        # Compute the objective function at the initial point.
        ctx.f = func(ctx.x)
        ctx.evaluations = 1
        ctx.gradients = 0
    end
    if reset || ctx.gradients < 1
        # Compute the gradient of the objective function at the initial point.
        grad!(ctx.g, ctx.x)
        ctx.gradients = 1
    end

    # Main loop.
    t0 = time()
    best_evals = -1
    ctx.iterations = 0
    ctx.status = :searching
    while true
        # Store best solution and functional value.
        if ctx.iterations < 1 || ctx.f < ctx.f_best
            ctx.f_best = ctx.f
            copy!(ctx.x_best, ctx.x)
            best_evals = ctx.evaluations
        end

        # Compute continuous-project-gradient and its sup-norm.
        xpby!(ctx.d, ctx.x, -xscl/gscl, ctx.g)
        proj!(ctx.d)
        ctx.projections += 1
        xpby!(ctx.d, ctx.d, -𝟙, ctx.x)
        ctx.gpsupn = sup_norm(ctx.d)

        # Check stopping criteria.
        if ctx.gpsupn ≤ zero(ctx.gpsupn) || ctx.gpsupn ≤ ctx.eps1*xscl || isnan(ctx.gpsupn)
            ctx.status = (ctx.gpsupn ≥ zero(ctx.gpsupn) ? :convergence : :value_error)
        elseif ctx.iterations ≥ ctx.maxiters
            ctx.status = :too_many_iterations
        elseif ctx.evaluations ≥ ctx.maxevals
            ctx.status = :too_many_evaluations
        end

        # Call observer with new iterate.
        if observer != nothing
            ctx.Δt = time() - t0
            let r = observer(ctx)
                if r isa Symbol
                    ctx.status = r
                end
            end
        end

        # Stop algorithm if requested.
        if ctx.status != :searching
            break
        end

        if ctx.iterations < 1
            # Set initial step-length knowing that the sup-norm of the projected gradient is
            # strictly positive.
            ctx.lambda = xscl/ctx.gpsupn # NOTE also convert to type T
            ctx.lambda = clamp(ctx.lambda, ctx.lmin, ctx.lmax)
        end

        # Save functional value for the non-monotone line search.
        ctx.costs[(ctx.iterations % length(ctx.costs)) + 1] = ctx.f

        # Compute first trial point for line-search.
        xpby!(ctx.x_new, ctx.x, -ctx.lambda*(xscl/gscl), ctx.g)
        proj!(ctx.x_new)
        ctx.projections += 1

        # Call non-monotone line-search method.
        linesearch!(ctx, func)
        if ctx.status != :searching
            # Line-search failed.
            if ctx.evaluations > best_evals && ctx.f_new < ctx.f_best
                # Some improvement achieved, update the best solution, pretend an ultimate
                # iteration has been done, and call observer if any.
                ctx.f_best = ctx.f_new
                copy!(ctx.x_best, ctx.x_new)
                ctx.iterations += 1
                if observer != nothing
                    ctx.Δt = time() - t0
                    observer(ctx)
                end
            end
            break
        end

        # Compute the gradient at the new iterate produced by the line-search.
        grad!(ctx.g_new, ctx.x_new)
        ctx.gradients += 1

        # Compute sts = ⟨s,s⟩, sty = ⟨s,y⟩ with s = x_new - x and y = g_new - g, update the
        # iterate, and compute the next spectral step-length. TODO check for rounding errors if sts ≤ 0
        ctx.lambda = ctx.lmax # assume a negative curvature
        if ctx.save_memory
            sts = zero(xscl*xscl)
            sty = zero(xscl*gscl)
            @inbounds @simd for i in eachindex(ctx.x, ctx.x_new, ctx.g, ctx.g_new)
                sᵢ = ctx.x_new[i] - ctx.x[i]
                yᵢ = ctx.g_new[i] - ctx.g[i]
                sts += oftype(sts, sᵢ*sᵢ)
                sty += oftype(sty, sᵢ*yᵢ)
                ctx.x[i] = ctx.x_new[i]
                ctx.g[i] = ctx.g_new[i]
            end
            if sty > zero(sty)
                ctx.lambda = (sts*gscl)/(sty*xscl)
            end
        else
            xpby!(ctx.s, ctx.x_new, -𝟙, ctx.x)
            xpby!(ctx.y, ctx.g_new, -𝟙, ctx.g)
            sts = inner(ctx.s, ctx.s)
            sty = inner(ctx.s, ctx.y)
            copy!(ctx.x, ctx.x_new)
            copy!(ctx.g, ctx.g_new)
            if sty > zero(sty)
                ctx.lambda = (sts*gscl)/(sty*xscl)
            end
        end
        ctx.lambda = clamp(ctx.lambda, ctx.lmin, ctx.lmax)
        ctx.f = ctx.f_new;
        ctx.iterations += 1
    end

    # Make sure `x` contains the best solution (this may be expected if `spg!` is directly
    # called).
    if ctx.iterations > 0 && ctx.f_best < ctx.f
        copy!(ctx.x, ctx.x_best)
        ctx.f = ctx.f_best
    end

    # Update elapsed time and return.
    ctx.Δt = time() - t0
    return ctx
end

function linesearch!(ctx::Context, func)
    # Compute the search direction given `x`, the feasible point at the start of the
    # line-search, and `x_new`, the first feasible point to try.
    xpby!(ctx.d, ctx.x_new, -𝟙, ctx.x)
    ctx.alpha = 𝟙 # corresponding step-length
    # Compute the parameters of the Armijo's stopping criterion.
    fmax = maximum(ctx.costs)
    gtd = inner(ctx.g, ctx.d)
    if !(gtd < zero(gtd))
        # Search direction is not a descent direction.
        if iszero(sup_norm(ctx.d))
            # `xnew` and `x` are equal.
            ctx.f_new = ctx.f
            ctx.status = :rounding_errors
        else
            ctx.status = :value_error
        end
        return
    end
    # Threshold to detect rounding errors. If the step-length `alpha` becomes smaller or
    # equal this, Armijo's condition automatically holds.
    alpha_min = tiny(fmax/(ctx.gamma*gtd))/2
    # Adjust the step length until one of the stopping criteria hold.
    while true
        # Evaluate objective function at trial point.
        ctx.f_new = func(ctx.x_new)
        ctx.evaluations += 1
        # Check for stopping criteria.
        if ctx.f_new ≤ fmax + ctx.gamma*ctx.alpha*gtd
            break
        elseif ctx.evaluations ≥ ctx.maxevals
            ctx.status = :too_many_evaluations
            break
        elseif ctx.alpha ≤ alpha_min
            ctx.status = :rounding_errors
            break
        end
        # Reduce the step length by a safeguarded quadratic interpolation.
        if ctx.alpha ≤ ctx.sigma1
            ctx.alpha /= 2
        else
            num = -gtd*ctx.alpha^2
            den = 2*(ctx.f_new - ctx.f - ctx.alpha*gtd)
            tmp = num/den
            if ctx.sigma1 ≤ tmp ≤ ctx.sigma2*ctx.alpha
                ctx.alpha = tmp
            else
                ctx.alpha /= 2
            end
        end
        # Next trial point (automatically feasible because we are backtracking and the
        # feasible set is convex).
        xpby!(ctx.x_new, ctx.x, ctx.alpha, ctx.d)
    end
end

end # module

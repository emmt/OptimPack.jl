"""
    OptimPack.LinearConjugateGradient

This module implements the *Linear Conjugate Gradient* method to solve a linear equation
`A*x = b` in `x` with `A` symmetric positive-definite or, equivalently, to minimize a
quadratic function `f(x) = (A*x - 2b)'*x + c`. The method may use a preconditioner. The
conjugate gradient method originally stems from:

- M. R. Hestenes & E. Stiefel, *"Methods of Conjugate Gradients for Solving
  Linear Systems"*, Journal of Research of the National Bureau of Standards
  **49**, pp. 409-436 (1952).

"""
module LinearConjugateGradient

# TODO In context, have specific types for `eltype(x)`, `f(x)`, and `eltype(∇f(x))`,
#      possibly with units but all with the same precision. See SPG code for an example.
#
#      Tx = eltype(x)
#      Tg = eltype(b)
#      Tf = typeof(zero(Tx)*zero(Tg))
#
#      If there are units, then the pre-conditioner must be at least a scaled identity.
#
# TODO To avoid rounding errors, scalars (in algorithm and context) must be at least on
#      double-precision.
#
# TODO Allow for explicitly specifying the loop-style in vectorized operations.

export
    issuccess,
    conjgrad,
    conjgrad!,
    mul!

# Public symbols that are not exported.
using TypeUtils: @public
@public Context, configure!, restart!, solve!

using OptimPack
using OptimPack: axpby!, configure!, inner, restart!, solve!, two_norm

using LinearAlgebra
using Printf
using Neutrals
using TypeUtils

const default_maxiters = typemax(Int)
default_restart(n::Integer) = min(50, Int(n)::Int)
default_ftol(::Type{T}) where {T<:Number} = (zero(T), 1e-8)
default_gtol(::Type{T}) where {T<:Number} = (zero(T), 1e-5)
default_xtol(::Type{T}) where {T<:Number} = (zero(T), 1e-6)

struct Identity <: Function end
LinearAlgebra.mul!(dst, A::Identity, src) = OptinPack.copy!(dst, src)

mutable struct Context{V}
    # Work-spaces.
    p::V # current search direction
    q::V # q = A*p
    r::V # current residuals: r = A*x - b
    z::V # preconditioned residuals: z = M*r

    # Configurable settings.
    maxiters::Int # maximum number of iterations
    restart::Int # number of iterations between restarts
    ftol::NTuple{2,Float64} # absolute and relative tolerances for function reduction
    gtol::NTuple{2,Float64} # absolute and relative tolerances for gradient norm
    xtol::NTuple{2,Float64} # absolute and relative tolerances for norm of variables change

    # Algorithm state.
    rho::Float64     # ‖∇f(x)‖²_M, squared gradient norm
    psi::Float64     # Δf(x), variation of function in last 2 iterations
    psimax::Float64  # Maximal variation of function in successive iterations
    elapsed::Float64 # Number of seconds since start of algorithm
    iterations::Int  # Number of iterations
    evaluations::Int # Number of multiplications by A
    status::Symbol   # Algorithm status

    function Context(p::V, q::V, r::V, z::V=r; kwds...) where {V}
        Tx = eltype(p)
        Tg = eltype(q)
        Tf = typeof(zero(Tx)*zero(Tg))
        ctx = new{V}(p, q, r, z)
        ctx.maxiters = default_maxiters
        ctx.restart = default_restart(length(p))
        ctx.ftol = default_ftol(Tf)
        ctx.gtol = default_gtol(Tg)
        ctx.xtol = default_xtol(Tx)
        restart!(ctx)
        isempty(kwds) || configure!(ctx; kwds...)
        return ctx
    end
end

"""
    LinearConjugateGradient.reason(status) -> str

yields a textual description of the status returned by the conjugate gradient
method.

"""
function reason(ctx::Context)
    status = ctx.status
    status === :not_positive_definite ? "LHS matrix is not positive definite" :
    status === :too_many_iterations   ? "too many iterations" :
    status === :f_test_satisfied      ? "function reduction test satisfied" :
    status === :g_test_satisfied      ? "gradient test satisfied" :
    status === :x_test_satisfied      ? "variables change test satisfied" :
    status === :searching             ? "work in progress" :
    status === :none                  ? "algorithm not started" :
    "unknown conjugate gradient status"
end

LinearAlgebra.issuccess(ctx::Context) =
    ctx.status ∈ (:f_test_satisfied, :g_test_satisfied, :x_test_satisfied)

function OptimPack.restart!(ctx::Context, status::Symbol = :none)
    ctx.rho = 𝟘
    ctx.psi = 𝟘
    ctx.psimax = 𝟘
    ctx.elapsed = 𝟘
    ctx.iterations = 0
    ctx.evaluations = 0
    ctx.status = status
    return ctx
end

function OptimPack.configure!(ctx::Context;
                              maxiters::Integer = ctx.maxiters,
                              restart::Integer = ctx.restart,
                              ftol::Tuple{Number,Real} = ctx.ftol,
                              gtol::Tuple{Number,Real} = ctx.gtol,
                              xtol::Tuple{Number,Real} = ctx.xtol)
    maxiters ≥ 𝟘 || bad_argument(
        "bad maximum number of iterations (maxiters = ", maxiters, ")")
    ftol[1] ≥ zero(ftol[1]) || bad_argument(
        "bad function reduction absolute tolerance (ftol[1] = ", ftol[1], ")")
    𝟘 ≤ ftol[2] < 𝟙 || bad_argument(
        "bad function reduction relative tolerance (ftol[2] = ", ftol[2], ")")
    gtol[1] ≥ zero(gtol[1]) || bad_argument(
        "bad gradient absolute tolerance (gtol[1] = ", gtol[1], ")")
    𝟘 ≤ gtol[2] < 𝟙 || bad_argument(
        "bad gradient relative tolerance (gtol[2] = ", gtol[2], ")")
    xtol[1] ≥ zero(xtol[1]) || bad_argument(
        "bad variables change absolute tolerance (xtol[1] = ", xtol[1], ")")
    𝟘 ≤ xtol[2] < 𝟙 || bad_argument(
        "bad variables change relative tolerance (xtol[2] = ", xtol[2], ")")
    ctx.maxiters = maxiters
    ctx.restart = restart
    ctx.ftol = ftol
    ctx.gtol = gtol
    ctx.xtol = xtol
    return ctx
end

function Base.show(io::IO, ctx::Context{V}) where {V}
    print(io, "LinearConjugateGradient.Context(")
    print(io, "    x::$V; # array of size $(size(ctx.r))\n")
    print(io, "    precond = $(ctx.z !== ctx.r),\n")
    print(io, "    maxiters = $(ctx.maxiters),\n")
    print(io, "    restart = $(ctx.restart),\n")
    print(io, "    ftol = $(ctx.ftol),\n")
    print(io, "    gtol = $(ctx.gtol),\n")
    print(io, "    xtol = $(ctx.xtol))")
end

"""
    LinearConjugateGradient.Context(x) -> ctx

Return a structure with all parameters and storage for temporary variables needed for
running the linear conjugate gradient algorithm. Argument `x` specifies the variables of the
problem and is used to allocate temporary variables by calling `similar(x)`. The returned
context holds no references on `x`. Algorithm parameters are specified by the following
keywords:

* `precond` is to specify whether to allocate temporary variables to store the
  preconditioned residuals. By default, `precond = false`. If false, only the
  un-preconditioned version of the algorithm can be run.

* `maxiters` is to specify the maximum number of iterations to perform which is practically
  unlimited by default.

* `restart` is to specify the number of consecutive iterations before restarting the
  conjugate gradient recurrence. Restarting the algorithm is to cope with the accumulation
  of rounding errors. By default, `restart = min(50,length(x)+1)`. Set `restart` to a value
  less or equal zero or greater than `maxiters` if you do not want that any restarts ever
  occur.

* `ftol = (fatol,frtol)` is to specify the absolute and relative tolerances for the function
  reduction. By default, `ftol = (0.0,1e-8)`.

* `gtol = (gatol,grtol)` is to specify the absolute and relative tolerances for stopping the
  algorithm based on the gradient of the objective function. Convergence occurs when the
  Mahalanobis norm of the residuals (which is that of the gradient of the associated
  objective function) is less or equal the largest of `gatol` and `grtol` times the
  Mahalanobis norm of the initial residuals. By default, `gtol = (0.0,1e-5)`.

* `xtol = (xatol,xrtol)` is to specify the absolute and relative tolerances for the change
  in variables. By default, `xtol = (0.0,1e-6)`.

Before running the algorithm with the same internal storage but but different parameters,
call `configure!`:

    OptimPack.configure!(ctx; kwds...)

"""
function Context(x::V; precond::Bool = false, kwds...) where {V}
    p = similar(x)
    q = similar(x)
    r = similar(x)
    z = (precond ? similar(x) : r)
    return Context(p, q, r, z; kwds...)
end

"""
    LinearConjugateGradient.solve!(ctx, A, b[, M], x; kwds...) -> ctx

Run the (preconditioned) linear conjugate gradient algorithm to solve the system of
equations `A*x = b` in `x` and according to the settings in `ctx`.

# Arguments

* `x` stores the initial solution on entry and the estimated solution on return.

* `A` implements the *left-hand-side (LHS) matrix* of the equations. It is used as
  `LinearAlgebra.mul!(dst,A,src)` to store in `dst` the result of applying `A` to `src` and
  where `src` and `dst` are similar to arguments `x` and `b`.

  FIXME Note that, as `A` and `M` must be symmetric, it may be faster to apply their adjoint.

* `b` is the *right-hand-side (RHS) vector* of the equations. It is left unchanged.

* `ctx` is a [`LinearConjugateGradient.Context`](@ref) structure storing all temporary
  variables and parameters of the algorithm. This argument is reusable and is required to
  avoid any additional allocations.

* Optional argument `M` is a preconditioner. If `M` is unspecified or if `M` is
  `OptimBase.Identity()`, the unpreconditioned version of the algorithm is run. The
  preconditioner can be specified in various forms (as for the LHS operator `A`).

# FIXME
Optional argument `io` is to specify an `IO` instance to which print various
information at each iterations. Nothing is printed if `io` is `devnull` which
is the default.


## Convergence criteria

Provided `A` be positive definite, the solution `x` of the equations `A*x = b`
is unique and is also the minimum of the following convex quadratic objective
function:

    f(x) = (1/2)*x'*A*x - b'*x + ϵ

where `ϵ` is an arbitrary constant. The gradient of this objective function is:

    ∇f(x) = A*x - b

hence solving `A*x = b` for `x` yields the minimum of `f(x)`. The variations of
`f(x)` between successive iterations, the norm of the gradient `∇f(x)`, or the
norm of the variation of variables `x` may be used to decide the convergence of
the algorithm (see keywords `ftol`, `gtol` and `xtol` below).

Let `x_{k}`, `f_{k} = f(x_{k})` and `∇f_{k} = ∇f(x_{k})` denote the variables,
the objective function and its gradient at iteration `k`. The argument `x`
gives the initial variables `x_{0}`. Starting with `k = 0`, the different
possibilities for the convergence of the algorithm are listed below.

* The convergence in the function reduction between successive iterations
  occurs at iteration `k ≥ 1` if:

  ```
  f_{k-1} - f_{k} ≤ max(fatol, frtol*max_{k' ≤ k}(f_{k'-1} - f_{k'}))
  ```

* The convergence in the gradient norm occurs at iteration `k ≥ 0` if:

  ```
  ‖∇f_{k}‖_M ≤ max(gatol, grtol*‖∇f_{0}‖_M)
  ```

  where `‖u‖_M = sqrt(u'*M*u)` is the Mahalanobis norm of `u` with precision
  matrix `M` which is equal to the usual Euclidean norm of `u` if no
  preconditioner is used or if `M` is the identity.

* The convergence in the variables occurs at iteration `k ≥ 1` if:

  ```
  ‖x_{k} - x_{k-1}‖ ≤ max(xatol, xrtol*‖x_{k}‖)
  ```

In the conjugate gradient algorithm, the objective function is always reduced
at each iteration, but be aware that the gradient and the change of variables
norms are not always reduced at each iteration.


## Returned Status

The returned value `status` is one of (in increasing order of their integer value):

- `:not_positive_definite` if the left-hand-side matrix `A` is found to be not positive
  definite;

- `:too_many_iterations` if the maximum number of iterations have been reached;

- `:f_test_satisfied` if convergence occurred because the function reduction satisfies the
  criterion specified by `ctx.ftol`;

- `:g_test_satisfied` if convergence occurred because the gradient norm satisfies the
  criterion specified by `ctx.gtol`;

- `:x_test_satisfied` if convergence occurred because the norm of the variation of variables
  satisfies the criterion specified by `ctx.xtol`.

Method [`LinearConjugateGradient.reason`](@ref) may be called to get a textual
explanation about the returned status. Method `issuccess(status)` may be called
to check whether algorithm has converged.

"""
function solve!(ctx::Context{V}, A, b::V, x::V; kwds...) where {V}
    # Run the unpreconditioned version of the algorithm.
    return solve!(ctx, A, b, Identity(), x; kwds...)
end

function solve!(ctx::Context{V}, A, b::V, M, x::V;
                observer=nothing, io::IO = devnull) where {V}
    # Get workspace variables.
    precond = !isa(M, Identity)
    p, q, r = ctx.p, ctx.q, ctx.r
    z = (precond ? ctx.z : ctx.r)
    if precond && z === r
        error("workspace variables Z must be different from R with ",
              "a preconditioner")
    end

    # Enforce types of some variables (FIXME: This will not work for BigFloat).
    local alpha::Float64, gamma::Float64
    local gtest::Float64

    # Initialize context.
    restart!(ctx, :searching)

    # Initialize local variables.
    t0 = time() # starting time
    xtest = (ctx.xtol[1] > 𝟘 || ctx.xtol[2] > 𝟘)
    oldrho = ctx.rho

    # Conjugate gradient iterations.
    while true
        # Is this the initial or a restarted iteration?
        restart = ctx.iterations == 0 || (ctx.restart > 0 && rem(ctx.iterations, ctx.restart) == 0)

        # Compute residuals and their squared norm.
        if restart
            # Compute residuals.
            if ctx.iterations > 0 || two_norm(x) != 𝟘
                # Compute r = b - A*x using r to temporarily store A*x.
                axpby!(r, 𝟙, b, -𝟙, mul!(r, A, x))
                ctx.evaluations += 1
            else
                # Spare applying A since x = 0.
                copyto!(r, b)
            end
        else
            # Update residuals.
            axpby!(r, 𝟙, r, -alpha, q) # r -= α⋅q
        end
        if precond
            # Apply preconditioner.
            mul!(z, M, r) # z = M*r
        end
        oldrho = ctx.rho
        ctx.rho = inner(r, z) # rho = ‖r‖²_M
        if ctx.iterations == 0
            gtest = tolerance(ctx.gtol, sqrt(ctx.rho))
        end
        if !isnothing(observer)
            ctx.elapsed = time() - t0
            observer(ctx)
        end
        if sqrt(ctx.rho) ≤ gtest
            # Normal convergence in the gradient norm.
            ctx.status = :g_test_satisfied
            break
        end
        if ctx.iterations ≥ ctx.maxiters
            # Too many iterations, give up.
            ctx.status = :too_many_iterations
            break
        end

        # Compute search direction.
        if restart
            # Restarting or first iteration.
            copyto!(p, z)
        else
            # Apply recurrence.
            beta = ctx.rho/oldrho
            axpby!(p, 𝟙, z, beta, p)
        end

        # Compute optimal step size.
        mul!(q, A, p) # q = A⋅p
        ctx.evaluations += 1
        gamma = inner(p, q) # γ = p'⋅A⋅p
        if !(gamma > zero(gamma))
            # Operator is not positive definite.
            ctx.status = :not_positive_definite
            break
        end
        alpha = ctx.rho/gamma

        # Update variables and check for convergence.
        axpby!(x, 𝟙, x, +alpha, p) # x += α⋅p
        ctx.psi = alpha*ctx.rho/2  # psi = f(x_{k}) - f(x_{k+1}) ≥ 0
        ctx.psimax = max(ctx.psi, ctx.psimax)
        if ctx.psi ≤ tolerance(ctx.ftol, ctx.psimax)
            # Normal convergence in the function reduction.
            ctx.status = :f_test_satisfied
            break
        end
        if xtest && alpha*two_norm(p) ≤ tolerance(ctx.xtol, x)
            # Normal convergence in the variables.
            ctx.status = :x_test_satisfied
            break
        end

        # Increment iteration number.
        ctx.iterations += 1
    end
    ctx.elapsed = time() - t0
    isnothing(observer) || observer(ctx)
    return ctx
end

"""
    tolerance(atol, rtol, val) -> max(0, atol, rtol*abs(val))
    tolerance(atol, rtol, arr) -> max(0, atol, rtol*two_norm(arr))

Given absolute and relative tolerances `atol` and `rtol` (both finite and nonnegative),
return a nonnegative tolerance for the scalar `val` or for the array `arr`.

If `rtol ≤ 0`, the computation of `two_norm(arr)` is not performed.

Absolute and relative tolerances, `atol` and `rtol`, may be specified as a 2-tuple
`(atol,rtol)`.

The result has at least double-precision.

"""
function tolerance((atol, rtol)::Tuple{Number,Real}, arg::Union{Number,AbstractArray})
    return tolerance(atol, rtol, arg)
end

function tolerance(atol::Number, rtol::Real, val::Number)
    T = get_precision(Float64, atol, rtol, val)
    tol = max(adapt_precision(T, atol), adapt_precision(T, rtol)*adapt_precision(T, val))
    return max(zero(tol), tol)
end

function tolerance(atol::Number, rtol::Real, arr::AbstractArray)
    T = float(eltype(arr))
    val = (rtol > 𝟘 ? convert(T, two_norm(arr))::T : zero(T))
    return tolerance(atol, rtol, val)
end

"""
    bad_argument(args...)

throws an `ArgumentError` exception with error message given by `args...`
converted into a string.

"""
bad_argument(msg::AbstractString) = throw(ArgumentError(msg))
@noinline bad_argument(args...) = bad_argument(string(args...))

observer(ctx::Context) = observer(stdout, ctx)
function observer(io::IO, ctx::Context)
    status = ctx.status
    if status === :searching
        t = ctx.elapsed*1E3 # elapsed time in milliseconds
        iter = ctx.iterations
        ∇f = sqrt(ctx.rho) # gradient norm ‖∇f(x)‖
        Δf = ctx.psi # variation of function Δf(x)
        if ctx.z !== ctx.r # FIXME ctx.precond
            iszero(iter) && print(
                io,
                "# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖     ‖∇f(x)‖_M\n",
                "# ---------------------------------------------------------\n")
            @printf io "%7d %11.3f %12.4e %12.4e %12.4e\n" iter t Δf two_norm(ctx.r) ∇f
        else
            iszero(iter) && print(
                io,
                "# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖\n",
                "# --------------------------------------------\n")
            @printf io "%7d %11.3f %12.4e %12.4e\n" iter t Δf ∇f
        end
    elseif status === :f_test_satisfied
        println(io, "# Convergence in the function reduction.")
    elseif status === :g_test_satisfied
        println(io, "# Convergence in the gradient norm.")
    elseif status === :x_test_satisfied
        println(io, "# Convergence in the variables.")
    elseif status === :too_many_iterations
        println(io, "# Too many iteration(s).")
    elseif status === :not_positive_definite
        println(io, "# Operator is not positive definite.")
    end
    return nothing
end

end # module

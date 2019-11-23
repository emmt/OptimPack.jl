#
# spg2.jl --
#
# Implements Spectral Projected Gradient Method (Version 2: "continuous
# projected gradient direction") to find the local minimizers of a given
# function with convex constraints, described in:
#
# [1] E. G. Birgin, J. M. Martinez, and M. Raydan, "Nonmonotone spectral
#     projected gradient methods on convex sets", SIAM Journal on Optimization
#     10, pp. 1196-1211 (2000).
#
# [2] E. G. Birgin, J. M. Martinez, and M. Raydan, "SPG: software for
#     convex-constrained optimization", ACM Transactions on Mathematical
#     Software (TOMS) 27, pp. 340-349 (2001).
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2014-2019: Éric Thiébaut.
#
# ----------------------------------------------------------------------------

module SPG

export
    spg2

using Printf
using ..OptimPack:
    DenseVariable,
    DenseVariableSpace,
    combine!,
    dot,
    norm2,
    norminf,
    wrap

mutable struct Info{T,N}
    x::Array{T,N}
    xbest::Array{T,N}
    f::Float64
    fbest::Float64
    pgtwon::Float64
    pginfn::Float64
    fcnt::Int
    pcnt::Int
    iter::Int
    status::Symbol
end

Info(x::DenseArray{T,N}, xbest::DenseArray{T,N}) where {T,N} =
    Info{T,N}(x, xbest, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, :SEARCHING)

"""
# Spectral Projected Gradient Method

The `spg2` method implements the Spectral Projected Gradient Method (Version 2:
"continuous projected gradient direction") to find the local minimizers of a
given function with convex constraints, described in the references below.  A
typical use is:

```julia
spg2(fg!, prj!, x0, m=10) -> res
```

The user must supply the functions `fg!` and `prj!` to evaluate the objective
function and its gradient and to project an arbitrary point onto the feasible
region.  These functions must be defined as in the following pseudo-code:

```julia
function fg!(x::T, g::T) where {T}
   g[:] = gradient_at(x)
   return function_value_at(x)
end

function prj!(dst::T, src::T) where {T}
    dst[:] = projection_of(src)
    return dst
end
```

For instance, to constrain the variables to be all nonnegative, the following
projector should be specified:

```julia
function prj!(dst::AbstractArray{T,N},
              src::AbstractArray{T,N}) where {T,N}
    @assert axes(dst) == axes(src)
    @inbounds @simd for i in eachindex(dst, src)
        dst[i] = max(src[i], zero(T))
    end
    return dst
end
```

Argument `x0` is the initial solution and argument `m` is the number of
previous function values to be considered in the nonmonotone line search.  If
`m ≤ 1`, then a monotone line search with Armijo-like stopping criterion will
be used.

The returned value `res` is an instance of `SPG.Info` storing information about
the final iterate.

The following keywords are available:

* `eps1` specifies the stopping criterion `‖pg‖_∞ ≤ eps1` with `pg` the
  projected gradient.  By default, `eps1 = 1e-6`.

* `eps2` specifies the stopping criterion `‖pg‖_2 ≤ eps2` with `pg` the
  projected gradient.  By default, `eps2 = 1e-6`.

* `eta` specifies a scaling parameter for the gradient.  The projected gradient
  is computed as:

  ```
  (x - prj(x - eta*g))/eta
  ```

  (with `g` the gradient at `x`) instead of `x - prj(x - g)` which corresponds
  to the default behavior (same as if `eta=1`) and is usually used in
  methodological publications although it does not scale correctly (for
  instance, if you make a change of variables or simply multiply the function
  by some factor).

* `ftol` specify the parameter of the nonmonotone Armijo-like stopping
  criterion.  By default: `ftol = 1e-4`.

* `lmin` and `lmax` specify safeguard bounds for the steplength.  By default,
  `lmin = 1e-30` and `lmax = 1e+30`.

* `amin` and `amax` specify safeguardbounds for the qiadratic interpolation.
   By default, `amin = 0.1` and `amax = 0.9`.

* `maxit` specifies the maximum number of iterations.

* `maxfc` specifies the maximum number of function evaluations.

* `verb` indicates whether to print some information at each iteration.

* `printer` specifies a subroutine to print some information at each iteration.
  This subroutine will be called as `printer(io, ws)` with `io` the output
  stream and `ws` an instance of `SPG.Info` with information about the current
  iterate.

* `io` specifes the output stream for iteration information.  It is `stdout` by
  default.

The `SPG.Info` type has the following members:

* `x` is the current iterate.
* `f` is the function value.
* `xbest` is the best solution so far.
* `fbest` is the function value at `xbest`.
* `pginfn` is the infinite norm of the projected gradient.
* `pgtwon` is the Eucliddean norm of the projected gradient.
* `iter` is the number of iterations.
* `fcnt` is the number of function (and gradient) evaluations.
* `pcnt` is the number of projections.
* `status` indicates the type of termination:

  | Status                  | Reason                                          |
  |:------------------------|:------------------------------------------------|
  | `:SEARCHING`            | Work in progress                                |
  | `:INFNORM_CONVERGENCE`  | Convergence of projected gradient infinite-norm |
  | `:TWONORM_CONVERGENCE`  | Convergence of projected gradient 2-norm        |
  | `:TOO_MANY_ITERATIONS`  | Too many iterations                             |
  | `:TOO_MANY_EVALUATIONS` | Too many function evaluations                   |


## References

* E. G. Birgin, J. M. Martinez, and M. Raydan, "Nonmonotone spectral projected
  gradient methods on convex sets", SIAM Journal on Optimization 10,
  pp. 1196-1211 (2000).

* E. G. Birgin, J. M. Martinez, and M. Raydan, "SPG: software for
  convex-constrained optimization", ACM Transactions on Mathematical Software
  (TOMS) 27, pp. 340-349 (2001).
"""
function spg2(fg!::Function,
              prj!::Function,
              x0::AbstractArray{T,N},
              m::Integer = 10;
              maxit::Integer = typemax(Int),
              maxfc::Integer = typemax(Int),
              eps1::Real = 1e-6,
              eps2::Real = 1e-6,
              eta::Real = 1.0,
              ftol::Real = 1e-4,
              lmin::Real = 1e-30,
              lmax::Real = 1e+30,
              amin::Real = 0.1,
              amax::Real = 0.9,
              printer = nothing,
              io::IO = stdout,
              verb::Bool = false) where {T<:AbstractFloat,N}
    x = copyto!(Array{T}(undef, size(x0)), x0)
    _spg!(fg!, prj!, x, Int(m), Int(maxit), Int(maxfc),
          Float64(eps1), Float64(eps2), Float64(eta), Float64(ftol),
          Float64(lmin), Float64(lmax), Float64(amin), Float64(amax),
          printer, io, verb)
end

function _spg!(fg!::Function,
               prj!::Function,
               x::Array{T,N},
               m::Int,
               maxit::Int,
               maxfc::Int,
               eps1::Float64,
               eps2::Float64,
               eta::Float64,
               ftol::Float64,
               lmin::Float64,
               lmax::Float64,
               amin::Float64,
               amax::Float64,
               printer,
               io::IO,
               verb::Bool) where {T<:AbstractFloat,N}

    # Allocate workspace.
    dims = size(x)
    space = DenseVariableSpace(T, dims)
    d  = Array{T}(undef, dims)
    g  = Array{T}(undef, dims)
    g  = Array{T}(undef, dims)
    x0 = Array{T}(undef, dims)
    g0 = Array{T}(undef, dims)
    xbest = Array{T}(undef, dims)

    # Wrap OptimPack variables over Julia workspace arrays.
    Vx = wrap(space, x)
    Vg = wrap(space, g)
    Vd = wrap(space, d)
    Vx0 = wrap(space, x0)
    Vg0 = wrap(space, g0)

    # Initialization.
    ws = Info(x, xbest)
    local sty, sts, f0, f
    if m > 1
        lastfv = Array{T}(undef, m)
        fill!(lastfv, T(Inf))
    else
        lastfv = nothing
    end

    # Project initial guess.
    prj!(x, x)
    copyto!(x0, x)
    ws.pcnt += 1

    # Evaluate function and gradient.
    f = fg!(x, g)
    ws.fcnt += 1

    # Initialize best solution and best function value.
    ws.fbest = f
    copyto!(xbest, x)

    # Main loop.
    while true

        # Compute continuous projected gradient (and its norms)
        # as: `pg = (x - prj(x - eta*g))/eta`.
        combine!(Vd, 1, Vx, -eta, Vg)
        prj!(d, d)
        combine!(Vd, 1/eta, Vx, -1/eta, Vd)
        ws.pcnt += 1
        ws.pgtwon = norm2(Vd)
        ws.pginfn = norminf(Vd)

        # Print iteration information
        if printer !== nothing
            printer(io, ws)
        end
        if verb
            @printf("ITER = %-5d  EVAL = %-5d  PROJ = %-5d  F(%s) =%24.17e  ||PG||oo = %17.10e\n",
                    ws.iter, ws.fcnt, ws.pcnt, (f ≤ ws.fbest ? "+" : "-"), f, ws.pginfn)
        end

        # Test stopping criteria.
        if ws.pginfn ≤ eps1
            # Gradient infinite-norm stopping criterion satisfied, stop.
            ws.status = :INFNORM_CONVERGENCE
            return ws
        end
        if ws.pgtwon ≤ eps2
            # Gradient 2-norm stopping criterion satisfied, stop.
            ws.status = :TWONORM_CONVERGENCE
            return ws
        end
        if ws.iter ≥ maxit
            # Maximum number of iterations exceeded, stop.
            ws.status = :TOO_MANY_ITERATIONS
            return ws
        end
        if ws.fcnt ≥ maxfc
            # Maximum number of function evaluations exceeded, stop.
            ws.status = :TOO_MANY_EVALUATIONS
            return ws
        end

        # Store function value for the nonmonotone line search and
        # find maximum function value since m last calls.
        if m > 1
            lastfv[(ws.iter%m) + 1] = f
            fmax = maximum(lastfv)
        else
            fmax = f
        end

        # Compute spectral steplength.
        if ws.iter == 0
            # Initial steplength. (FIXME: check type stability)
            lambda = min(lmax, max(lmin, 1/ws.pginfn))
        else
            Vs = Vx0 # alias
            Vy = Vg0 # alias
            combine!(Vs, 1, Vx, -1, Vx0)
            combine!(Vy, 1, Vg, -1, Vg0)
            sty = dot(Vs, Vy)
            if sty > 0
                # Safeguarded Barzilai & Borwein spectral steplength.
                sts = dot(Vs, Vs)
                lambda = min(lmax, max(lmin, sts/sty))
            else
                lambda = lmax
            end
        end

        # Save current point.
        copyto!(x0, x)
        copyto!(g0, g)
        f0 = f

        # Compute the spectral projected gradient direction and <G,D>
        combine!(Vx, 1, Vx, -lambda, Vg)
        prj!(x, x)
        ws.pcnt += 1
        combine!(Vd, 1, Vx, -1, Vx0) # d = x - x0
        delta = dot(Vg0, Vd)

        # Nonmonotone line search.
        stp = 1.0 # Step length for first trial.
        while true
            # Evaluate function and gradient at trial point.
            f = fg!(x, g)
            ws.fcnt += 1

            # Compare the new function value against the best function
            # value and, if smaller, update the best function value and the
            # corresponding best point.
            if f < ws.fbest
                ws.fbest = f
                copyto!(xbest, x)
            end

            # Test stopping criteria.
            if f ≤ fmax + stp*ftol*delta
                # Nonmonotone Armijo-like stopping criterion satisfied, stop.
                break
            end
            if ws.fcnt ≥ maxfc
                # Maximum number of function evaluations exceeded, stop.
                ws.status = :TOO_MANY_EVALUATIONS
                return ws
            end

            # Safeguarded quadratic interpolation.
            q = -delta*(stp*stp)
            r = (f - f0 - stp*delta)*2
            if r > 0 && amin*r ≤ q ≤ amax*stp*r
                stp = q/r
            else
                stp /= 2
            end

            # Compute trial point.
            combine!(Vx, 1, Vx0, stp, Vd) # x = x0 + stp*d
        end

        if ws.status != :SEARCHING
            # The number of function evaluations was exceeded inside the line
            # search.
            return ws
        end

        # Proceed with next iteration.
        ws.iter += 1

    end
end

end # module
